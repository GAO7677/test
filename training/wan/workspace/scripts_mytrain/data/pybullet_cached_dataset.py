from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from torch.utils.data import Dataset


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_split(logical_key: str) -> str:
    digest = hashlib.sha1(logical_key.encode("utf-8")).hexdigest()
    value = int(digest[:12], 16) / float(16**12 - 1)
    if value < 0.90:
        return "train"
    if value < 0.95:
        return "val"
    return "test"


def _sample_list_records(
    sample_list: str | Path,
    *,
    case_prompt_variant: str = "prompt_physics_consistency_v1",
    caption_field: str | None = None,
    prompt_cache_dir: str | Path | None = None,
) -> list[dict]:
    """Read an explicit list of sample directories for a train run.

    The list is intentionally one sample directory per line.  This keeps the
    selection policy outside the training manifest while deriving the same
    logical key used by the cache builders: ``<group>/<sample_dir_name>``.
    """
    sample_list_path = Path(sample_list).expanduser().resolve()
    if not sample_list_path.is_file():
        raise FileNotFoundError(f"Sample list does not exist: {sample_list_path}")

    records: list[dict] = []
    seen_keys: set[str] = set()
    seen_paths: set[Path] = set()
    if caption_field:
        if prompt_cache_dir is None:
            raise ValueError("caption_field requires an explicit prompt_cache_dir")
        override_root = Path(prompt_cache_dir).expanduser().resolve()
        override_config = CachedPyBulletDataset._load_cache_config(override_root)
        override_entries = CachedPyBulletDataset._index_by_key(
            override_root / override_config.get("index_file", "index.jsonl"), "logical_key"
        )
        override_texts = CachedPyBulletDataset._index_by_key(
            override_root / override_config.get("prompt_index_file", "prompt_index.jsonl"), "prompt_sha256"
        )
    # A selection directory may carry a caption overlay manifest while the
    # list itself still contains the original source sample directories.  Use
    # that overlay when present, but keep metadata.json as the fallback so
    # older lists behave exactly as before.
    overlay_captions: dict[str, str] = {}
    overlay_manifest_path = sample_list_path.parent / "manifest.json"
    if overlay_manifest_path.is_file():
        manifest = json.loads(overlay_manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, list):
            raise ValueError(f"Expected a list in {overlay_manifest_path}")
        for row in manifest:
            if not isinstance(row, dict):
                raise ValueError(f"Manifest rows must be objects: {overlay_manifest_path}")
            family_key = str(row.get("family_key") or "").strip()
            case_id = str(row.get("case_id") or "").strip()
            caption = str(row.get("caption") or row.get("short_caption") or "").strip()
            if family_key and case_id and caption:
                overlay_captions[f"{family_key}/{case_id}"] = caption
    for line_number, line in enumerate(
        sample_list_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        index_path = Path(value).expanduser()
        if index_path.suffix.lower() == ".json":
            if not index_path.is_absolute():
                index_path = sample_list_path.parent / index_path
            index_path = index_path.resolve()
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            source = payload.get("source", {})
            cache = payload.get("cache", {})
            logical_key = str(payload.get("logical_key") or "").strip()
            if not logical_key:
                raise ValueError(f"Missing logical_key: {index_path}")
            if logical_key in seen_keys:
                raise ValueError(f"Duplicate logical key in sample list: {logical_key}")
            if caption_field:
                metadata_value = source.get("metadata_json_resolved") or source.get("metadata_json")
                if not metadata_value:
                    raise ValueError(f"Missing source metadata path: {index_path}")
                metadata_path = Path(metadata_value).expanduser()
                if not metadata_path.is_absolute():
                    metadata_path = index_path.parent / metadata_path
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                caption = metadata
                for component in caption_field.split("."):
                    caption = caption.get(component) if isinstance(caption, dict) else None
                if not isinstance(caption, str) or not caption.strip():
                    raise ValueError(f"Missing or empty {caption_field}: {metadata_path}")
                caption = caption.strip()
                caption_sha = _text_sha256(caption)
                entry = override_entries.get(logical_key)
                text_entry = override_texts.get(caption_sha)
                if (entry is None or entry.get("positive_prompt") != caption_sha
                        or entry.get("caption_sha256") != caption_sha
                        or text_entry is None or text_entry.get("prompt") != caption):
                    raise ValueError(f"Metadata caption / prompt cache mismatch: {logical_key}")
                prompt = {
                    "prompt": caption,
                    "cache_config": str(override_root / "cache_config.json"),
                    "tensor": {"path": str(override_root / entry["embedding_file"])},
                }
            else:
                prompt = cache.get(case_prompt_variant)
                if prompt is None and case_prompt_variant == "base":
                    prompt = cache.get("prompt_base")
            if prompt is None:
                raise ValueError(
                    f"Prompt variant {case_prompt_variant!r} is absent in {index_path}"
                )
            vae = cache.get("vae", {})
            vae_config_path = Path(str(vae.get("cache_config"))).expanduser().resolve()
            prompt_config_path = Path(str(prompt.get("cache_config"))).expanduser().resolve()
            vae_config = json.loads(vae_config_path.read_text(encoding="utf-8"))
            prompt_config = json.loads(prompt_config_path.read_text(encoding="utf-8"))
            source_video = source.get("video_resolved") or source.get("video")
            vae_tensor = vae.get("tensor", {})
            prompt_tensor = prompt.get("tensor", {})
            record = {
                "logical_key": logical_key,
                "caption": str(prompt.get("prompt") or source.get("caption") or "").strip(),
                "sample_path": str(source.get("case_root") or source.get("dataset_root") or ""),
                "case_json": str(index_path),
                "case_index_mode": True,
                "video_path": str(source_video or ""),
                "case_vae_tensor": str(Path(str(vae_tensor.get("path"))).expanduser().resolve()),
                "case_vae_tensor_key": vae_config.get("latent", {}).get("tensor_key", "vae_latent"),
                "case_vae_shape": vae_config.get("latent", {}).get("shape"),
                "case_vae_dtype": vae_config.get("latent", {}).get("dtype"),
                "case_prompt_tensor": str(Path(str(prompt_tensor.get("path"))).expanduser().resolve()),
                "case_prompt_tensor_key": prompt_config.get("embedding", {}).get("tensor_key", "prompt_embedding"),
                "case_prompt_shape": prompt_config.get("embedding", {}).get("shape"),
                "case_prompt_dtype": prompt_config.get("embedding", {}).get("dtype"),
                "case_vae_cache_config": str(vae_config_path),
                "case_prompt_cache_config": str(prompt_config_path),
            }
            if not record["caption"]:
                raise ValueError(f"Empty caption in {index_path}")
            if caption_field:
                record.update(
                    caption_field=caption_field,
                    caption_metadata_json=str(metadata_path.resolve()),
                    caption_sha256=caption_sha,
                    prompt_encoding_id=override_config["encoding_id"],
                )
            seen_keys.add(logical_key)
            records.append(record)
            continue
        sample_dir = Path(value).expanduser()
        if caption_field:
            raise ValueError("caption_field currently requires a sample_list of case JSON paths")
        if not sample_dir.is_absolute():
            sample_dir = sample_list_path.parent / sample_dir
        sample_dir = sample_dir.resolve()
        if sample_dir in seen_paths:
            raise ValueError(f"Duplicate sample path at {sample_list_path}:{line_number}")
        seen_paths.add(sample_dir)
        metadata_path = sample_dir / "metadata.json"
        video_path = sample_dir / "video.mp4"
        if not sample_dir.is_dir() or not metadata_path.is_file() or not video_path.is_file():
            raise ValueError(
                f"Sample list entry must contain metadata.json and video.mp4: "
                f"{sample_dir} (line {line_number})"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        try:
            text_block = metadata["object_identity"]["text"]
            caption = str(
                text_block.get("new_caption") or text_block.get("caption") or ""
            ).strip()
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError(
                f"Missing object_identity.text.new_caption/caption: {metadata_path}"
            ) from error
        if not caption:
            raise ValueError(f"Empty caption: {metadata_path}")
        group = sample_dir.parent.name.strip()
        if not group:
            raise ValueError(f"Cannot infer dataset group from sample path: {sample_dir}")
        logical_key = f"{group}/{sample_dir.name}"
        if logical_key in seen_keys:
            raise ValueError(f"Duplicate logical key in sample list: {logical_key}")
        seen_keys.add(logical_key)
        caption = overlay_captions.get(logical_key, caption)
        records.append(
            {
                "logical_key": logical_key,
                "caption": caption,
                "sample_path": str(sample_dir),
            }
        )
    if not records:
        raise ValueError(f"No sample paths found in {sample_list_path}")
    return records


class CachedPyBulletDataset(Dataset):
    """Read cached PyBullet samples, optionally selected by a path list."""

    def __init__(
        self,
        *,
        root: str | Path,
        vae_cache_dir: str | Path,
        prompt_cache_dir: str | Path,
        split: str = "train",
        height: int = 512,
        width: int = 896,
        num_frames: int = 49,
        prompt_cache_use_cache_text: bool = False,
        sample_list: str | Path | None = None,
        case_prompt_variant: str = "prompt_physics_consistency_v1",
        caption_field: str | None = None,
        dataset_sources: str | Path | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.vae_cache_dir = Path(vae_cache_dir).expanduser().resolve()
        self.prompt_cache_dir = Path(prompt_cache_dir).expanduser().resolve()
        self.sample_list = (
            None if sample_list is None else Path(sample_list).expanduser().resolve()
        )
        self.split = split
        self.prompt_cache_use_cache_text = bool(prompt_cache_use_cache_text)
        self.caption_field = caption_field
        self.dataset_sources = None if dataset_sources is None else Path(dataset_sources).expanduser().resolve()
        self.source_counts: dict[str, int] = {}
        if self.dataset_sources and (self.sample_list is not None or caption_field or self.prompt_cache_use_cache_text):
            raise ValueError("dataset_sources cannot be combined with sample_list, caption_field, or prompt_cache_use_cache_text")
        if caption_field and (self.sample_list is None or self.prompt_cache_use_cache_text):
            raise ValueError("caption_field requires sample_list and strict cache-text validation (no prompt_cache_use_cache_text)")
        self.expected_latent_shape = (
            48,
            (num_frames - 1) // 4 + 1,
            height // 16,
            width // 16,
        )
        self.expected_prompt_shape = (512, 4096)

        if split not in {"train", "val", "test", "all"}:
            raise ValueError(f"Unsupported split: {split}")
        if (num_frames - 1) % 4:
            raise ValueError("num_frames must satisfy 4n+1")

        if self.dataset_sources is not None:
            if split not in {"train", "all"}:
                raise ValueError("dataset_sources uses explicit training selections; use split=train or all")
            sources = json.loads(self.dataset_sources.read_text(encoding="utf-8"))
            if not isinstance(sources, list) or not sources:
                raise ValueError("dataset_sources must contain a nonempty list of source configurations")
            records = []
            seen = set()
            for source in sources:
                name = source["name"]
                if not isinstance(name, str) or not name or name in self.source_counts:
                    raise ValueError(f"Invalid or duplicate dataset source name: {name}")
                if not source.get("caption_field"):
                    raise ValueError(f"Missing explicit caption_field for dataset source: {name}")
                paths = {}
                for key in ("sample_list", "prompt_cache_dir"):
                    path = Path(source[key]).expanduser()
                    paths[key] = path if path.is_absolute() else self.dataset_sources.parent / path
                selected = _sample_list_records(
                    paths["sample_list"],
                    case_prompt_variant=source.get("case_prompt_variant", "base"),
                    caption_field=source["caption_field"],
                    prompt_cache_dir=paths["prompt_cache_dir"],
                )
                if "expected_samples" in source and len(selected) != source["expected_samples"]:
                    raise ValueError(f"Unexpected sample count for source {name}: {len(selected)}")
                for record in selected:
                    if record["logical_key"] in seen:
                        raise ValueError(f"Duplicate logical key across dataset sources: {record['logical_key']}")
                    seen.add(record["logical_key"])
                    record["dataset_source"] = name
                self.source_counts[name] = len(selected)
                records.extend(selected)
        elif self.sample_list is not None:
            if split not in {"train", "all"}:
                raise ValueError(
                    "An explicit sample_list is a train selection; use split=train or split=all"
                )
            # The list is already the deterministic selection (asset + top
            # generic), so do not apply the generic SHA1 train/val/test split.
            records = _sample_list_records(
                self.sample_list,
                case_prompt_variant=case_prompt_variant,
                caption_field=caption_field,
                prompt_cache_dir=self.prompt_cache_dir,
            )
        else:
            manifest_path = self.root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, list):
                raise ValueError(f"Expected a list in {manifest_path}")

            records = []
            for row in manifest:
                logical_key = f"{row['family_key']}/{row['case_id']}"
                if split != "all" and _stable_split(logical_key) != split:
                    continue
                caption = str(row.get("caption") or row.get("short_caption") or "").strip()
                if not caption:
                    raise ValueError(f"Missing caption for {logical_key}")
                records.append({"logical_key": logical_key, "caption": caption})
        if not records:
            raise ValueError(f"No PyBullet samples found for split={split}")

        self.case_index_mode = bool(records[0].get("case_index_mode", False))
        if any(bool(record.get("case_index_mode", False)) != self.case_index_mode for record in records):
            raise ValueError("Cannot mix case-sidecar JSON paths with sample directories in one sample list")
        if self.case_index_mode:
            self._initialize_case_index_cache(records, height=height, width=width, num_frames=num_frames)
            self.records = records
            return

        vae_config = self._load_cache_config(self.vae_cache_dir)
        preprocess = vae_config.get("preprocess", {})
        # Older cache builders recorded the temporal policy as ``window_mode``;
        # newer dataset manifests call the same field ``sampling_strategy``.
        # Accept both spellings so an otherwise identical prefix cache is not
        # rejected solely because it was produced by the existing builder.
        sampling_strategy = preprocess.get(
            "sampling_strategy",
            preprocess.get("window_mode"),
        )
        actual_preprocess = (
            preprocess.get("height"),
            preprocess.get("width"),
            preprocess.get("num_frames"),
            sampling_strategy,
        )
        expected_preprocess = (height, width, num_frames, "prefix")
        if actual_preprocess != expected_preprocess:
            raise ValueError(
                f"VAE cache preprocessing mismatch: {actual_preprocess} != {expected_preprocess}"
            )
        latent_config = vae_config.get("latent", {})
        if tuple(latent_config.get("shape", ())) != self.expected_latent_shape:
            raise ValueError(
                "VAE cache latent shape mismatch: "
                f"{latent_config.get('shape')} != {self.expected_latent_shape}"
            )
        self.vae_tensor_key = latent_config.get("tensor_key", "vae_latent")
        self.vae_dtype = self._cache_dtype(latent_config.get("dtype"), "VAE")
        self.vae_entries = self._index_by_key(
            self.vae_cache_dir / vae_config.get("index_file", "index.jsonl"),
            "logical_key",
        )

        prompt_config = self._load_cache_config(self.prompt_cache_dir)
        embedding_config = prompt_config.get("embedding", {})
        if tuple(embedding_config.get("shape", ())) != self.expected_prompt_shape:
            raise ValueError(
                "Prompt cache embedding shape mismatch: "
                f"{embedding_config.get('shape')} != {self.expected_prompt_shape}"
            )
        self.prompt_tensor_key = embedding_config.get("tensor_key", "prompt_embedding")
        self.prompt_dtype = self._cache_dtype(embedding_config.get("dtype"), "prompt")
        self.prompt_entries = self._index_by_key(
            self.prompt_cache_dir / prompt_config.get("index_file", "index.jsonl"),
            "logical_key",
        )
        self.prompt_files = self._index_by_key(
            self.prompt_cache_dir
            / prompt_config.get("prompt_index_file", "prompt_index.jsonl"),
            "prompt_sha256",
        )

        missing_vae = []
        missing_prompt = []
        mismatched_prompt = []
        for record in records:
            logical_key = record["logical_key"]
            if logical_key not in self.vae_entries:
                missing_vae.append(logical_key)
            prompt_entry = self.prompt_entries.get(logical_key)
            if prompt_entry is None:
                missing_prompt.append(logical_key)
                continue
            manifest_prompt_sha = _text_sha256(record["caption"])
            cache_prompt_sha = prompt_entry.get("positive_prompt")
            if self.prompt_cache_use_cache_text:
                if cache_prompt_sha not in self.prompt_files:
                    missing_prompt.append(logical_key)
                    continue
                cache_prompt = self.prompt_files[cache_prompt_sha].get("prompt")
                if not isinstance(cache_prompt, str) or not cache_prompt.strip():
                    raise ValueError(
                        f"Prompt cache entry has no prompt text: {logical_key}"
                    )
                # This mode is explicit: the cache's positive prompt is the
                # training caption, even when the current dataset manifest
                # was generated before a prompt suffix was added.
                record["caption"] = cache_prompt
            elif cache_prompt_sha != manifest_prompt_sha:
                mismatched_prompt.append(logical_key)
            elif manifest_prompt_sha not in self.prompt_files:
                missing_prompt.append(logical_key)
        if missing_vae:
            raise ValueError(f"Missing VAE cache entries: {missing_vae[:5]}")
        if missing_prompt:
            raise ValueError(f"Missing prompt cache entries: {missing_prompt[:5]}")
        if mismatched_prompt:
            raise ValueError(f"Prompt cache caption mismatch: {mismatched_prompt[:5]}")

        self.records = records

    def _initialize_case_index_cache(self, records, *, height: int, width: int, num_frames: int):
        expected_latent_shape = self.expected_latent_shape
        expected_prompt_shape = self.expected_prompt_shape
        self.vae_dtype = self._cache_dtype(records[0].get("case_vae_dtype"), "VAE")
        self.prompt_dtype = self._cache_dtype(records[0].get("case_prompt_dtype"), "prompt")
        for record in records:
            if (self._cache_dtype(record.get("case_vae_dtype"), "VAE") != self.vae_dtype
                    or self._cache_dtype(record.get("case_prompt_dtype"), "prompt") != self.prompt_dtype):
                raise ValueError(f"Incompatible cache dtypes for {record['logical_key']}")
            if not Path(record["video_path"]).is_file():
                raise ValueError(f"Missing source video in case sidecar: {record['case_json']}")
            if not Path(record["case_vae_tensor"]).is_file():
                raise ValueError(
                    f"Missing VAE tensor; cache generation is incomplete: {record['case_vae_tensor']}"
                )
            if not Path(record["case_prompt_tensor"]).is_file():
                raise ValueError(
                    f"Missing prompt tensor; cache generation is incomplete: {record['case_prompt_tensor']}"
                )
            if record.get("caption_field"):
                with safe_open(record["case_prompt_tensor"], framework="pt", device="cpu") as handle:
                    tensor_metadata = handle.metadata() or {}
                expected = {
                    "logical_key": record["logical_key"],
                    "caption_sha256": record["caption_sha256"],
                    "encoding_id": record["prompt_encoding_id"],
                }
                if any(tensor_metadata.get(key) != value for key, value in expected.items()):
                    raise ValueError(f"Prompt tensor metadata mismatch: {record['logical_key']}")
            if list(record.get("case_vae_shape") or []) != list(expected_latent_shape):
                raise ValueError(
                    f"Case VAE shape mismatch for {record['logical_key']}: "
                    f"{record.get('case_vae_shape')} != {list(expected_latent_shape)}"
                )
            if list(record.get("case_prompt_shape") or []) != list(expected_prompt_shape):
                raise ValueError(
                    f"Case prompt shape mismatch for {record['logical_key']}: "
                    f"{record.get('case_prompt_shape')} != {list(expected_prompt_shape)}"
                )
        self.vae_tensor_key = records[0]["case_vae_tensor_key"]
        self.prompt_tensor_key = records[0]["case_prompt_tensor_key"]

    @staticmethod
    def _load_cache_config(cache_dir: Path) -> dict:
        config_path = cache_dir / "cache_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("status") != "complete":
            raise ValueError(f"Cache is not complete: {config_path}")
        return config

    @staticmethod
    def _cache_dtype(dtype_name: str | None, label: str) -> torch.dtype:
        dtypes = {"float32": torch.float32, "bfloat16": torch.bfloat16}
        if dtype_name not in dtypes:
            raise ValueError(f"Unsupported {label} cache dtype: {dtype_name}")
        return dtypes[dtype_name]

    @staticmethod
    def _index_by_key(path: Path, key: str) -> dict[str, dict]:
        entries = {}
        for entry in _read_jsonl(path):
            value = str(entry[key])
            if value in entries:
                raise ValueError(f"Duplicate {key}={value} in {path}")
            entries[value] = entry
        return entries

    @staticmethod
    def _load_tensor(
        path: Path,
        tensor_key: str,
        expected_shape: tuple[int, ...],
        expected_dtype: torch.dtype,
    ) -> torch.Tensor:
        with safe_open(path, framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(tensor_key)
        if tuple(tensor.shape) != expected_shape or tensor.dtype != expected_dtype:
            raise ValueError(
                f"Invalid cached tensor {path}: shape={tuple(tensor.shape)} "
                f"dtype={tensor.dtype}, expected={expected_shape} {expected_dtype}"
            )
        return tensor

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        logical_key = record["logical_key"]
        if self.case_index_mode:
            return {
                "logical_key": logical_key,
                "latents": self._load_tensor(
                    Path(record["case_vae_tensor"]),
                    record["case_vae_tensor_key"],
                    self.expected_latent_shape,
                    self.vae_dtype,
                ),
                "prompt_embeds": self._load_tensor(
                    Path(record["case_prompt_tensor"]),
                    record["case_prompt_tensor_key"],
                    self.expected_prompt_shape,
                    self.prompt_dtype,
                ),
            }
        vae_entry = self.vae_entries[logical_key]
        prompt_sha = self.prompt_entries[logical_key]["positive_prompt"]
        prompt_entry = self.prompt_files[prompt_sha]
        return {
            "logical_key": logical_key,
            "latents": self._load_tensor(
                self.vae_cache_dir / vae_entry["latent_file"],
                self.vae_tensor_key,
                self.expected_latent_shape,
                self.vae_dtype,
            ),
            "prompt_embeds": self._load_tensor(
                self.prompt_cache_dir / prompt_entry["embedding_file"],
                self.prompt_tensor_key,
                self.expected_prompt_shape,
                self.prompt_dtype,
            ),
        }

    @staticmethod
    def collate_fn(samples: list[dict]) -> dict:
        return {
            "logical_keys": [sample["logical_key"] for sample in samples],
            "latents": torch.stack([sample["latents"] for sample in samples]),
            "prompt_embeds": torch.stack(
                [sample["prompt_embeds"] for sample in samples]
            ),
        }

    @property
    def stats(self) -> dict:
        families = {}
        for record in self.records:
            family = record["logical_key"].split("/", 1)[0]
            families[family] = families.get(family, 0) + 1
        return {
            "root": str(self.root),
            "split": self.split,
            "sample_list": None if self.sample_list is None else str(self.sample_list),
            "explicit_sample_selection": self.sample_list is not None or self.dataset_sources is not None,
            "case_index_mode": self.case_index_mode,
            "num_samples": len(self.records),
            "families": families,
            "latent_shape": list(self.expected_latent_shape),
            "prompt_shape": list(self.expected_prompt_shape),
            "vae_cache_dir": str(self.vae_cache_dir),
            "prompt_cache_dir": str(self.prompt_cache_dir),
            "prompt_cache_use_cache_text": self.prompt_cache_use_cache_text,
            "caption_field": self.caption_field,
            "dataset_sources": None if self.dataset_sources is None else str(self.dataset_sources),
            "source_counts": self.source_counts,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate cached PyBullet training data")
    parser.add_argument("--pybullet_root", required=True)
    parser.add_argument("--vae_cache_dir", required=True)
    parser.add_argument("--prompt_cache_dir", required=True)
    parser.add_argument("--sample_list", default=None)
    parser.add_argument("--caption_field", default=None)
    parser.add_argument("--dataset_sources", default=None)
    parser.add_argument("--split", default="train", choices=("train", "val", "test", "all"))
    args = parser.parse_args()
    dataset = CachedPyBulletDataset(
        root=args.pybullet_root,
        vae_cache_dir=args.vae_cache_dir,
        prompt_cache_dir=args.prompt_cache_dir,
        sample_list=args.sample_list,
        caption_field=args.caption_field,
        dataset_sources=args.dataset_sources,
        split=args.split,
    )
    first = dataset[0]
    print(json.dumps(dataset.stats, indent=2, ensure_ascii=True, sort_keys=True))
    print(
        f"first={first['logical_key']} latents={tuple(first['latents'].shape)} "
        f"{first['latents'].dtype} prompt={tuple(first['prompt_embeds'].shape)} "
        f"{first['prompt_embeds'].dtype}"
    )


if __name__ == "__main__":
    main()
