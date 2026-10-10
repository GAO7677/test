#!/usr/bin/env python3
"""Run Wan2.2 + an optional strict PhysRVG DiT + a Full-SA LoRA on JSON lists."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
import torchvision.transforms.functional as transform_functional
from accelerate.utils import set_seed
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
from peft import PeftModel
from PIL import Image
from safetensors.torch import load_file
from torchvision.transforms import InterpolationMode

from fastvideo.models.wan_v2v.model_wan_v2v import WanTransformer3DModel
from fastvideo.models.wan_v2v.pipeline_wan_v2v import WanImageToVideoPipeline
import fastvideo_xssc_object as xssc_object


NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)

# This suffix is intentionally kept here as a detection fallback rather than
# unconditionally appended to every inference prompt.  Some training caches
# encode the suffix directly in prompt_index.jsonl while leaving
# cache_config.json:positive_prompt_suffix empty.
PHYSICS_CONSISTENCY_SUFFIX = (
    "Follow physically plausible dynamics and maintain temporal consistency. "
    "Keep object identity and geometry consistent over time. "
    "Do not introduce, remove, teleport, or apply impossible motion to objects."
)


def parse_args(default_context_noise_mode: str = "clean") -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a JSON list with Wan2.2, an optional PhysRVG DiT, and a PEFT adapter."
    )
    parser.add_argument("--input-json-list", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--context-noise-mode", choices=("clean",)  # inferv2 noisy context disabled by user
       ,
        default=default_context_noise_mode,
        help="Rebuild context from clean VAE latents and fixed noise at each scheduler sigma.",
    )
    parser.add_argument("--model-id", type=Path, required=True)
    parser.add_argument(
        "--physrvg-dit-checkpoint",
        type=Path,
        default=None,
        help="Optional full PhysRVG DiT checkpoint, loaded with strict=True before LoRA.",
    )
    parser.add_argument(
        "--lora-checkpoint",
        type=Path,
        default=None,
        help="Optional PEFT LoRA checkpoint. Omit for the LoRA-OFF baseline.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=NEGATIVE_PROMPT,
        help="Negative prompt passed to the pipeline for this inference profile.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help=(
            "Optional prompt override applied to every case. If omitted, the "
            "case input_caption is used."
        ),
    )
    parser.add_argument(
        "--prompt-cache-dir",
        type=Path,
        default=None,
        help=(
            "Optional training prompt cache used to resolve whether a prompt "
            "suffix was part of the training captions."
        ),
    )
    parser.add_argument(
        "--prompt-suffix-mode",
        choices=("auto", "never"),
        default="auto",
        help=(
            "In auto mode, append a suffix only when the supplied training "
            "cache's positive prompts contain one."
        ),
    )
    parser.add_argument(
        "--do-cfg",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable classifier-free guidance in the Wan pipeline.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rng-mode",
        choices=("explicit_generator", "global_seed_per_case"),
        default="explicit_generator",
        help=(
            "Randomness source: explicit_generator uses a per-case torch.Generator; "
            "global_seed_per_case matches PhysRVG-72f-adapted by resetting all "
            "global RNGs before each case and passing no explicit generator."
        ),
    )
    parser.add_argument(
        "--context-frames",
        type=int,
        default=None,
        help="Use only the first K decoded context frames; omitted keeps all frames.",
    )
    parser.add_argument(
        "--context-mask-mode",
        choices=("official_fixed2", "dynamic_effective"),
        default="official_fixed2",
        help=(
            "Condition mask policy: official_fixed2 keeps the trained first two "
            "latent frames; dynamic_effective conditions all latent frames encoded "
            "from the supplied context prefix."
        ),
    )
    parser.add_argument(
        "--reset-global-seed-per-case",
        action="store_true",
        help="Reset torch CPU/CUDA global RNG before every case for paired ablations.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="Write <case>.mp4 and <case>.json directly under --output-root.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--object-xssc-trainable",
        type=Path,
        default=None,
        help="Optional full-SA+Object XSSC state saved beside the PEFT adapter.",
    )
    parser.add_argument("--xssc-root", type=Path, default=Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/xssc_rsfq2_ytvis_dinov3_vitl16_256"))
    parser.add_argument("--xssc-config", type=Path, default=Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/xssc_rsfq2_ytvis_dinov3_vitl16_256/upstream/config-randsfq/rsfq2_c-movi_c-dinov3_vitl16_256-slot512-transfer15000.py"))
    parser.add_argument("--xssc-checkpoint", type=Path, default=Path("/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/dinov3_xSSC/restart_save1000_20260720T140029Z/movi_c_transfer15000_b64_acc3_20260721T134713Z/rsfq2_c-movi_c-dinov3_vitl16_256-slot512-transfer15000/42/step-050000.pth"))
    parser.add_argument("--dinov3-root", type=Path, default=Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/xssc_rsfq2_ytvis_dinov3_vitl16_256/third_party/dinov3"))
    parser.add_argument("--dinov3-checkpoint", type=Path, default=Path("/data/gaoya/ckpt/facebook-dinov3-vitl16-pretrain-lvd1689m/model.safetensors"))
    parser.add_argument("--sam2-config", type=Path, default=Path("/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_l.yaml"))
    parser.add_argument("--sam2-checkpoint", type=Path, default=Path("/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_large.pt"))
    parser.add_argument("--xssc-box-cache-dir", type=Path, default=Path("/data/gaoya/agent-data/cache/physrvg_fastvideo_xssc_object_amg"))
    parser.add_argument("--xssc-chunk-size", type=int, default=2)
    parser.add_argument("--object-context-frames", type=int, default=8)
    parser.add_argument("--object-lora-rank", type=int, default=32)
    parser.add_argument("--object-lora-alpha", type=float, default=32.0)
    parser.add_argument("--object-lora-dropout", type=float, default=0.0)
    parser.add_argument("--object-gate-init", type=float, default=0.1)
    parser.add_argument("--slot-dedup-threshold", type=float, default=0.94)
    parser.add_argument("--slot-dedup-min-keep", type=int, default=3)
    return parser.parse_args()


def ensure_exists(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def resolve_prompt_suffix(
    prompt_cache_dir: Path | None,
    mode: str,
) -> tuple[str, dict[str, Any]]:
    """Resolve the suffix actually used by a training prompt cache.

    The cache stores positive-prompt hashes in index.jsonl and their text in
    prompt_index.jsonl.  We inspect only those positive prompts, rather than
    all cache entries, because the cache also contains object/phrase prompts.
    This matters for caches whose config-level suffix field is empty even
    though the suffix has already been baked into every positive prompt.
    """
    if mode == "never" or prompt_cache_dir is None:
        return "", {
            "mode": mode,
            "cache_dir": None if prompt_cache_dir is None else str(prompt_cache_dir),
            "suffix": "",
            "suffix_detected": False,
            "reason": "disabled_or_no_cache",
        }

    cache_dir = ensure_exists(prompt_cache_dir, "prompt-cache-dir")
    if not cache_dir.is_dir():
        raise ValueError(f"prompt-cache-dir is not a directory: {cache_dir}")
    config_path = cache_dir / "cache_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"prompt cache config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"prompt cache config must be an object: {config_path}")

    configured_suffix = str(config.get("positive_prompt_suffix") or "").strip()
    if configured_suffix:
        return configured_suffix, {
            "mode": mode,
            "cache_dir": str(cache_dir),
            "suffix": configured_suffix,
            "suffix_detected": True,
            "reason": "cache_config_positive_prompt_suffix",
        }

    index_path = cache_dir / str(config.get("index_file", "index.jsonl"))
    prompt_index_path = cache_dir / str(
        config.get("prompt_index_file", "prompt_index.jsonl")
    )
    if not index_path.is_file() or not prompt_index_path.is_file():
        raise FileNotFoundError(
            "prompt cache index files not found: "
            f"{index_path}, {prompt_index_path}"
        )
    prompt_rows = {
        str(row.get("prompt_sha256")): row
        for row in _read_jsonl(prompt_index_path)
        if row.get("prompt_sha256")
    }
    positive_hashes = {
        str(row["positive_prompt"])
        for row in _read_jsonl(index_path)
        if row.get("positive_prompt")
    }
    positive_prompts = [
        str(prompt_rows[prompt_hash]["prompt"]).strip()
        for prompt_hash in positive_hashes
        if prompt_hash in prompt_rows
        and isinstance(prompt_rows[prompt_hash].get("prompt"), str)
    ]
    if not positive_prompts:
        return "", {
            "mode": mode,
            "cache_dir": str(cache_dir),
            "suffix": "",
            "suffix_detected": False,
            "reason": "no_positive_prompts",
        }

    # The current physics-consistency cache has the suffix baked into the
    # prompt text, while its config field is empty.  Detect that exact suffix
    # only when it is present on all positive prompts; this avoids treating an
    # arbitrary object phrase in the cache as a training-caption suffix.
    suffix_matches = sum(
        prompt.endswith(PHYSICS_CONSISTENCY_SUFFIX) for prompt in positive_prompts
    )
    if suffix_matches == len(positive_prompts):
        return PHYSICS_CONSISTENCY_SUFFIX, {
            "mode": mode,
            "cache_dir": str(cache_dir),
            "suffix": PHYSICS_CONSISTENCY_SUFFIX,
            "suffix_detected": True,
            "reason": "positive_prompt_text_suffix",
            "positive_prompt_count": len(positive_prompts),
        }
    return "", {
        "mode": mode,
        "cache_dir": str(cache_dir),
        "suffix": "",
        "suffix_detected": False,
        "reason": "positive_prompts_have_no_uniform_known_suffix",
        "positive_prompt_count": len(positive_prompts),
        "suffix_match_count": suffix_matches,
    }


def apply_prompt_suffix(prompt: str, suffix: str) -> tuple[str, bool]:
    prompt = str(prompt).strip()
    suffix = str(suffix).strip()
    if not suffix or prompt.endswith(suffix):
        return prompt, False
    return f"{prompt} {suffix}", True


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def read_input_list(path: Path) -> list[Path]:
    items = [
        Path(line.strip()).expanduser().resolve()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not items:
        raise ValueError(f"input JSON list is empty: {path}")
    return items


def crop_and_resize(image: Image.Image, target_height: int, target_width: int) -> Image.Image:
    source_width, source_height = image.size
    scale = max(target_width / source_width, target_height / source_height)
    image = transform_functional.resize(
        image,
        (round(source_height * scale), round(source_width * scale)),
        interpolation=InterpolationMode.BILINEAR,
    )
    return transform_functional.center_crop(image, (target_height, target_width))


def open_video_reader(video_path: Path):
    try:
        return imageio.get_reader(str(video_path), format="FFMPEG")
    except Exception:
        return imageio.get_reader(str(video_path))


def video_length(reader: Any) -> int:
    try:
        value = int(reader.count_frames())
        if value > 0:
            return value
    except Exception:
        pass
    try:
        value = int(reader.get_length())
        if value > 0:
            return value
    except Exception:
        pass
    metadata = reader.get_meta_data()
    fps = metadata.get("fps")
    duration = metadata.get("duration")
    if fps and duration:
        value = int(round(float(fps) * float(duration)))
        if value > 0:
            return value
    raise RuntimeError("unable to determine context video frame count")


def load_context_video(
    video_path: Path,
    height: int,
    width: int,
    max_frames: int | None = None,
) -> list[Image.Image]:
    reader = open_video_reader(video_path)
    try:
        frame_count = video_length(reader)
        if max_frames is not None:
            if max_frames < 1:
                raise ValueError("max_frames must be positive")
            frame_count = min(frame_count, int(max_frames))
        frames = [
            crop_and_resize(
                Image.fromarray(reader.get_data(frame_id)).convert("RGB"),
                height,
                width,
            )
            for frame_id in range(frame_count)
        ]
    finally:
        reader.close()
    if not frames:
        raise RuntimeError(f"no readable frames found in {video_path}")
    return frames


def save_context_sheet(frames: list[Image.Image], output_path: Path) -> None:
    canvas = Image.new("RGB", (sum(frame.width for frame in frames), max(frame.height for frame in frames)))
    cursor_x = 0
    for frame in frames:
        canvas.paste(frame, (cursor_x, 0))
        cursor_x += frame.width
    canvas.save(output_path, format="JPEG", quality=95)


def load_pipeline(
    args: argparse.Namespace,
) -> tuple[WanImageToVideoPipeline, int, dict[str, Any], Any, dict[str, Any]]:
    from fastvideo.models.wan_v2v.context_only_attention import load_attention_policy

    policy = load_attention_policy(args.lora_checkpoint)
    args.self_attention_policy = policy
    if policy["mode"] == "context_only":
        if args.context_mask_mode != "official_fixed2" or args.context_frames != 8:
            raise ValueError("context_only checkpoint requires context8 and official_fixed2")
    vae = AutoencoderKLWan.from_pretrained(
        str(args.model_id), subfolder="vae", torch_dtype=torch.float32
    )
    transformer = WanTransformer3DModel.from_pretrained(
        str(args.model_id), subfolder="transformer", torch_dtype=torch.bfloat16
    )
    if policy["mode"] == "context_only":
        if transformer.config.patch_size[0] != 1:
            raise ValueError("context_only requires temporal patch size 1")
        transformer.context_only_latent_frames = policy["context_latent_frames"]
    pipe = WanImageToVideoPipeline.from_pretrained(
        str(args.model_id),
        transformer=transformer,
        vae=vae,
        torch_dtype=torch.bfloat16,
    )
    physrvg_dit = {
        "checkpoint": None,
        "strict": False,
        "source_tensors": 0,
    }
    if args.physrvg_dit_checkpoint is not None:
        # Load the large PhysRVG checkpoint directly onto the inference device.
        # This avoids a CPU-side 10 GB staging copy and the subsequent transfer.
        state_dict = load_file(str(args.physrvg_dit_checkpoint), device=str(args.device))
        source_tensors = len(state_dict)
        pipe.transformer.load_state_dict(state_dict, strict=True)
        del state_dict
        physrvg_dit = {
            "checkpoint": str(args.physrvg_dit_checkpoint),
            "strict": True,
            "source_tensors": source_tensors,
        }
    loaded_lora_targets = 0
    if args.lora_checkpoint is not None:
        pipe.transformer = PeftModel.from_pretrained(
            pipe.transformer,
            str(args.lora_checkpoint),
            is_trainable=False,
        )
        pipe.transformer.set_adapter("default")
        loaded_lora_targets = sum(
            1
            for module in pipe.transformer.modules()
            if getattr(module, "lora_A", None) is not None and "default" in module.lora_A
        )
        if loaded_lora_targets == 0:
            raise RuntimeError("the LoRA checkpoint did not attach to any transformer module")
    object_system = None
    object_info = {"enabled": False, "checkpoint": None, "source_tensors": 0}
    object_checkpoint = args.object_xssc_trainable
    if object_checkpoint is None and args.lora_checkpoint is not None:
        candidate = args.lora_checkpoint / "physrvg_object_xssc_trainable.safetensors"
        if candidate.is_file():
            object_checkpoint = candidate
    if object_checkpoint is not None:
        if args.lora_checkpoint is None:
            raise ValueError("Object XSSC state requires --lora-checkpoint")
        object_checkpoint = ensure_exists(object_checkpoint, "object-xssc-trainable")
        object_args = Namespace(
            object_lora_rank=args.object_lora_rank,
            object_lora_alpha=args.object_lora_alpha,
            object_lora_dropout=args.object_lora_dropout,
            object_gate_init=args.object_gate_init,
            object_context_frames=args.object_context_frames,
        )
        xssc_object.inject_object_branch(pipe.transformer, object_args)
        object_state = load_file(str(object_checkpoint), device=str(args.device))
        model_state = pipe.transformer.state_dict()
        unknown = sorted(set(object_state) - set(model_state))
        mismatched = [
            key for key, value in object_state.items()
            if key in model_state and tuple(value.shape) != tuple(model_state[key].shape)
        ]
        if unknown or mismatched:
            raise RuntimeError(
                "Object XSSC checkpoint mismatch: "
                f"unknown={unknown[:3]} mismatched={mismatched[:3]}"
            )
        pipe.transformer.load_state_dict(object_state, strict=False)
        object_source_tensors = len(object_state)
        del object_state
        object_system = xssc_object.FastVideoXSSCSystem(
            Namespace(
                tiny_vae_root=Path("/home/gaoya/Code_Video/taehv"),
                tiny_vae_checkpoint=Path("/data/gaoya/ckpt/taew2_2.pth"),
                xssc_root=args.xssc_root,
                xssc_config=args.xssc_config,
                xssc_checkpoint=args.xssc_checkpoint,
                dinov3_root=args.dinov3_root,
                dinov3_checkpoint=args.dinov3_checkpoint,
                xssc_box_cache_dir=args.xssc_box_cache_dir,
                xssc_chunk_size=args.xssc_chunk_size,
                object_context_frames=args.object_context_frames,
                slot_dedup_threshold=args.slot_dedup_threshold,
                slot_dedup_min_keep=args.slot_dedup_min_keep,
                sam2_config=args.sam2_config,
                sam2_checkpoint=args.sam2_checkpoint,
                xssc_loss_weight=0.1,
            ),
            torch.device(args.device),
            torch.bfloat16,
            load_tiny_vae=False,
        )
        object_info = {
            "enabled": True,
            "checkpoint": str(object_checkpoint),
            "source_tensors": object_source_tensors,
        }
    pipe.to(torch.device(args.device))
    return pipe, loaded_lora_targets, physrvg_dit, object_system, object_info


def run_case(
    *,
    pipe: WanImageToVideoPipeline,
    args: argparse.Namespace,
    input_json_path: Path,
    global_index: int,
    loaded_lora_targets: int,
    physrvg_dit: dict[str, Any],
    object_system: Any,
    object_info: dict[str, Any],
) -> dict[str, Any]:
    payload = json.loads(input_json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"input JSON must contain an object: {input_json_path}")
    input_video_value = payload.get("input_video")
    input_caption = payload.get("input_caption")
    if not isinstance(input_video_value, str) or not input_video_value.strip():
        raise ValueError(f"missing input_video in {input_json_path}")
    if not isinstance(input_caption, str) or not input_caption.strip():
        raise ValueError(f"missing input_caption in {input_json_path}")

    input_video_path = Path(input_video_value).expanduser().resolve()
    if not input_video_path.is_file():
        raise FileNotFoundError(f"input video not found: {input_video_path}")
    sample_stem = input_json_path.stem
    if args.flat_output:
        case_dir = args.output_root
        output_video = case_dir / f"{sample_stem}.mp4"
        metadata_path = case_dir / f"{sample_stem}.json"
        input_copy_path = case_dir / f"{sample_stem}.input.json"
        context_sheet_path = case_dir / f"{sample_stem}_input_context.jpg"
    else:
        case_dir = args.output_root / "cases" / f"{global_index:02d}_{sample_stem}"
        output_video = case_dir / "generated.mp4"
        metadata_path = case_dir / "metadata.json"
        input_copy_path = case_dir / "input.json"
        context_sheet_path = case_dir / "input_context.jpg"
    if output_video.exists() and metadata_path.exists() and not args.force:
        return {
            "case_index": global_index,
            "input_json": str(input_json_path),
            "output_video": str(output_video),
            "status": "skipped_existing",
        }

    case_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_json_path, input_copy_path)
    read_frame_limit = (
        int(args.context_frames) if args.context_frames is not None else None
    )
    all_context_frames = load_context_video(
        input_video_path,
        int(args.height),
        int(args.width),
        max_frames=read_frame_limit,
    )
    if args.context_frames is not None:
        requested_context_frames = int(args.context_frames)
        if requested_context_frames < 1:
            raise ValueError("--context-frames must be positive")
        if requested_context_frames > len(all_context_frames):
            raise ValueError(
                f"requested {requested_context_frames} context frames, "
                f"but input contains {len(all_context_frames)} readable frames: "
                f"{input_video_path}"
            )
        context_frames = all_context_frames[:requested_context_frames]
    else:
        requested_context_frames = None
        context_frames = all_context_frames
    context_mask_mode = getattr(args, "context_mask_mode", "official_fixed2")
    if context_mask_mode not in {"official_fixed2", "dynamic_effective"}:
        raise ValueError(f"unsupported context mask mode: {context_mask_mode}")
    vae_temporal_scale = int(pipe.vae_scale_factor_temporal)
    effective_context_latent_frames = (
        (len(context_frames) - 1) // vae_temporal_scale + 1
    )
    output_latent_frames = (
        (int(args.num_frames) - 1) // vae_temporal_scale + 1
    )
    if effective_context_latent_frames > output_latent_frames:
        raise ValueError(
            f"context has {effective_context_latent_frames} latent frames, "
            f"but output has only {output_latent_frames}; reduce context-frames "
            f"or increase num-frames"
        )
    save_context_sheet(context_frames, context_sheet_path)
    object_bundle = None
    if object_system is not None:
        context_array = np.stack([np.asarray(frame) for frame in context_frames], axis=0)
        context_tensor = (
            torch.from_numpy(context_array)
            .permute(0, 3, 1, 2)
            .float()
            .div(255)
            .unsqueeze(0)
            .to(args.device)
        )
        object_bundle = object_system.prepare_context(context_tensor)

    rng_mode = str(getattr(args, "rng_mode", "explicit_generator"))
    reset_global_seed_per_case = bool(
        getattr(args, "reset_global_seed_per_case", False)
    )
    if rng_mode == "global_seed_per_case":
        if not reset_global_seed_per_case:
            raise ValueError(
                "--rng-mode global_seed_per_case requires "
                "--reset-global-seed-per-case"
            )
        set_seed(int(args.seed))
        generator = None
    else:
        if reset_global_seed_per_case:
            torch.manual_seed(int(args.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(args.seed))
        generator = torch.Generator(device=args.device).manual_seed(int(args.seed))
    prompt_source = str(args.prompt).strip() if args.prompt is not None else input_caption.strip()
    generation_prompt, suffix_applied = apply_prompt_suffix(
        prompt_source,
        str(getattr(args, "prompt_suffix", "")),
    )
    if not generation_prompt:
        raise ValueError("generation prompt must not be empty")
    negative_prompt = str(getattr(args, "negative_prompt", NEGATIVE_PROMPT))
    do_cfg = bool(getattr(args, "do_cfg", False))
    sample = pipe(
        video=context_frames,
        device=torch.device(args.device),
        prompt=generation_prompt,
        negative_prompt=negative_prompt,
        height=int(args.height),
        width=int(args.width),
        num_frames=int(args.num_frames),
        num_inference_steps=int(args.num_inference_steps),
        guidance_scale=float(args.guidance_scale),
        do_cfg=do_cfg,
        generator=generator,
        dynamic_condition_mask=(context_mask_mode == "dynamic_effective"),
        context_noise_mode=getattr(args, "context_noise_mode", "clean"),
        object_slots=None if object_bundle is None else object_bundle["object_slots"],
        object_keep_mask=None if object_bundle is None else object_bundle["object_keep"],
    )[0]
    export_to_video(sample[0], str(output_video), fps=int(args.fps))

    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_index": global_index,
        "input_json": str(input_json_path),
        "input_caption": input_caption.strip(),
        "prompt": generation_prompt,
        "prompt_policy": {
            "source": "fixed_override" if args.prompt is not None else "input_caption",
            "suffix_applied": suffix_applied,
            "suffix": str(getattr(args, "prompt_suffix", "")),
            "cache": getattr(args, "prompt_suffix_info", {}),
        },
        "output_video": str(output_video),
        "input": {
            "json_path": str(input_json_path),
            "video_path": str(input_video_path),
            "caption": input_caption.strip(),
            "prompt": generation_prompt,
            "prompt_policy": {
                "source": "fixed_override" if args.prompt is not None else "input_caption",
                "suffix_applied": suffix_applied,
                "suffix": str(getattr(args, "prompt_suffix", "")),
                "cache": getattr(args, "prompt_suffix_info", {}),
            },
            "context_frames": len(context_frames),
            "available_context_frames": len(all_context_frames),
            "requested_context_frames": requested_context_frames,
            "frame_indices": list(range(len(context_frames))),
            "context_contact_sheet": context_sheet_path.name,
        },
        "model": {
            "load_mode": (
                "wan22_diffusers_base_plus_physrvg_dit_strict_plus_full_sa_object_xssc_lora"
                if physrvg_dit["strict"] and object_info["enabled"]
                else "wan22_diffusers_base_plus_physrvg_dit_strict_plus_full_sa_lora"
                if physrvg_dit["strict"] and args.lora_checkpoint is not None
                else "wan22_diffusers_base_plus_physrvg_dit_strict_lora_off"
                if physrvg_dit["strict"]
                else "wan22_diffusers_base_plus_full_sa_lora"
                if args.lora_checkpoint is not None
                else "wan22_diffusers_base_lora_off"
            ),
            "model_id": str(args.model_id),
            "dit_checkpoint": physrvg_dit["checkpoint"],
            "dit_strict": physrvg_dit["strict"],
            "dit_source_tensors": physrvg_dit["source_tensors"],
            "lora_checkpoint": (
                str(args.lora_checkpoint) if args.lora_checkpoint is not None else None
            ),
            "loaded_lora_target_modules": loaded_lora_targets,
            "object_xssc": object_info,
        },
        "inference": {
            "device": args.device,
            "height": int(args.height),
            "width": int(args.width),
            "num_frames": int(args.num_frames),
            "fps": int(args.fps),
            "num_inference_steps": int(args.num_inference_steps),
            "guidance_scale": float(args.guidance_scale),
            "prompt": generation_prompt,
            "negative_prompt": negative_prompt,
            "classifier_free_guidance_enabled": do_cfg,
            "seed": int(args.seed),
            "rng_mode": rng_mode,
            "requested_context_frames": requested_context_frames,
            "effective_context_frames": len(context_frames),
            "self_attention_policy": getattr(args, "self_attention_policy", {"mode": "full"}),
            "context_mask_mode": context_mask_mode,
            "context_noise_mode": getattr(args, "context_noise_mode", "clean"),
            "context_timestep_mode": (
                "match_noise" if getattr(args, "context_noise_mode", "clean") != "clean" else "force_zero"
            ),
            "condition_mask_strategy": context_mask_mode,
            "effective_context_latent_frames": effective_context_latent_frames,
            "output_latent_frames": output_latent_frames,
            "global_seed_reset_per_case": bool(
                getattr(args, "reset_global_seed_per_case", False)
            ),
        },
        "output": {"video": output_video.name},
    }
    write_json(metadata_path, metadata)
    return {
        "case_index": global_index,
        "input_json": str(input_json_path),
        "output_video": str(output_video),
        "status": "generated",
    }


def main(default_context_noise_mode: str = "clean", *, args=None, pipeline_validator=None) -> None:
    if args is None:
        args = parse_args(default_context_noise_mode)
    if args.shard_count <= 0:
        raise ValueError("--shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must be in [0, shard-count)")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.num_frames < 1 or (args.num_frames - 1) % 4 != 0:
        raise ValueError("--num-frames must satisfy 4n+1 for Wan VAE temporal alignment")

    args.input_json_list = ensure_exists(args.input_json_list, "input-json-list")
    args.model_id = ensure_exists(args.model_id, "model-id")
    if args.physrvg_dit_checkpoint is not None:
        args.physrvg_dit_checkpoint = ensure_exists(
            args.physrvg_dit_checkpoint, "physrvg-dit-checkpoint"
        )
        if not args.physrvg_dit_checkpoint.is_file():
            raise FileNotFoundError(
                f"physrvg-dit-checkpoint is not a file: {args.physrvg_dit_checkpoint}"
            )
    if args.lora_checkpoint is not None:
        args.lora_checkpoint = ensure_exists(args.lora_checkpoint, "lora-checkpoint")
    if args.prompt_cache_dir is not None:
        args.prompt_cache_dir = ensure_exists(args.prompt_cache_dir, "prompt-cache-dir")
    args.prompt_suffix, args.prompt_suffix_info = resolve_prompt_suffix(
        args.prompt_cache_dir,
        args.prompt_suffix_mode,
    )
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    all_cases = read_input_list(args.input_json_list)
    if args.limit is not None:
        all_cases = all_cases[: args.limit]
    selected_cases = [
        (index, path)
        for index, path in enumerate(all_cases)
        if index % args.shard_count == args.shard_index
    ]
    for _, path in selected_cases:
        ensure_exists(path, "input-json")

    pipe, loaded_lora_targets, physrvg_dit, object_system, object_info = load_pipeline(args)
    if pipeline_validator is not None:
        pipeline_validator(pipe, args)

    print(
        f"[load] base={args.model_id} dit={physrvg_dit['checkpoint']} "
        f"dit_strict={physrvg_dit['strict']} lora={args.lora_checkpoint} "
        f"lora_target_modules={loaded_lora_targets} object_xssc={object_info['enabled']} "
        f"prompt_suffix_detected={bool(args.prompt_suffix)} "
        f"prompt_suffix_reason={args.prompt_suffix_info.get('reason')}",
        flush=True,
    )
    results: list[dict[str, Any]] = []
    for global_index, input_json_path in selected_cases:
        print(f"[case {global_index:02d}] {input_json_path.name}", flush=True)
        try:
            result = run_case(
                pipe=pipe,
                args=args,
                input_json_path=input_json_path,
                global_index=global_index,
                loaded_lora_targets=loaded_lora_targets,
                physrvg_dit=physrvg_dit,
                object_system=object_system,
                object_info=object_info,
            )
        except Exception as exc:
            result = {
                "case_index": global_index,
                "input_json": str(input_json_path),
                "status": "failed",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        results.append(result)
        print(f"[case {global_index:02d}] {result['status']}", flush=True)
        torch.cuda.empty_cache()

    counts: dict[str, int] = {}
    for result in results:
        status = str(result["status"])
        counts[status] = counts.get(status, 0) + 1
    suffix = f"shard-{args.shard_index:02d}-of-{args.shard_count:02d}"
    write_json(
        args.output_root / f"summary.{suffix}.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_json_list": str(args.input_json_list),
            "total_cases_in_list": len(all_cases),
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "selected_cases": len(selected_cases),
            "counts": counts,
            "results": results,
        },
    )
    if counts.get("failed", 0):
        raise SystemExit(f"{counts['failed']} case(s) failed")


if __name__ == "__main__":
    main()
