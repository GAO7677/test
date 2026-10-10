"""Collision/impact supervision for the PhysRVG latent-mask loss.

The filtered PyBullet training dataset keeps RGB videos and object masks, but
not the simulator contact stream.  This module consumes a small replay cache
written by ``replay_pybullet_collision_supervision.py`` and turns contact
impulses into weights aligned with the 49-frame / 13-latent-frame training
layout.

The label policy is deliberately conservative:

* dynamic--dynamic contact starts are included;
* dynamic--static-object contact starts are included (obstacles/supports);
* dynamic--ground contact starts are included only after frame 0, which
  captures a later landing while excluding initial floor placement;
* a short temporal window is used around an impact, while persistent contact
  is not repeatedly labelled as a new collision.

The resulting score is a diagnostic/training weight, not a claim that the
dataset contains manually annotated collision frames.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np


CATEGORY_NAMES = ("dynamic_dynamic", "dynamic_static", "dynamic_ground")


def _pair_category(
    obj_a: str,
    obj_b: str,
    dynamic_names: set[str],
) -> str | None:
    a_dynamic = obj_a in dynamic_names
    b_dynamic = obj_b in dynamic_names
    if a_dynamic and b_dynamic:
        return "dynamic_dynamic"
    if a_dynamic or b_dynamic:
        if obj_a == "ground" or obj_b == "ground":
            return "dynamic_ground"
        return "dynamic_static"
    return None


def _force_by_pair(
    records: Iterable[dict],
    *,
    dynamic_names: set[str],
    frame_count: int,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], str]]:
    forces: dict[tuple[str, str], np.ndarray] = {}
    categories: dict[tuple[str, str], str] = {}
    for record in records:
        frame = int(record.get("frame", -1))
        if frame < 0 or frame >= frame_count:
            continue
        obj_a = str(record.get("obj_a", ""))
        obj_b = str(record.get("obj_b", ""))
        category = _pair_category(obj_a, obj_b, dynamic_names)
        if category is None:
            continue
        pair = (obj_a, obj_b)
        values = forces.setdefault(pair, np.zeros(frame_count, dtype=np.float32))
        force = 0.0
        for contact in record.get("contacts", []):
            force += max(0.0, float(contact.get("normal_force_n", 0.0)))
        values[frame] += float(force)
        categories[pair] = category
    return forces, categories


def _runs(active: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    ends = np.flatnonzero(active & ~np.r_[active[1:], False])
    # ``ends`` contains the last active index; expose half-open intervals so
    # a run ending at the final frame still has a non-empty slice.
    return [(int(start), int(end) + 1) for start, end in zip(starts, ends)]


def build_collision_supervision(
    records: Iterable[dict],
    *,
    dynamic_names: set[str],
    frame_count: int = 49,
    latent_frames: int = 13,
    latent_stride: int = 4,
    window_radius: int = 1,
    window_sigma: float = 1.0,
) -> dict[str, object]:
    """Build frame/latent impact scores from a PyBullet contact stream."""
    if frame_count <= 0 or latent_frames <= 0 or latent_stride <= 0:
        raise ValueError("frame_count, latent_frames and latent_stride must be positive")
    if window_radius < 0 or window_sigma <= 0.0:
        raise ValueError("invalid temporal window parameters")

    forces, categories = _force_by_pair(
        records,
        dynamic_names=dynamic_names,
        frame_count=frame_count,
    )
    all_force_values = np.concatenate(list(forces.values())) if forces else np.zeros(1, dtype=np.float32)
    positive_forces = all_force_values[all_force_values > 0.0]
    global_max_force = float(positive_forces.max()) if positive_forces.size else 0.0
    contact_epsilon = max(1.0e-6, global_max_force * 1.0e-4)

    category_force = {
        name: np.zeros(frame_count, dtype=np.float32) for name in CATEGORY_NAMES
    }
    frame_scores = np.zeros(frame_count, dtype=np.float32)
    category_scores = {
        name: np.zeros(frame_count, dtype=np.float32) for name in CATEGORY_NAMES
    }
    frame_has_collision = np.zeros(frame_count, dtype=np.bool_)
    events: list[dict[str, object]] = []

    for pair, force in forces.items():
        category = categories[pair]
        category_force[category] += force
        active = force > contact_epsilon
        for start, end in _runs(active):
            # Frame 0 contact is normally the initial placement/support.  It
            # is deliberately not called an impact; later contact runs are
            # eligible for obstacle hits and landings.
            if start == 0:
                continue
            local_end = min(end, start + 4)
            peak_force = float(force[start:local_end].max())
            if peak_force <= contact_epsilon:
                continue
            amplitude = min(1.0, peak_force / max(global_max_force, contact_epsilon))
            events.append(
                {
                    "frame": int(start),
                    "pair": list(pair),
                    "category": category,
                    "peak_force_n_first_4_frames": peak_force,
                    "score": amplitude,
                    "reason": "contact_onset_after_frame0",
                }
            )
            for offset in range(-window_radius, window_radius + 1):
                target = start + offset
                if not 0 <= target < frame_count:
                    continue
                temporal = float(np.exp(-0.5 * (offset / window_sigma) ** 2))
                score = float(amplitude * temporal)
                category_scores[category][target] = max(
                    category_scores[category][target], score
                )
                frame_scores[target] = max(frame_scores[target], score)
                frame_has_collision[target] = True

        # Some solver contacts persist through an impact instead of separating
        # for a sampled frame.  Add only a strong local force jump, rather than
        # weighting every frame of the persistent contact.
        for frame in range(1, frame_count):
            if not active[frame] or not active[frame - 1]:
                continue
            previous = float(force[frame - 1])
            current = float(force[frame])
            if previous <= contact_epsilon:
                continue
            if current < 1.5 * previous or current < 0.35 * global_max_force:
                continue
            if frame + 1 < frame_count and current < float(force[frame + 1]):
                continue
            amplitude = min(1.0, current / max(global_max_force, contact_epsilon))
            events.append(
                {
                    "frame": int(frame),
                    "pair": list(pair),
                    "category": category,
                    "peak_force_n_first_4_frames": current,
                    "score": amplitude,
                    "reason": "persistent_contact_force_peak",
                }
            )
            for offset in range(-window_radius, window_radius + 1):
                target = frame + offset
                if not 0 <= target < frame_count:
                    continue
                temporal = float(np.exp(-0.5 * (offset / window_sigma) ** 2))
                score = float(amplitude * temporal)
                category_scores[category][target] = max(
                    category_scores[category][target], score
                )
                frame_scores[target] = max(frame_scores[target], score)
                frame_has_collision[target] = True

    latent_scores = np.zeros(latent_frames, dtype=np.float32)
    latent_category_scores = {
        name: np.zeros(latent_frames, dtype=np.float32) for name in CATEGORY_NAMES
    }
    latent_has_collision = np.zeros(latent_frames, dtype=np.bool_)
    for latent_frame in range(latent_frames):
        start = latent_frame * latent_stride
        end = min(frame_count, start + latent_stride)
        if start >= frame_count:
            continue
        latent_scores[latent_frame] = float(frame_scores[start:end].max(initial=0.0))
        latent_has_collision[latent_frame] = bool(frame_has_collision[start:end].any())
        for category in CATEGORY_NAMES:
            latent_category_scores[category][latent_frame] = float(
                category_scores[category][start:end].max(initial=0.0)
            )

    return {
        "frame_scores": frame_scores,
        "category_scores": category_scores,
        "category_force_n": category_force,
        "frame_has_collision": frame_has_collision,
        "latent_scores": latent_scores,
        "latent_category_scores": latent_category_scores,
        "latent_has_collision": latent_has_collision,
        "events": events,
        "global_max_force_n": global_max_force,
        "contact_force_epsilon_n": contact_epsilon,
        "frame_count": int(frame_count),
        "latent_frames": int(latent_frames),
        "latent_stride": int(latent_stride),
        "window_radius": int(window_radius),
        "window_sigma": float(window_sigma),
        "category_names": list(CATEGORY_NAMES),
    }


def save_collision_supervision(
    output_dir: Path,
    *,
    logical_key: str,
    dynamic_names: set[str],
    records: Iterable[dict],
    build_kwargs: dict[str, object] | None = None,
    replay_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build and persist one compact collision-supervision entry."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result = build_collision_supervision(
        records,
        dynamic_names=dynamic_names,
        **(build_kwargs or {}),
    )
    arrays: dict[str, np.ndarray] = {
        "frame_scores": np.asarray(result["frame_scores"], dtype=np.float32),
        "frame_has_collision": np.asarray(result["frame_has_collision"], dtype=np.bool_),
        "latent_scores": np.asarray(result["latent_scores"], dtype=np.float32),
        "latent_has_collision": np.asarray(result["latent_has_collision"], dtype=np.bool_),
    }
    for category in CATEGORY_NAMES:
        arrays[f"category_scores_{category}"] = np.asarray(
            result["category_scores"][category], dtype=np.float32
        )
        arrays[f"category_force_n_{category}"] = np.asarray(
            result["category_force_n"][category], dtype=np.float32
        )
        arrays[f"latent_category_scores_{category}"] = np.asarray(
            result["latent_category_scores"][category], dtype=np.float32
        )
    np.savez_compressed(output_dir / "collision_supervision.npz", **arrays)
    metadata = {
        "logical_key": logical_key,
        "dynamic_names": sorted(dynamic_names),
        "static_contact_is_obstacle_or_support": True,
        "ground_contact_is_impact_only_after_frame0": True,
        "events": result["events"],
        "global_max_force_n": result["global_max_force_n"],
        "contact_force_epsilon_n": result["contact_force_epsilon_n"],
        "frame_count": result["frame_count"],
        "latent_frames": result["latent_frames"],
        "latent_stride": result["latent_stride"],
        "window_radius": result["window_radius"],
        "window_sigma": result["window_sigma"],
        "category_names": list(CATEGORY_NAMES),
        "weight_formula": "clip(1 + collision_weight_strength * latent_score, 1, collision_weight_cap)",
        "replay": replay_metadata or {},
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def load_collision_supervision(cache_root: Path, logical_key: str) -> tuple[dict[str, np.ndarray], dict]:
    """Load one compact entry for diagnostics or training."""
    entry_root = cache_root / "cases" / logical_key
    arrays_path = entry_root / "collision_supervision.npz"
    metadata_path = entry_root / "metadata.json"
    if not arrays_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"missing collision supervision entry for {logical_key}: {entry_root}")
    with np.load(arrays_path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return arrays, metadata


@lru_cache(maxsize=8192)
def _cached_latent_scores(cache_root: str, logical_key: str) -> np.ndarray:
    arrays, _ = load_collision_supervision(Path(cache_root), logical_key)
    return np.asarray(arrays["latent_scores"], dtype=np.float32)


def batch_collision_weights(
    cache_root: Path,
    logical_keys: Iterable[str],
    *,
    strength: float = 2.0,
    cap: float = 3.0,
) -> np.ndarray:
    """Return cached ``[batch, latent_frames]`` collision weights."""
    root = str(cache_root.expanduser().resolve())
    scores = np.stack([_cached_latent_scores(root, str(key)) for key in logical_keys])
    return collision_weights(scores, strength=strength, cap=cap)


def collision_weights(
    latent_scores: np.ndarray,
    *,
    strength: float = 2.0,
    cap: float = 3.0,
) -> np.ndarray:
    """Convert normalized latent impact scores to multiplicative loss weights."""
    if strength < 0.0 or cap < 1.0:
        raise ValueError("strength must be non-negative and cap must be >= 1")
    scores = np.asarray(latent_scores, dtype=np.float32)
    return np.clip(1.0 + float(strength) * np.clip(scores, 0.0, 1.0), 1.0, float(cap))
