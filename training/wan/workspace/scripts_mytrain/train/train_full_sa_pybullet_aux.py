from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import signal
import sys
import time
import traceback
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train_full_sa_pybullet as base
from collision_supervision import batch_collision_weights
from fastvideo.lora_profiles import RL_LORA_TARGET_MODULES, rl_lora_config_kwargs


REF_EXPERIMENT_ROOT = Path(
    os.environ.get(
        "PHYSRVG_REF_EXPERIMENT_ROOT",
        "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/"
        "code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/"
        "attention_trajectory_distillation_project",
    )
)
if not (REF_EXPERIMENT_ROOT / "frozen_motion_probe.py").is_file():
    REF_EXPERIMENT_ROOT = SCRIPT_DIR
if str(REF_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(REF_EXPERIMENT_ROOT))

from frozen_motion_probe import fixed_query_frame_head_probabilities


LATENT_MASK_MODES = frozenset(
    {"latent_mask", "latent_mask_full_frame", "latent_mask_utonia_scene"}
)
UTONIA_SCENE_MODES = frozenset({"vjepa_utonia_scene", "latent_mask_utonia_scene"})
UTONIA_LOCAL_SCENE_MODES = frozenset({"utonia_local_scene"})
UTONIA_OBJECT_SCENE_MODES = frozenset({"utonia_object_scene_attention"})
UTONIA_SUBTRACTION_MODES = frozenset({"utonia_subtraction"})
SCENE_CONDITION_MODES = (
    UTONIA_SCENE_MODES
    | UTONIA_LOCAL_SCENE_MODES
    | UTONIA_OBJECT_SCENE_MODES
    | UTONIA_SUBTRACTION_MODES
)
VJEPA_MODES = frozenset({"vjepa", "vjepa_vggt_scene", "vjepa_utonia_scene"})
LPOS_MODES = frozenset({"lpos"})
RAW_MODES = frozenset(
    {
        "raw",
        "raw_all_noise_future_loss",
        "raw_full_frame",
        "utonia_local_scene",
        "utonia_object_scene_attention",
        "utonia_subtraction",
    }
)


def _parse_vjepa_input_size(value: str | int) -> tuple[int, int]:
    """Parse a square size (384) or an explicit HxW size (384x672)."""
    text = str(value).strip().lower().replace(" ", "")
    if "x" in text:
        height_text, width_text = text.split("x", 1)
        height, width = int(height_text), int(width_text)
    else:
        height = width = int(text)
    if height <= 0 or width <= 0:
        raise ValueError(f"V-JEPA input size must be positive, got {value!r}")
    if height % 16 or width % 16:
        raise ValueError(
            f"V-JEPA input size must be divisible by patch size 16, got {height}x{width}"
        )
    return height, width


def _init_wandb(run_dir: Path, config: dict[str, Any]):
    mode = os.environ.get("WANDB_MODE", "disabled").strip().lower()
    if mode in {"", "disabled", "offline"}:
        return None
    import wandb

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "physrvg-full-sa"),
        name=os.environ.get("WANDB_NAME", config["arguments"]["run_name"]),
        mode=mode,
        config=config,
        dir=str(run_dir),
        job_type="train",
    )
    (run_dir / "wandb_run.json").write_text(
        json.dumps(
            {
                "id": run.id,
                "name": run.name,
                "project": run.project,
                "entity": run.entity,
                "url": run.url,
                "mode": mode,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return run


def _utonia_gradient_diagnostics(model: torch.nn.Module) -> dict[str, float]:
    """Report whether the Utonia query path receives a real training gradient."""

    groups = {
        "utonia_scene_grad_query_tokens_l2": ("utonia_scene_projector.query_tokens",),
        "utonia_scene_grad_projector_l2": ("utonia_scene_projector.",),
        "utonia_scene_grad_scene_gate_l2": (".scene_gate",),
        # PEFT registers adapters on the shared base projection module, so
        # their de-duplicated names are commonly ``...attn2.to_q.lora_B.
        # utonia_scene.weight`` rather than ``...scene_attn...``.
        "utonia_scene_grad_scene_lora_l2": (
            ".lora_A.utonia_scene.",
            ".lora_B.utonia_scene.",
        ),
    }
    diagnostics: dict[str, float] = {}
    named_parameters = list(model.named_parameters())
    for metric_name, patterns in groups.items():
        squared_sum = 0.0
        matched = 0
        for name, parameter in named_parameters:
            if not any(pattern in name for pattern in patterns):
                continue
            matched += 1
            if parameter.grad is not None:
                squared_sum += float(parameter.grad.detach().float().square().sum().item())
        diagnostics[metric_name] = math.sqrt(squared_sum) if matched else 0.0
    gate_values = [
        parameter.detach().float().abs().mean()
        for name, parameter in named_parameters
        if ".scene_gate" in name
    ]
    diagnostics["utonia_scene_gate_abs_mean"] = (
        float(torch.stack(gate_values).mean().item()) if gate_values else 0.0
    )
    return diagnostics


def _prepare_utonia_local_scene_inputs(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten strict `[B,8,K,64,*]` local inputs for the generic scene API."""

    features = batch["utonia_local_features"].to(
        device=device, dtype=torch.bfloat16, non_blocking=True
    )
    relation = batch["utonia_local_relation"].to(
        device=device, dtype=torch.bfloat16, non_blocking=True
    )
    distance = batch["utonia_local_distance"].to(
        device=device, dtype=torch.bfloat16, non_blocking=True
    )
    motion = batch["utonia_local_motion"].to(
        device=device, dtype=torch.bfloat16, non_blocking=True
    )
    time = batch["utonia_local_time"].to(
        device=device, dtype=torch.bfloat16, non_blocking=True
    )
    mask = batch["utonia_local_mask"].to(device=device, non_blocking=True).bool()
    if features.ndim != 5 or tuple(features.shape[:2]) != (features.shape[0], 8):
        raise ValueError(f"local Utonia features must be [B,8,K,64,1386], got {features.shape}")
    if features.shape[3] != 64 or features.shape[-1] != 1386:
        raise ValueError(f"unexpected local Utonia feature dimensions: {features.shape}")
    if relation.shape != features.shape[:-1] + (3,):
        raise ValueError(f"local relation shape mismatch: {relation.shape} vs {features.shape}")
    if distance.shape != features.shape[:-1] + (1,):
        raise ValueError(f"local distance shape mismatch: {distance.shape} vs {features.shape}")
    if motion.shape != features.shape[:3] + (15,):
        raise ValueError(f"local motion shape mismatch: {motion.shape} vs {features.shape}")
    if time.shape != features.shape[:2]:
        raise ValueError(f"local time shape mismatch: {time.shape} vs {features.shape}")
    if mask.shape != features.shape[:-1]:
        raise ValueError(f"local mask shape mismatch: {mask.shape} vs {features.shape}")
    if not bool(mask.flatten(1).any(dim=1).all()):
        raise ValueError("every sample must contain at least one valid local scene token")

    motion = motion.unsqueeze(-2).expand(*features.shape[:-1], motion.shape[-1])
    time = time[:, :, None, None, None].expand(*features.shape[:-1], 1)
    local_inputs = torch.cat((features, relation, distance, motion, time), dim=-1)
    # Preserve the final 1406-D feature axis; flatten only time, object, and
    # local-neighbor axes into the K/V token sequence.
    return local_inputs.flatten(1, 3), mask.flatten(1, 3)


def _prepare_utonia_object_scene_inputs(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move the non-repeated static scene and object trajectory to one GPU."""

    result = {
        "scene_features": batch["utonia_object_scene_features"].to(
            device=device, dtype=torch.bfloat16, non_blocking=True
        ),
        "scene_world": batch["utonia_object_scene_world"].to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
        "scene_mask": batch["utonia_object_scene_mask"].to(
            device=device, non_blocking=True
        ).bool(),
        "object_surface_local": batch["utonia_object_surface_local"].to(
            device=device, dtype=torch.bfloat16, non_blocking=True
        ),
        "object_surface_world": batch["utonia_object_surface_world"].to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
        "object_surface_valid": batch["utonia_object_surface_valid"].to(
            device=device, non_blocking=True
        ).bool(),
        "object_motion": batch["utonia_object_motion"].to(
            device=device, dtype=torch.bfloat16, non_blocking=True
        ),
        "object_valid": batch["utonia_object_valid"].to(
            device=device, non_blocking=True
        ).bool(),
        "time": batch["utonia_object_time"].to(
            device=device, dtype=torch.bfloat16, non_blocking=True
        ),
    }
    features = result["scene_features"]
    local = result["object_surface_local"]
    if features.ndim != 3 or int(features.shape[-1]) != 1386:
        raise ValueError(f"object-conditioned scene features must be [B,M,1386]: {features.shape}")
    if local.ndim != 4 or tuple(local.shape[-2:]) != (64, 3):
        raise ValueError(f"object local surfaces must be [B,K,64,3]: {local.shape}")
    if not bool(result["scene_mask"].any(dim=1).all()):
        raise ValueError("every batch row requires at least one valid static scene token")
    if not bool(result["object_valid"].flatten(1).any(dim=1).all()):
        raise ValueError("every batch row requires at least one valid context object")
    return result


def _prepare_utonia_subtraction_inputs(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    variant: str,
) -> dict[str, torch.Tensor]:
    """Move only tensors consumed by the selected subtraction variant.

    The CPU dataset schema remains identical across candidates so ordering and
    sample filtering cannot drift.  Motion-only does not transfer scene tensors
    to the GPU; geometry omits the 1386-D Utonia features; only the final
    candidate transfers the full local-feature source.
    """

    result = {
        "object_motion": batch["utonia_object_motion"].to(
            device=device, dtype=torch.bfloat16, non_blocking=True
        ),
        "object_valid": batch["utonia_object_valid"].to(
            device=device, non_blocking=True
        ).bool(),
        "time": batch["utonia_object_time"].to(
            device=device, dtype=torch.bfloat16, non_blocking=True
        ),
    }
    if variant != "motion":
        result.update(
            {
                "scene_world": batch["utonia_object_scene_world"].to(
                    device=device, dtype=torch.float32, non_blocking=True
                ),
                "scene_mask": batch["utonia_object_scene_mask"].to(
                    device=device, non_blocking=True
                ).bool(),
                "object_surface_world": batch["utonia_object_surface_world"].to(
                    device=device, dtype=torch.float32, non_blocking=True
                ),
                "object_surface_valid": batch["utonia_object_surface_valid"].to(
                    device=device, non_blocking=True
                ).bool(),
            }
        )
    if variant == "utonia":
        result["scene_features"] = batch["utonia_object_scene_features"].to(
            device=device, dtype=torch.bfloat16, non_blocking=True
        )
    return result


def _import_module_from_root(module_name: str, root: str):
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Module root does not exist: {root_path}")
    root_text = str(root_path)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    module = importlib.import_module(module_name)
    module_path = Path(module.__file__).resolve()
    if root_path != module_path.parent and root_path not in module_path.parents:
        raise ImportError(
            f"Imported {module_name!r} from {module_path}, expected it under {root_path}"
        )
    return module


def _load_vjepa_encoder(
    repo: str,
    checkpoint: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
):
    backbones = _import_module_from_root("src.hub.backbones", repo)
    encoder, _ = backbones.vjepa2_1_vit_large_384(pretrained=False)
    payload = torch.load(
        Path(checkpoint).expanduser().resolve(),
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported V-JEPA checkpoint object: {type(payload)!r}")
    state = None
    for key in ("ema_encoder", "target_encoder", "encoder", "model", "state_dict"):
        candidate = payload.get(key)
        if isinstance(candidate, dict) and candidate:
            state = candidate
            break
    if state is None and payload and all(
        isinstance(value, torch.Tensor) for value in payload.values()
    ):
        state = payload
    if state is None:
        raise KeyError("V-JEPA checkpoint has no encoder state")
    normalized = {}
    for key, value in state.items():
        name = str(key)
        while name.startswith("module."):
            name = name[len("module.") :]
        if name.startswith("backbone."):
            name = name[len("backbone.") :]
        normalized[name] = value
    encoder.load_state_dict(normalized, strict=True)
    del normalized, state, payload
    encoder.requires_grad_(False)
    encoder.eval()
    # The encoder is frozen, but its activations still participate in the
    # gradient path for the predicted video. Keep it in the training dtype to
    # avoid duplicating the large float32 encoder on each 48 GiB GPU.
    encoder.to(device=device, dtype=dtype)
    return encoder


def _load_tiny_vae(root: str, checkpoint: str, device: torch.device, dtype: torch.dtype):
    taehv = _import_module_from_root("taehv", root)
    decoder = taehv.TAEHV(str(Path(checkpoint).expanduser().resolve()))
    decoder.requires_grad_(False)
    decoder.eval()
    decoder.to(device=device, dtype=dtype)
    return decoder, taehv.apply_model_with_memblocks


class VJEPAAuxiliary:
    def __init__(self, args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
        self.weight = float(args.vjepa_loss_weight)
        self.sigma_min = float(args.vjepa_sigma_min)
        self.sigma_max = float(args.vjepa_sigma_max)
        self.every_n = int(args.vjepa_every_n_forwards)
        self.num_frames = int(args.vjepa_num_frames)
        self.input_height, self.input_width = _parse_vjepa_input_size(
            args.vjepa_input_size
        )
        self.range_weight = float(args.vjepa_range_penalty_weight)
        self.frame_sampling = str(args.vjepa_frame_sampling)
        self.local_probability = float(args.vjepa_local_sampling_probability)
        self.local_context = int(args.vjepa_local_context_frames)
        self.forward_count = 0
        self.tiny_vae, self.tiny_vae_apply = _load_tiny_vae(
            args.tiny_vae_root, args.tiny_vae_checkpoint, device, dtype
        )
        self.encoder = _load_vjepa_encoder(
            args.vjepa_repo,
            args.vjepa_checkpoint,
            device,
            dtype=dtype,
        )

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        latent_ntchw = latents.permute(0, 2, 1, 3, 4).contiguous()
        device_type = latent_ntchw.device.type
        with torch.autocast(
            device_type=device_type,
            dtype=latents.dtype,
            enabled=device_type == "cuda",
        ):
            video = self.tiny_vae_apply(
                self.tiny_vae.decoder,
                latent_ntchw,
                False,
                False,
            )
            if self.tiny_vae.patch_size > 1:
                video = F.pixel_shuffle(video, self.tiny_vae.patch_size)
        skip_trim = self.tiny_vae.is_cogvideox and latent_ntchw.shape[1] % 2 == 0
        if not skip_trim:
            video = video[:, self.tiny_vae.frames_to_trim :]
        return video

    def _select_indices(self, time_steps: int, device: torch.device) -> tuple[torch.Tensor, bool]:
        if self.frame_sampling == "full":
            indices = torch.arange(time_steps, device=device, dtype=torch.long)
            if indices.numel() % 2:
                indices = torch.cat((indices, indices[-1:]))
            return indices, False
        use_local = self.frame_sampling == "local"
        if self.frame_sampling == "mixed":
            use_local = bool(
                torch.rand((), device=device).item() < self.local_probability
            )
        context_cutoff = 7
        if use_local and time_steps >= self.num_frames:
            max_start = time_steps - self.num_frames
            start = max(0, min(max_start, context_cutoff - self.local_context + 1))
            return torch.arange(
                start, start + self.num_frames, device=device, dtype=torch.long
            ), True
        return (
            torch.linspace(
                0, time_steps - 1, steps=self.num_frames, device=device
            ).round().to(torch.long),
            False,
        )

    def _preprocess(self, video: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        frames = video.index_select(1, indices).float()
        batch, selected, channels, height, width = frames.shape
        if self.input_height == self.input_width:
            # Preserve the original 384x384 preprocessing exactly.
            resize_short = round((256.0 / 224.0) * self.input_height)
            scale = resize_short / min(height, width)
        else:
            # For rectangular input, resize to cover the requested canvas while
            # preserving the source aspect ratio, then center-crop only if needed.
            scale = max(self.input_height / height, self.input_width / width)
        resized_height = max(self.input_height, round(height * scale))
        resized_width = max(self.input_width, round(width * scale))
        frames = F.interpolate(
            frames.reshape(batch * selected, channels, height, width),
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        top = (resized_height - self.input_height) // 2
        left = (resized_width - self.input_width) // 2
        frames = frames[
            :, :, top : top + self.input_height, left : left + self.input_width
        ]
        mean = frames.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = frames.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        frames = (frames - mean) / std
        return frames.view(
            batch, selected, channels, self.input_height, self.input_width
        ).permute(0, 2, 1, 3, 4).contiguous()

    def _encode(self, video: torch.Tensor) -> torch.Tensor:
        encoder_dtype = next(self.encoder.parameters()).dtype
        if video.device.type == "cuda":
            with torch.autocast(
                device_type="cuda",
                dtype=encoder_dtype,
                enabled=encoder_dtype in (torch.float16, torch.bfloat16),
            ):
                features = self.encoder(video)
        else:
            features = self.encoder(video.to(dtype=encoder_dtype))
        if not isinstance(features, torch.Tensor) or features.ndim < 2:
            raise TypeError(f"V-JEPA encoder returned {type(features)!r}")
        return features

    @staticmethod
    def _future_tokens(
        features: torch.Tensor, indices: torch.Tensor, context_cutoff: int
    ) -> tuple[torch.Tensor, float]:
        tubelet = 2
        if indices.numel() % tubelet:
            raise ValueError("V-JEPA frame selection must contain even frames")
        temporal = int(indices.numel()) // tubelet
        if features.ndim != 3 or features.shape[1] % temporal:
            raise ValueError(
                f"Cannot reshape V-JEPA features {tuple(features.shape)} into {temporal} temporal tokens"
            )
        tubelets = indices.view(temporal, tubelet)
        future = (tubelets > int(context_cutoff)).all(dim=1)
        if not bool(future.any()):
            raise ValueError("V-JEPA sampling produced no future-only tubelets")
        spatial = features.shape[1] // temporal
        grouped = features.view(features.shape[0], temporal, spatial, features.shape[-1])
        return grouped[:, future], float(future.float().mean().item())

    def compute(
        self,
        pred_x0: torch.Tensor,
        target_x0: torch.Tensor,
        sigma: torch.Tensor,
        *,
        force: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        self.forward_count += 1
        sigma_value = float(sigma.detach().float().item())
        applies_local = force or (
            self.forward_count % self.every_n == 0
            and self.sigma_min <= sigma_value <= self.sigma_max
        )
        # Every DDP rank must take the same auxiliary branch; otherwise one
        # rank can enter the V-JEPA graph while another skips it and the
        # reducer waits forever for mismatched gradient collectives.
        if dist.is_initialized():
            apply_tensor = torch.tensor(
                [int(applies_local)], device=pred_x0.device, dtype=torch.int32
            )
            dist.broadcast(apply_tensor, src=0)
            applies = bool(apply_tensor.item())
        else:
            applies = applies_local
        metrics = {
            "loss_vjepa": 0.0,
            "loss_vjepa_range": 0.0,
            "vjepa_weighted_contribution": 0.0,
            "vjepa_applied": float(applies),
            "vjepa_sigma": sigma_value,
        }
        if not applies:
            return pred_x0.sum() * 0.0, metrics

        with torch.no_grad():
            target_raw = self._decode(target_x0)
            indices, sampling_local = self._select_indices(
                int(target_raw.shape[1]), target_raw.device
            )
            target_input = self._preprocess(target_raw.clamp(0.0, 1.0), indices)
            target_features = self._encode(target_input).detach()

        pred_raw = self._decode(pred_x0)
        pred_clipped = pred_raw.clamp(0.0, 1.0)
        pred_video = pred_raw + (pred_clipped - pred_raw).detach()
        range_loss = F.relu(-pred_raw).square().mean() + F.relu(
            pred_raw - 1.0
        ).square().mean()
        pred_input = self._preprocess(pred_video, indices)
        pred_features = self._encode(pred_input)
        if pred_features.shape != target_features.shape:
            raise RuntimeError(
                f"V-JEPA feature shape mismatch: {pred_features.shape}/{target_features.shape}"
            )
        pred_features, future_fraction = self._future_tokens(
            pred_features, indices, 7
        )
        target_features, _ = self._future_tokens(target_features, indices, 7)
        pred_features = F.normalize(pred_features.float(), dim=-1)
        target_features = F.normalize(target_features.float(), dim=-1)
        feature_loss = (pred_features - target_features).square().sum(dim=-1).mean()
        combined = feature_loss + self.range_weight * range_loss
        weighted = self.weight * combined
        metrics.update(
            {
                "loss_vjepa": float(feature_loss.detach().item()),
                "loss_vjepa_range": float(range_loss.detach().item()),
                "loss_vjepa_combined": float(combined.detach().item()),
                "vjepa_weighted_contribution": float(weighted.detach().item()),
                "vjepa_sampling_local": float(sampling_local),
                "vjepa_future_token_fraction": future_fraction,
            }
        )
        return weighted, metrics


class LPosAuxiliary:
    """Differentiable position loss for the direct x0 prediction path.

    The official RL metric extracts trajectories with SAM2 after decoding a
    generated video, so it cannot provide gradients to the DiT.  Training
    instead uses the same object-mask supervision already produced for the
    PyBullet cache: the GT object appearance in context frame 7 forms a
    frozen template, and a soft template match on decoded ``pred_x0`` gives a
    differentiable predicted centroid for each future frame.
    """

    def __init__(self, args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
        self.weight = float(args.lpos_loss_weight)
        self.input_height, self.input_width = _parse_vjepa_input_size(args.lpos_input_size)
        self.context_frame = int(args.lpos_context_frame)
        self.temperature = float(args.lpos_temperature)
        self.object_index = int(args.lpos_object_index)
        self.mask_root = Path(args.lpos_mask_cache_root).expanduser().resolve()
        self.tiny_vae, self.tiny_vae_apply = _load_tiny_vae(
            args.tiny_vae_root, args.tiny_vae_checkpoint, device, dtype
        )
        self._mask_cache: dict[str, torch.Tensor] = {}

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        latent_ntchw = latents.permute(0, 2, 1, 3, 4).contiguous()
        device_type = latent_ntchw.device.type
        with torch.autocast(
            device_type=device_type,
            dtype=latents.dtype,
            enabled=device_type == "cuda",
        ):
            video = self.tiny_vae_apply(
                self.tiny_vae.decoder,
                latent_ntchw,
                False,
                False,
            )
            if self.tiny_vae.patch_size > 1:
                video = F.pixel_shuffle(video, self.tiny_vae.patch_size)
        skip_trim = self.tiny_vae.is_cogvideox and latent_ntchw.shape[1] % 2 == 0
        if not skip_trim:
            video = video[:, self.tiny_vae.frames_to_trim :]
        return video

    def _load_mask(self, logical_key: str, device: torch.device) -> torch.Tensor:
        cached = self._mask_cache.get(logical_key)
        if cached is None:
            path = self.mask_root / "cases" / logical_key / "object_masks.npz"
            if not path.is_file():
                raise FileNotFoundError(f"missing L_pos object mask cache entry: {path}")
            with np.load(path, allow_pickle=False) as arrays:
                masks = np.asarray(arrays["masks_othw"], dtype=np.float32)
            if masks.ndim != 4 or not 0 <= self.object_index < masks.shape[0]:
                raise ValueError(f"invalid L_pos mask shape/object index for {logical_key}: {masks.shape}")
            cached = torch.from_numpy(masks[self.object_index])
            self._mask_cache[logical_key] = cached
        return cached.to(device=device, dtype=torch.float32)

    @staticmethod
    def _resize_video(video: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, frames, channels, source_height, source_width = video.shape
        if (source_height, source_width) == (height, width):
            return video
        resized = F.interpolate(
            video.reshape(batch * frames, channels, source_height, source_width),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return resized.reshape(batch, frames, channels, height, width)

    def compute(
        self,
        pred_x0: torch.Tensor,
        target_x0: torch.Tensor,
        logical_keys: list[str],
        *,
        context_frames: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        with torch.no_grad():
            target_raw = self._decode(target_x0)
        pred_raw = self._decode(pred_x0)
        pred_clipped = pred_raw.clamp(0.0, 1.0)
        pred_video = pred_raw + (pred_clipped - pred_raw).detach()
        target_video = target_raw.clamp(0.0, 1.0)
        pred_video = self._resize_video(pred_video, self.input_height, self.input_width)
        target_video = self._resize_video(target_video, self.input_height, self.input_width)

        frames = int(pred_video.shape[1])
        context_frame = min(max(self.context_frame, 0), frames - 1)
        future_start = min(max(int(context_frames), context_frame + 1), frames - 1)
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, self.input_height, device=pred_video.device),
            torch.linspace(-1.0, 1.0, self.input_width, device=pred_video.device),
            indexing="ij",
        )
        coords = torch.stack((xx, yy), dim=0).view(2, -1)
        losses = []
        valid_frames = []
        for sample_index, logical_key in enumerate(logical_keys):
            mask = self._load_mask(logical_key, pred_video.device)
            if mask.shape[0] != frames:
                raise ValueError(
                    f"L_pos frame mismatch for {logical_key}: mask={mask.shape[0]} video={frames}"
                )
            mask = F.interpolate(
                mask[:, None],
                size=(self.input_height, self.input_width),
                mode="area",
            )[:, 0]
            context_mask = mask[context_frame]
            context_mass = context_mask.sum().clamp_min(1.0e-6)
            template = (
                target_video[sample_index, context_frame] * context_mask[None]
            ).sum(dim=(-1, -2)) / context_mass
            pred_future = pred_video[sample_index, future_start:]
            distance = (
                pred_future - template[None, :, None, None]
            ).square().mean(dim=1)
            soft_object = torch.exp(-distance / max(self.temperature, 1.0e-6))
            pred_mass = soft_object.flatten(1).sum(-1).clamp_min(1.0e-6)
            pred_centroid = (
                soft_object.flatten(1)[:, None, :] * coords[None, :, :]
            ).sum(-1) / pred_mass[:, None]
            gt_mask = mask[future_start:]
            gt_mass = gt_mask.flatten(1).sum(-1)
            gt_centroid = (
                gt_mask.flatten(1)[:, None, :] * coords[None, :, :]
            ).sum(-1) / gt_mass.clamp_min(1.0e-6)[:, None]
            valid = gt_mass > 1.0e-4
            if bool(valid.any()):
                per_frame = torch.sqrt(
                    (pred_centroid - gt_centroid).square().sum(dim=1) + 1.0e-8
                )
                losses.append(per_frame[valid].mean())
                valid_frames.append(float(valid.sum().item()))
            else:
                losses.append(pred_x0[sample_index].sum() * 0.0)
                valid_frames.append(0.0)

        raw_loss = torch.stack(losses).mean()
        weighted = self.weight * raw_loss
        return weighted, {
            "loss_lpos": float(raw_loss.detach().item()),
            "lpos_weighted_contribution": float(weighted.detach().item()),
            "lpos_valid_future_frames": float(sum(valid_frames) / max(1, len(valid_frames))),
            "lpos_context_frame": float(context_frame),
        }


class ProbeCollector:
    def __init__(
        self,
        selected_heads_by_block: dict[int, tuple[int, ...]],
        query_rows: torch.Tensor,
        query_weights: torch.Tensor,
        grid: tuple[int, int, int],
        fixed_queries: dict[int, torch.Tensor] | None = None,
    ):
        self.selected = selected_heads_by_block
        self.query_rows = query_rows
        self.query_weights = query_weights
        self.grid = tuple(map(int, grid))
        self.fixed_queries = fixed_queries or {}
        self.maps: list[torch.Tensor] = []
        self.queries: dict[int, torch.Tensor] = {}

    def record(self, block_id: int, q: torch.Tensor, k: torch.Tensor) -> None:
        heads = self.selected.get(block_id, ())
        if not heads:
            return
        fixed = self.fixed_queries.get(block_id)
        maps = fixed_query_frame_head_probabilities(
            q,
            k,
            head_indices=heads,
            query_rows=self.query_rows,
            query_weights=self.query_weights,
            grid=self.grid,
            num_heads=q.shape[-1] // (q.shape[-1] // 24),
            fixed_query=fixed,
        )
        self.maps.append(maps)
        q_heads = q.reshape(q.shape[0], q.shape[1], 24, -1).permute(0, 2, 1, 3)
        head_tensor = torch.as_tensor(heads, device=q.device, dtype=torch.long)
        rows = self.query_rows.to(device=q.device)
        self.queries[block_id] = q_heads.index_select(1, head_tensor).index_select(2, rows)

    def finalize(self) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        expected = sum(len(v) for v in self.selected.values())
        if len(self.maps) != len(self.selected):
            raise RuntimeError(
                f"probe captured {len(self.maps)} selected blocks, expected {len(self.selected)}"
            )
        maps = torch.cat(self.maps, dim=1)
        if maps.shape[1] != expected:
            raise RuntimeError(f"probe captured {maps.shape[1]} heads, expected {expected}")
        return maps, self.queries


class ProbeAttentionProcessor:
    def __init__(self, block_id: int, collector_ref: dict[str, Any]):
        self.block_id = int(block_id)
        self.collector_ref = collector_ref

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        rotary_emb=None,
        **kwargs,
    ):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = attn.norm_q(query).unflatten(2, (attn.heads, -1))
        key = attn.norm_k(key).unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))
        if rotary_emb is not None:
            cos, sin = rotary_emb
            cos = cos[..., 0::2]
            sin = sin[..., 1::2]

            def apply_rope(x):
                x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
                result = torch.empty_like(x)
                result[..., 0::2] = x1 * cos - x2 * sin
                result[..., 1::2] = x1 * sin + x2 * cos
                return result.type_as(x)

            query = apply_rope(query)
            key = apply_rope(key)
        collector = self.collector_ref.get("collector")
        if collector is not None:
            collector.record(
                self.block_id,
                query.flatten(2),
                key.flatten(2),
            )
        from diffusers.models.attention_dispatch import dispatch_attention_fn

        output = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=None,
        )
        output = output.flatten(2, 3).type_as(query)
        output = attn.to_out[0](output)
        return attn.to_out[1](output)


class FrozenProbe:
    def __init__(
        self,
        args: argparse.Namespace,
        device: torch.device,
        dtype: torch.dtype,
        selected_heads_by_block: dict[int, tuple[int, ...]],
    ):
        from fastvideo.models.wan_v2v.model_wan_v2v import WanTransformer3DModel

        self.model = WanTransformer3DModel.from_pretrained(
            args.model_id, subfolder="transformer", torch_dtype=dtype
        )
        if len(self.model.blocks) != args.expected_num_blocks:
            raise RuntimeError("Frozen probe block count mismatch")
        self.model.requires_grad_(False)
        self.model.eval().to(device)
        self.selected = selected_heads_by_block
        self.refs = {"collector": None}
        for block_id in self.selected:
            self.model.blocks[block_id].attn1.set_processor(
                ProbeAttentionProcessor(block_id, self.refs)
            )
        self.device = device

    def run(
        self,
        latents: torch.Tensor,
        token_timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        *,
        query_rows: torch.Tensor,
        query_weights: torch.Tensor,
        grid: tuple[int, int, int],
        fixed_queries: dict[int, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        collector = ProbeCollector(
            self.selected,
            query_rows,
            query_weights,
            grid,
            fixed_queries=fixed_queries,
        )
        self.refs["collector"] = collector
        try:
            save_on_cpu_pin_memory = os.environ.get(
                "PHYSRVG_SAVE_ON_CPU_PIN_MEMORY", "1"
            ).strip().lower() not in {"0", "false", "no", "off"}
            offload_context = (
                torch.autograd.graph.save_on_cpu(
                    pin_memory=save_on_cpu_pin_memory
                )
                if fixed_queries is not None and save_on_cpu_pin_memory
                else nullcontext()
            )
            with offload_context:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    self.model(
                        hidden_states=latents,
                        timestep=token_timesteps,
                        encoder_hidden_states=prompt_embeds,
                        encoder_hidden_states_image=None,
                        return_dict=False,
                    )
        finally:
            self.refs["collector"] = None
        return collector.finalize()


def _load_head_selection(path: str, weight_power: float) -> tuple[dict[int, tuple[int, ...]], torch.Tensor, dict]:
    config_path = Path(path).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    targets = config.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError(f"invalid head target list: {config_path}")
    source_path = Path(str(config["selection_source"])).expanduser().resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    scores = {
        (int(entry["block"]), int(entry["head"])): float(entry["pck32"])
        for entry in source["entries"]
        if int(entry.get("step", config["ranking_step"])) == int(config["ranking_step"])
    }
    pairs = [(int(item["block"]), int(item["head"])) for item in targets]
    missing = [pair for pair in pairs if pair not in scores]
    if missing:
        raise KeyError(f"PCK source missing selected heads: {missing[:5]}")
    raw = torch.tensor([scores[pair] for pair in pairs], dtype=torch.float64)
    weights = (raw / raw.max()).pow(float(weight_power))
    weights = (weights / weights.sum()).to(torch.float32)
    selected: dict[int, list[int]] = {}
    for block, head in pairs:
        selected.setdefault(block, []).append(head)
    selected_final = {block: tuple(heads) for block, heads in sorted(selected.items())}
    audit = {
        "config": str(config_path),
        "selection_source": str(source_path),
        "weight_power": float(weight_power),
        "head_pairs": [list(pair) for pair in pairs],
        "weights": weights.tolist(),
    }
    return selected_final, weights, audit


def _latent_mask_record_reason(
    cache_root: Path,
    logical_key: str,
    *,
    grid: tuple[int, int, int],
    query_frame: int,
    object_index: int,
) -> str | None:
    """Return a concise reason when a record cannot provide this auxiliary loss."""
    path = cache_root / "cases" / logical_key / "object_masks.npz"
    try:
        if not path.is_file():
            return "missing_cache_entry"
        with np.load(path, allow_pickle=False) as arrays:
            masks = np.asarray(arrays["masks_othw"], dtype=np.float32)
        if masks.ndim != 4:
            return f"invalid_mask_ndim_{masks.ndim}"
        if not 0 <= object_index < masks.shape[0]:
            return f"invalid_object_index_{object_index}"
        masks = masks[object_index]
        frames, height, width = grid
        expected_frames = 4 * (frames - 1) + 1
        if masks.shape[0] != expected_frames:
            return f"mask_frame_count_{masks.shape[0]}_expected_{expected_frames}"
        if not 0 <= query_frame < frames:
            return f"query_frame_{query_frame}_outside_0_{frames - 1}"

        sampled = torch.from_numpy(masks[::4]).unsqueeze(0)
        occupancy = F.adaptive_avg_pool2d(sampled, (height, width))[0]
        flat = occupancy.flatten(1)
        if not bool((flat[query_frame] > 0).any()):
            return "no_source_query_tokens"
        frame_ids = torch.arange(frames)
        valid = (flat.sum(-1) > 0) & (frame_ids > query_frame)
        if not bool(valid.any()):
            return "no_valid_future_frames"
    except Exception as exc:
        return f"{type(exc).__name__}_{exc}"
    return None


def _logical_keys_digest(logical_keys: list[str]) -> str:
    payload = "\n".join(str(key) for key in logical_keys).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _filter_latent_mask_records_from_manifest(
    dataset,
    manifest_path: Path,
    cache_root: Path,
    *,
    grid: tuple[int, int, int],
    query_frame: int,
    object_index: int,
    rank: int,
) -> dict[str, Any]:
    """Reuse a previously validated keep-list without a DDP-side cache scan."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise RuntimeError(f"unsupported latent-mask preflight manifest: {manifest_path}")
    if Path(str(payload.get("cache_root", ""))).expanduser().resolve() != cache_root.resolve():
        raise RuntimeError(
            "latent-mask preflight manifest cache root mismatch: "
            f"manifest={payload.get('cache_root')} current={cache_root}"
        )
    if tuple(map(int, payload.get("grid", []))) != tuple(map(int, grid)):
        raise RuntimeError(
            "latent-mask preflight manifest grid mismatch: "
            f"manifest={payload.get('grid')} current={list(grid)}"
        )
    if int(payload.get("query_frame", -1)) != int(query_frame):
        raise RuntimeError("latent-mask preflight manifest query_frame mismatch")
    if int(payload.get("object_index", -1)) != int(object_index):
        raise RuntimeError("latent-mask preflight manifest object_index mismatch")

    cache_config_path = cache_root / "cache_config.json"
    expected_cache_config_sha256 = payload.get("cache_config_sha256")
    if expected_cache_config_sha256:
        if not cache_config_path.is_file():
            raise RuntimeError(f"missing cache_config.json for manifest validation: {cache_config_path}")
        actual_cache_config_sha256 = hashlib.sha256(cache_config_path.read_bytes()).hexdigest()
        if actual_cache_config_sha256 != expected_cache_config_sha256:
            raise RuntimeError(
                "latent-mask cache_config.json changed since preflight manifest was created"
            )

    records = list(dataset.records)
    current_keys = [str(record["logical_key"]) for record in records]
    expected_num_samples = int(payload.get("dataset_num_samples", -1))
    if len(current_keys) != expected_num_samples:
        raise RuntimeError(
            "latent-mask preflight manifest dataset size mismatch: "
            f"manifest={expected_num_samples} current={len(current_keys)}"
        )
    expected_keys_digest = str(payload.get("dataset_logical_keys_sha256", ""))
    if expected_keys_digest and _logical_keys_digest(current_keys) != expected_keys_digest:
        raise RuntimeError("latent-mask preflight manifest dataset logical-key digest mismatch")

    kept_keys = [str(key) for key in payload.get("kept_logical_keys", [])]
    kept_key_set = set(kept_keys)
    if len(kept_key_set) != len(kept_keys):
        raise RuntimeError("latent-mask preflight manifest contains duplicate kept logical keys")
    if not kept_key_set.issubset(set(current_keys)):
        raise RuntimeError("latent-mask preflight manifest contains unknown logical keys")
    dataset.records = [record for record in records if str(record["logical_key"]) in kept_key_set]
    if not dataset.records:
        raise RuntimeError("latent-mask preflight manifest keeps no training records")
    if len(dataset.records) != int(payload.get("kept_count", len(kept_keys))):
        raise RuntimeError("latent-mask preflight manifest kept-count mismatch")

    audit = {
        "original_num_samples": len(records),
        "kept_num_samples": len(dataset.records),
        "dropped_num_samples": len(records) - len(dataset.records),
        "grid": list(grid),
        "query_frame": int(query_frame),
        "object_index": int(object_index),
        "manifest_path": str(manifest_path),
        "manifest_source_run": payload.get("source_successful_run"),
        "dropped_examples": payload.get("dropped_examples", []) if rank == 0 else [],
    }
    if rank == 0:
        print(
            "latent-mask preflight: "
            f"reused_manifest kept={audit['kept_num_samples']} "
            f"dropped={audit['dropped_num_samples']}",
            flush=True,
        )
    return audit


def _filter_latent_mask_records(
    dataset,
    cache_root: Path,
    *,
    grid: tuple[int, int, int],
    query_frame: int,
    object_index: int,
    rank: int,
    device: torch.device,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Preflight masks once and broadcast the keep list to every DDP rank."""
    if manifest_path is not None:
        return _filter_latent_mask_records_from_manifest(
            dataset,
            manifest_path,
            cache_root,
            grid=grid,
            query_frame=query_frame,
            object_index=object_index,
            rank=rank,
        )
    records = list(dataset.records)
    keep_flags = torch.empty(len(records), dtype=torch.uint8, device=device)
    dropped_examples = []
    if rank == 0:
        for index, record in enumerate(records):
            reason = _latent_mask_record_reason(
                cache_root,
                record["logical_key"],
                grid=grid,
                query_frame=query_frame,
                object_index=object_index,
            )
            if reason is None:
                keep_flags[index] = 1
            else:
                keep_flags[index] = 0
                dropped_examples.append(
                    {
                        "index": index,
                        "logical_key": record["logical_key"],
                        "reason": reason,
                    }
                )
    if dist.is_initialized():
        dist.broadcast(keep_flags, src=0)

    keep = keep_flags.cpu().tolist()
    dataset.records = [record for record, flag in zip(records, keep) if flag]
    if not dataset.records:
        raise RuntimeError("latent-mask preflight rejected every training record")

    audit = {
        "original_num_samples": len(records),
        "kept_num_samples": len(dataset.records),
        "dropped_num_samples": len(records) - len(dataset.records),
        "grid": list(grid),
        "query_frame": int(query_frame),
        "object_index": int(object_index),
        "dropped_examples": dropped_examples if rank == 0 else [],
    }
    if rank == 0:
        print(
            "latent-mask preflight: "
            f"kept={audit['kept_num_samples']} "
            f"dropped={audit['dropped_num_samples']}",
            flush=True,
        )
    return audit


def _audit_collision_cache(dataset, cache_root: Path) -> dict[str, Any]:
    """Check collision labels before the first batch touches the cache."""
    records = list(dataset.records)
    missing = [
        str(record["logical_key"])
        for record in records
        if not (
            cache_root / "cases" / str(record["logical_key"]) / "collision_supervision.npz"
        ).is_file()
    ]
    if missing:
        raise RuntimeError(
            "collision supervision cache is incomplete for the current dataset: "
            f"missing={len(missing)}/{len(records)} examples={missing[:8]}"
        )
    return {
        "cache_root": str(cache_root),
        "num_samples": len(records),
        "missing_num_samples": 0,
    }


def _mask_occupancy(
    cache_root: Path,
    logical_key: str,
    *,
    grid: tuple[int, int, int],
    query_frame: int,
    object_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    path = cache_root / "cases" / logical_key / "object_masks.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing latent-mask cache entry: {path}")
    with np.load(path) as arrays:
        masks = np.asarray(arrays["masks_othw"], dtype=np.float32)
    if masks.ndim != 4 or not 0 <= object_index < masks.shape[0]:
        raise ValueError(f"invalid mask shape/object index for {logical_key}: {masks.shape}")
    masks = masks[object_index]
    frames, height, width = grid
    if masks.shape[0] != 4 * (frames - 1) + 1:
        raise ValueError(f"mask frame mismatch for {logical_key}: {masks.shape}")
    sampled = torch.from_numpy(masks[::4]).unsqueeze(0)
    occupancy = F.adaptive_avg_pool2d(sampled, (height, width))[0]
    flat = occupancy.flatten(1)
    source = flat[query_frame]
    rows = (source > 0).nonzero(as_tuple=False).flatten()
    if rows.numel() == 0:
        raise RuntimeError(f"mask has no source query tokens: {logical_key}")
    query_rows = query_frame * height * width + rows
    query_weights = source[rows]
    query_weights = query_weights / query_weights.sum().clamp_min(1.0e-12)
    frame_ids = torch.arange(frames)
    valid = (flat.sum(-1) > 0) & (frame_ids > query_frame)
    if not bool(valid.any()):
        raise RuntimeError(f"mask has no valid future frames: {logical_key}")
    audit = {
        "valid_target_frames": float(valid.sum().item()),
        "query_token_count": float(rows.numel()),
        "mask_min_frame_area": float(masks.reshape(masks.shape[0], -1).sum(1).min()),
        "mask_max_frame_area": float(masks.reshape(masks.shape[0], -1).sum(1).max()),
    }
    return occupancy, query_rows, query_weights, valid, audit


def _token_timesteps(
    condition_latents: torch.Tensor,
    timestep: torch.Tensor,
    zero_timestep_latent_frames: int,
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    _, _, frames, height, width = condition_latents.shape
    mask = torch.ones(
        1, 1, frames, height, width, device=condition_latents.device, dtype=condition_latents.dtype
    )
    mask[:, :, :zero_timestep_latent_frames] = 0
    patch_t, patch_h, patch_w = map(int, patch_size)
    if patch_t != 1:
        raise ValueError(f"cached Wan training expects temporal patch size 1, got {patch_size}")
    tokens = (mask[0, 0, :, ::patch_h, ::patch_w] * timestep).flatten()
    return tokens.unsqueeze(0).expand(condition_latents.shape[0], -1)


def _token_timesteps_with_context(
    condition_latents: torch.Tensor,
    context_timestep: torch.Tensor,
    future_timestep: torch.Tensor,
    context_latent_frames: int,
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    """Build token timesteps when context and future use different noise levels."""
    _, _, frames, height, width = condition_latents.shape
    patch_t, patch_h, patch_w = map(int, patch_size)
    if patch_t != 1:
        raise ValueError(f"cached Wan training expects temporal patch size 1, got {patch_size}")
    frame_timesteps = torch.full(
        (frames,),
        future_timestep.reshape(-1)[0],
        device=condition_latents.device,
        dtype=future_timestep.dtype,
    )
    frame_timesteps[:context_latent_frames] = context_timestep.reshape(-1)[0]
    tokens = frame_timesteps[:, None, None].expand(frames, height, width)
    tokens = tokens[::patch_t, ::patch_h, ::patch_w].flatten()
    return tokens.unsqueeze(0).expand(condition_latents.shape[0], -1)


def _context_noise_pair(
    scheduler: FlowMatchEulerDiscreteScheduler,
    future_timestep: torch.Tensor,
    future_sigma: torch.Tensor,
    context_noise_scale: float,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve (context timestep, actual sigma, target sigma) for interpolation.

    Endpoint branches deliberately bypass the scheduler lookup.  The terminal
    zero sigma in ``scheduler.sigmas`` has no paired training timestep, so it
    must never be used to implement lambda=0.
    """
    scale = float(context_noise_scale)
    if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
        raise ValueError(f"context_noise_scale must be in [0, 1], got {context_noise_scale!r}")
    if scale == 0.0:
        zero = torch.zeros_like(future_sigma, device=device)
        return torch.zeros_like(future_timestep, device=device), zero, zero
    if scale == 1.0:
        sigma = future_sigma.to(device=device)
        return future_timestep.to(device=device), sigma, sigma

    paired_count = len(scheduler.timesteps)
    if len(scheduler.sigmas) < paired_count:
        raise RuntimeError("scheduler has fewer sigmas than training timesteps")
    paired_sigmas = scheduler.sigmas[:paired_count].to(device=device, dtype=future_sigma.dtype)
    target = future_sigma.to(device=device) * scale
    nearest = torch.argmin((paired_sigmas - target.reshape(-1)[0]).abs())
    context_sigma = paired_sigmas[nearest].reshape_as(future_sigma)
    context_timestep = scheduler.timesteps[nearest].to(device=device, dtype=future_timestep.dtype).reshape_as(future_timestep)
    return context_timestep, context_sigma, target


def _flow_forward(
    model,
    scheduler: FlowMatchEulerDiscreteScheduler,
    clean_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    *,
    context_latent_frames: int,
    patch_size: tuple[int, int, int],
    fixed_index: int | None = None,
    object_slots: torch.Tensor | None = None,
    object_keep_mask: torch.Tensor | None = None,
    scene_world_points: torch.Tensor | None = None,
    scene_confidence: torch.Tensor | None = None,
    scene_features: torch.Tensor | None = None,
    scene_feature_mask: torch.Tensor | None = None,
    logical_keys: list[str] | None = None,
    collision_cache_root: Path | None = None,
    collision_weight_strength: float = 2.0,
    collision_weight_cap: float = 3.0,
    all_frame_noise: bool = False,
    full_frame_flow: bool = False,
    context_timestep_mode: str = "match_noise",
    context_noise_scale: float | None = None,
    noise_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if context_timestep_mode not in {"match_noise", "force_zero"}:
        raise ValueError(
            f"unsupported context_timestep_mode: {context_timestep_mode}"
        )
    if fixed_index is None:
        index = torch.randint(0, len(scheduler.timesteps), (1,), device=clean_latents.device)
        index_int = int(index.item())
    else:
        index_int = int(fixed_index)
        index = torch.tensor([index_int], device=clean_latents.device)
    timestep = scheduler.timesteps[index]
    sigma = scheduler.sigmas[index_int].to(device=clean_latents.device)
    noise = torch.randn_like(clean_latents) if noise_override is None else noise_override
    if noise.shape != clean_latents.shape:
        raise ValueError(
            f"noise_override shape must match clean_latents: {tuple(noise.shape)} vs {tuple(clean_latents.shape)}"
        )
    noisy = scheduler.scale_noise(clean_latents, timestep, noise)
    explicit_context_noise = context_noise_scale is not None
    if explicit_context_noise:
        context_timestep, context_sigma, context_sigma_target = _context_noise_pair(
            scheduler,
            timestep,
            sigma,
            float(context_noise_scale),
            device=clean_latents.device,
        )
        context_noise = noise[:, :, :context_latent_frames]
        future_noisy = noisy[:, :, context_latent_frames:]
        if float(context_noise_scale) == 0.0:
            # Reuse C's exact mask expression, including operation order.
            c_mask = torch.ones_like(clean_latents[:, :1])
            c_mask[:, :, :context_latent_frames] = 0
            model_input = (1 - c_mask) * clean_latents + c_mask * noisy
        elif float(context_noise_scale) == 1.0:
            # Reuse the exact full-tensor scheduler result used by A.
            model_input = noisy
        else:
            context_clean = clean_latents[:, :, :context_latent_frames]
            context_noisy = (
                (1.0 - context_sigma) * context_clean
                + context_sigma * context_noise
            )
            model_input = torch.cat((context_noisy, future_noisy), dim=2)
        token_timesteps = _token_timesteps_with_context(
            model_input,
            context_timestep,
            timestep,
            context_latent_frames,
            patch_size,
        )
    else:
        context_timestep = torch.zeros_like(timestep)
        context_sigma = torch.zeros_like(sigma)
        context_sigma_target = torch.zeros_like(sigma)
        model_input = None
    noise_start_latent = (
        0 if (all_frame_noise or full_frame_flow) else context_latent_frames
    )
    loss_start_latent = 0 if full_frame_flow else context_latent_frames
    if model_input is None:
        condition_mask = torch.ones_like(clean_latents[:, :1])
        condition_mask[:, :, :noise_start_latent] = 0
        model_input = (1 - condition_mask) * clean_latents + condition_mask * noisy
        zero_timestep_latent_frames = (
            context_latent_frames
            if context_timestep_mode == "force_zero"
            else noise_start_latent
        )
        token_timesteps = _token_timesteps(
            model_input, timestep, zero_timestep_latent_frames, patch_size
        )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        prediction = model(
            hidden_states=model_input,
            timestep=token_timesteps,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_image=None,
            object_slots=object_slots,
            object_keep_mask=object_keep_mask,
            scene_world_points=scene_world_points,
            scene_confidence=scene_confidence,
            scene_features=scene_features,
            scene_feature_mask=scene_feature_mask,
            return_dict=False,
        )[0]
    target = noise - clean_latents
    prediction_future = prediction[:, :, loss_start_latent:].float()
    target_future = target[:, :, loss_start_latent:].float()
    if collision_cache_root is None:
        flow_loss = F.mse_loss(prediction_future, target_future)
    else:
        if logical_keys is None or len(logical_keys) != clean_latents.shape[0]:
            raise ValueError("collision-weighted flow loss requires one logical key per sample")
        weight_np = batch_collision_weights(
            collision_cache_root,
            logical_keys,
            strength=collision_weight_strength,
            cap=collision_weight_cap,
        )
        collision_weight = torch.from_numpy(weight_np).to(
            device=prediction.device,
            dtype=prediction.dtype,
        )
        per_latent_error = (prediction_future - target_future).square().mean(dim=1)
        flow_loss = (
            per_latent_error * collision_weight[:, loss_start_latent:, None, None]
        ).sum() / max(1, per_latent_error.numel())
    return flow_loss, {
        "prediction": prediction,
        "target": target,
        "model_input": model_input,
        "clean_latents": clean_latents,
        "prompt_embeds": prompt_embeds,
        "token_timesteps": token_timesteps,
        "timestep": timestep,
        "sigma": sigma,
        "future_timestep": timestep,
        "future_sigma": sigma,
        "context_timestep": context_timestep,
        "context_sigma": context_sigma,
        "context_sigma_target": context_sigma_target,
        "context_noise_scale": torch.tensor(
            float(context_noise_scale) if explicit_context_noise else float("nan"),
            device=clean_latents.device,
            dtype=sigma.dtype,
        ),
        "context_noise_reused": torch.tensor(1.0 if explicit_context_noise else 0.0, device=clean_latents.device),
        "loss_latent_start": torch.tensor(float(loss_start_latent), device=clean_latents.device),
    }


def _pred_x0(state: dict[str, torch.Tensor], context_latent_frames: int) -> torch.Tensor:
    sigma = state["sigma"]
    while sigma.ndim < state["model_input"].ndim:
        sigma = sigma.unsqueeze(-1)
    predicted = state["model_input"] - sigma * state["prediction"]
    result = predicted.clone()
    result[:, :, :context_latent_frames] = state["clean_latents"][:, :, :context_latent_frames]
    return result


def _latent_mask_loss(
    args: argparse.Namespace,
    probe: FrozenProbe,
    state: dict[str, torch.Tensor],
    logical_keys: list[str],
    head_weights: torch.Tensor,
    mask_cache: Path,
    selected_heads: dict[int, tuple[int, ...]],
    patch_size: tuple[int, int, int],
) -> tuple[torch.Tensor, dict[str, float]]:
    target_x0 = state["clean_latents"].detach()
    pred_x0 = _pred_x0(state, args.context_latent_frames)
    patch_t, patch_h, patch_w = map(int, patch_size)
    if patch_t != 1:
        raise ValueError(f"latent-mask supervision expects temporal patch size 1, got {patch_size}")
    grid = (
        int(target_x0.shape[2]) // patch_t,
        int(target_x0.shape[3]) // patch_h,
        int(target_x0.shape[4]) // patch_w,
    )
    total_losses = []
    diagnostics = []
    for i, logical_key in enumerate(logical_keys):
        occupancy, rows, weights, valid, audit = _mask_occupancy(
            mask_cache,
            logical_key,
            grid=grid,
            query_frame=args.latent_mask_query_frame,
            object_index=args.latent_mask_object_index,
        )
        occupancy = occupancy.to(device=target_x0.device, dtype=torch.float32)
        rows = rows.to(device=target_x0.device)
        weights = weights.to(device=target_x0.device)
        valid = valid.to(device=target_x0.device)
        target_sample = target_x0[i : i + 1]
        pred_sample = pred_x0[i : i + 1]
        prompt_sample = state["prompt_embeds"][i : i + 1]
        epsilon = torch.randn_like(target_sample)
        teacher_probe_input = (1.0 - args.latent_mask_noise_level) * target_sample + args.latent_mask_noise_level * epsilon
        student_probe_input = (1.0 - args.latent_mask_noise_level) * pred_sample + args.latent_mask_noise_level * epsilon
        teacher_probe_input[:, :, :args.context_latent_frames] = target_sample[:, :, :args.context_latent_frames]
        student_probe_input[:, :, :args.context_latent_frames] = target_sample[:, :, :args.context_latent_frames]
        probe_steps = _token_timesteps(
            teacher_probe_input,
            torch.tensor([args.latent_mask_probe_timestep], device=target_sample.device),
            args.context_latent_frames,
            patch_size,
        )
        with torch.no_grad():
            teacher_maps, teacher_queries = probe.run(
                teacher_probe_input,
                probe_steps,
                prompt_sample,
                query_rows=rows,
                query_weights=weights,
                grid=grid,
            )
        student_maps, _ = probe.run(
            student_probe_input,
            probe_steps,
            prompt_sample,
            query_rows=rows,
            query_weights=weights,
            grid=grid,
            fixed_queries=teacher_queries,
        )
        head_mix = head_weights.to(device=student_maps.device, dtype=student_maps.dtype)
        teacher_attention = (teacher_maps * head_mix.view(1, -1, 1, 1)).sum(1)
        student_attention = (student_maps * head_mix.view(1, -1, 1, 1)).sum(1)
        teacher_attention = teacher_attention / teacher_attention.sum(-1, keepdim=True).clamp_min(1.0e-12)
        student_attention = student_attention / student_attention.sum(-1, keepdim=True).clamp_min(1.0e-12)
        target = occupancy.flatten(1)
        target = target / target.sum(-1, keepdim=True).clamp_min(1.0e-12)
        ce = -(target * student_attention.clamp_min(1.0e-12).log()).sum(-1)
        raw_ce = (ce * valid.view(1, -1).to(ce)).sum() / valid.sum().clamp_min(1)
        collision_weight = torch.ones(
            grid[0],
            device=ce.device,
            dtype=ce.dtype,
        )
        if args.collision_cache_root:
            weight_np = batch_collision_weights(
                Path(args.collision_cache_root).expanduser().resolve(),
                [logical_key],
                strength=args.collision_weight_strength,
                cap=args.collision_weight_cap,
            )[0]
            collision_weight = torch.from_numpy(weight_np).to(
                device=ce.device,
                dtype=ce.dtype,
            )
            collision_weight[: args.context_latent_frames] = 1.0
        weighted_ce = ce * collision_weight.view(1, -1)
        loss = float(args.latent_mask_weight) * (
            weighted_ce * valid.view(1, -1).to(ce)
        ).sum() / valid.sum().clamp_min(1)
        total_losses.append(loss)
        support = target > 0
        diagnostics.append({
            **audit,
            "raw_soft_ce": float(raw_ce.detach().item()),
            "collision_weight_mean": float(collision_weight[valid].float().mean().detach().item()),
            "collision_weight_max": float(collision_weight[valid].float().max().detach().item()),
            "collision_active_fraction": float((collision_weight[valid] > 1.0).float().mean().detach().item()),
            "student_attention_mass_in_mask": float((student_attention * support).sum(-1)[0, valid].mean().detach().item()),
            "teacher_attention_mass_in_mask": float((teacher_attention * support).sum(-1)[0, valid].mean().detach().item()),
        })
    loss = torch.stack(total_losses).mean()
    metrics = {
        "loss_latent_mask": float(loss.detach().item()),
        "latent_mask_raw_ce": float(sum(x["raw_soft_ce"] for x in diagnostics) / len(diagnostics)),
        "latent_mask_valid_frames": float(sum(x["valid_target_frames"] for x in diagnostics) / len(diagnostics)),
        "latent_mask_query_tokens": float(sum(x["query_token_count"] for x in diagnostics) / len(diagnostics)),
        "latent_mask_collision_weight_mean": float(sum(x["collision_weight_mean"] for x in diagnostics) / len(diagnostics)),
        "latent_mask_collision_weight_max": float(sum(x["collision_weight_max"] for x in diagnostics) / len(diagnostics)),
        "latent_mask_collision_active_fraction": float(sum(x["collision_active_fraction"] for x in diagnostics) / len(diagnostics)),
        "latent_mask_attention_mass_student": float(sum(x["student_attention_mass_in_mask"] for x in diagnostics) / len(diagnostics)),
        "latent_mask_attention_mass_teacher": float(sum(x["teacher_attention_mass_in_mask"] for x in diagnostics) / len(diagnostics)),
    }
    return loss, metrics


def _build_parser(
    default_mode: str,
    default_dataset_format: str = "manifest",
    default_lora_profile: str = "full_sa",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"PhysRVG Full-SA with {default_mode} auxiliary loss")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--physrvg_dit_checkpoint", required=True)
    parser.add_argument("--resume_lora_checkpoint", default=None)
    parser.add_argument(
        "--freeze_main_lora",
        action="store_true",
        help="freeze the loaded main LoRA so a condition-branch experiment measures only its net contribution",
    )
    parser.add_argument("--self_attention_mode", choices=("full", "context_only"), default="full")
    parser.add_argument("--resume_global_step", type=int, default=0)
    parser.add_argument("--resume_vggt_scene_checkpoint", default=None)
    parser.add_argument("--resume_utonia_scene_checkpoint", default=None)
    parser.add_argument("--pybullet_root", required=True)
    parser.add_argument("--vae_cache_dir", required=True)
    parser.add_argument("--prompt_cache_dir", required=True)
    parser.add_argument(
        "--dataset_sources",
        default=None,
        help="JSON list of named case JSON selections, each with its own caption_field and prompt_cache_dir; cannot be combined with sample_list or caption_field.",
    )
    parser.add_argument(
        "--caption_field",
        default=None,
        help="Read this top-level source metadata field and use prompt_cache_dir instead of case JSON prompt references; requires a case JSON sample_list.",
    )
    parser.add_argument(
        "--sample_list",
        default=None,
        help=(
            "Optional one-sample-directory-per-line train selection. "
            "When set, the list is used as the complete train set."
        ),
    )
    parser.add_argument(
        "--prompt_cache_use_cache_text",
        action="store_true",
        help="Use the positive prompt text stored in the prompt cache instead of validating against manifest captions.",
    )
    parser.add_argument(
        "--case_prompt_variant",
        default="prompt_physics_consistency_v1",
        help="Prompt cache key in a case-sidecar JSON; use base for the no-suffix cache.",
    )
    parser.add_argument(
        "--dataset_format",
        choices=("manifest", "pybullet0613_cached"),
        default=default_dataset_format,
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--context_frames", type=int, default=8)
    parser.add_argument("--context_latent_frames", type=int, default=2)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=20000)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--optimizer", choices=("adamw", "paged_adamw8bit"), default="paged_adamw8bit")
    parser.add_argument("--lr_scheduler", default="constant_with_warmup")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora_profile",
        choices=("full_sa", "rl"),
        default=default_lora_profile,
    )
    parser.add_argument("--expected_num_blocks", type=int, default=30)
    parser.add_argument("--expected_trainable_params", type=int, default=23592960)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--ddp_gradient_as_bucket_view",
        action="store_true",
        help="Reuse DDP communication buckets as gradient storage to reduce peak memory",
    )
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument(
        "--save_every_epoch",
        action="store_true",
        help="save one LoRA/scene checkpoint at the end of every completed epoch",
    )
    parser.add_argument(
        "--reset_accumulation_each_epoch",
        action="store_true",
        help="discard an incomplete accumulation at epoch boundaries instead of mixing epochs",
    )
    parser.add_argument("--checkpoints_total_limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--auxiliary_loss",
        choices=(
            "latent_mask",
            "latent_mask_full_frame",
            "latent_mask_utonia_scene",
            "vjepa",
            "xssc_object",
            "vjepa_vggt_scene",
            "vjepa_utonia_scene",
            "lpos",
            "raw",
            "raw_all_noise_future_loss",
            "raw_full_frame",
            "utonia_local_scene",
            "utonia_object_scene_attention",
            "utonia_subtraction",
        ),
        default=default_mode,
    )
    parser.add_argument(
        "--full_frame_flow",
        action="store_true",
        help=(
            "noise and supervise all 13 latent frames; currently used to run "
            "the object-conditioned scene branch under the raw full-frame control objective"
        ),
    )
    parser.add_argument(
        "--context_timestep_mode",
        choices=("match_noise", "force_zero"),
        default="match_noise",
        help=(
            "match_noise assigns the sampled timestep to noised context tokens "
            "and zero to clean context tokens; force_zero is an explicit "
            "mismatch control that marks context tokens as timestep zero even "
            "when they are noised"
        ),
    )
    parser.add_argument(
        "--context_noise_scale",
        type=float,
        default=None,
        help=(
            "Interpolate context noise against the sampled future sigma. "
            "Endpoints 0 and 1 hard-reproduce C and A; None preserves the legacy path."
        ),
    )
    parser.add_argument(
        "--enable_saved_tensor_cpu_offload",
        action="store_true",
        help=(
            "Explicitly enable saved-tensor CPU offload for any raw control "
            "mode; use this to keep topology-matched controls identical"
        ),
    )
    parser.add_argument(
        "--disable_saved_tensor_cpu_offload",
        action="store_true",
        help=(
            "Disable the legacy activation offload used by the single-GPU "
            "raw_all_noise_future_loss run; useful for topology-matched controls"
        ),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force_auxiliary_loss", action="store_true")
    parser.add_argument("--latent_mask_cache_root", default="/data/gaoya/agent-data/cache/pybullet0713_gt_latent_mask_v1")
    parser.add_argument("--latent_mask_head_config", default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/configs/physiciq67_pck32_s039_latest3350_top100_heads.json")
    parser.add_argument("--latent_mask_weight", type=float, default=0.01)
    parser.add_argument("--latent_mask_probe_timestep", type=float, default=500.0)
    parser.add_argument("--latent_mask_noise_level", type=float, default=0.5)
    parser.add_argument("--latent_mask_pck_weight_power", type=float, default=30.0)
    parser.add_argument("--latent_mask_query_frame", type=int, default=1)
    parser.add_argument("--latent_mask_object_index", type=int, default=0)
    parser.add_argument("--latent_mask_preflight_manifest", default=None)
    parser.add_argument(
        "--collision_cache_root",
        default=None,
        help=(
            "Optional compact PyBullet collision-supervision cache. When set, "
            "both future flow MSE and latent-mask CE use the per-latent impact weight."
        ),
    )
    parser.add_argument("--collision_weight_strength", type=float, default=2.0)
    parser.add_argument("--collision_weight_cap", type=float, default=3.0)
    parser.add_argument("--vjepa_loss_weight", type=float, default=0.01)
    parser.add_argument("--vjepa_sigma_min", type=float, default=0.2)
    parser.add_argument("--vjepa_sigma_max", type=float, default=0.8)
    parser.add_argument("--vjepa_every_n_forwards", type=int, default=2)
    parser.add_argument("--vjepa_num_frames", type=int, default=16)
    parser.add_argument("--vjepa_input_size", default="384")
    parser.add_argument("--vjepa_repo", default="/home/gaoya/Code_Video/vjepa2_tinyvae_mse/vjepa2")
    parser.add_argument("--vjepa_checkpoint", default="/data/gaoya/ckpt/VJEPA2/vjepa2_1_vitl_dist_vitG_384.pt")
    parser.add_argument("--tiny_vae_root", default="/home/gaoya/Code_Video/taehv")
    parser.add_argument("--tiny_vae_checkpoint", default="/data/gaoya/ckpt/taew2_2.pth")
    parser.add_argument("--vjepa_range_penalty_weight", type=float, default=0.1)
    parser.add_argument("--vjepa_frame_sampling", choices=("global", "local", "mixed", "full"), default="mixed")
    parser.add_argument("--vjepa_local_sampling_probability", type=float, default=0.5)
    parser.add_argument("--vjepa_local_context_frames", type=int, default=4)
    parser.add_argument("--lpos_loss_weight", type=float, default=0.01)
    # L_pos is evaluated on the native training-resolution decode.  This is
    # an auxiliary-loss implementation parameter; the main model/data
    # resolution remains controlled independently by --height/--width.
    parser.add_argument("--lpos_input_size", default="512x896")
    parser.add_argument(
        "--lpos_mask_cache_root",
        default="/data/gaoya/agent-data/cache/pybullet0713_gt_latent_mask_v1",
    )
    parser.add_argument("--lpos_context_frame", type=int, default=7)
    parser.add_argument("--lpos_temperature", type=float, default=0.02)
    parser.add_argument("--lpos_object_index", type=int, default=0)
    parser.add_argument("--xssc_root", default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/xssc_rsfq2_ytvis_dinov3_vitl16_256")
    parser.add_argument("--xssc_config", default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/xssc_rsfq2_ytvis_dinov3_vitl16_256/upstream/config-randsfq/rsfq2_c-movi_c-dinov3_vitl16_256-slot512-transfer15000.py")
    parser.add_argument("--xssc_checkpoint", default="/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/dinov3_xSSC/restart_save1000_20260720T140029Z/movi_c_transfer15000_b64_acc3_20260721T134713Z/rsfq2_c-movi_c-dinov3_vitl16_256-slot512-transfer15000/42/step-050000.pth")
    parser.add_argument("--dinov3_root", default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/xssc_rsfq2_ytvis_dinov3_vitl16_256/third_party/dinov3")
    parser.add_argument("--dinov3_checkpoint", default="/data/gaoya/ckpt/facebook-dinov3-vitl16-pretrain-lvd1689m/model.safetensors")
    parser.add_argument("--sam2_config", default="/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2_checkpoint", default="/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_large.pt")
    parser.add_argument("--xssc_box_cache_dir", default="/data/gaoya/agent-data/cache/physrvg_fastvideo_xssc_object_amg")
    parser.add_argument("--xssc_loss_weight", type=float, default=0.1)
    parser.add_argument("--xssc_future_start_frame", type=int, default=8)
    parser.add_argument("--xssc_chunk_size", type=int, default=2)
    parser.add_argument("--object_context_frames", type=int, default=8)
    parser.add_argument("--object_lora_rank", type=int, default=32)
    parser.add_argument("--object_lora_alpha", type=float, default=32.0)
    parser.add_argument("--object_lora_dropout", type=float, default=0.0)
    parser.add_argument("--object_gate_init", type=float, default=0.1)
    parser.add_argument("--slot_dedup_threshold", type=float, default=0.94)
    parser.add_argument("--slot_dedup_min_keep", type=int, default=3)
    parser.add_argument("--vggt_scene_cache_dir", default=None)
    parser.add_argument("--vggt_scene_grid_height", type=int, default=32)
    parser.add_argument("--vggt_scene_grid_width", type=int, default=56)
    parser.add_argument("--vggt_scene_patch_size", type=int, nargs=3, default=(4, 4, 4))
    parser.add_argument("--vggt_scene_embed_dim", type=int, default=256)
    parser.add_argument("--vggt_scene_lora_rank", type=int, default=32)
    parser.add_argument("--vggt_scene_lora_alpha", type=float, default=32.0)
    parser.add_argument("--vggt_scene_lora_dropout", type=float, default=0.0)
    parser.add_argument("--vggt_scene_gate_init", type=float, default=0.05)
    parser.add_argument("--utonia_scene_cache_dir", default=None)
    parser.add_argument("--utonia_scene_token_count", type=int, default=1792)
    parser.add_argument("--utonia_scene_feature_dim", type=int, default=1386)
    parser.add_argument(
        "--utonia_scene_prequery_token_limit",
        type=int,
        default=1792,
        help="maximum valid native Utonia tokens passed to the learnable-query projector",
    )
    parser.add_argument(
        "--utonia_scene_query_chunk_size",
        type=int,
        default=32,
        help="query rows processed per attention call to bound native-token attention memory",
    )
    parser.add_argument("--utonia_scene_lora_rank", type=int, default=32)
    parser.add_argument("--utonia_scene_lora_alpha", type=float, default=32.0)
    parser.add_argument("--utonia_scene_lora_dropout", type=float, default=0.0)
    parser.add_argument("--utonia_scene_gate_init", type=float, default=0.05)
    parser.add_argument("--utonia_local_cache_dir", default=None)
    parser.add_argument("--utonia_local_bridge_root", default=None)
    parser.add_argument("--utonia_local_preflight_manifest", default=None)
    parser.add_argument("--utonia_local_normalization_stats", default=None)
    parser.add_argument("--utonia_local_feature_dim", type=int, default=1386)
    parser.add_argument("--utonia_local_neighbors", type=int, default=64)
    parser.add_argument("--utonia_local_gate_hidden_dim", type=int, default=64)
    parser.add_argument("--utonia_local_scene_lora_rank", type=int, default=32)
    parser.add_argument("--utonia_local_scene_lora_alpha", type=float, default=32.0)
    parser.add_argument("--utonia_local_scene_lora_dropout", type=float, default=0.0)
    parser.add_argument("--utonia_local_scene_gate_init", type=float, default=0.0)
    parser.add_argument(
        "--expected_utonia_local_trainable_params",
        type=int,
        default=28106453,
    )
    parser.add_argument("--resume_utonia_local_checkpoint", default=None)
    parser.add_argument("--utonia_object_scene_cache_dir", default=None)
    parser.add_argument("--utonia_object_scene_oracle_bridge_dir", default=None)
    parser.add_argument("--utonia_object_motion_stats", default=None)
    parser.add_argument("--utonia_object_query_slots", type=int, default=1)
    parser.add_argument("--utonia_object_selector_dim", type=int, default=256)
    parser.add_argument("--utonia_object_distance_scale_m", type=float, default=0.25)
    parser.add_argument("--utonia_object_scene_lora_rank", type=int, default=32)
    parser.add_argument("--utonia_object_scene_lora_alpha", type=float, default=32.0)
    parser.add_argument("--utonia_object_scene_lora_dropout", type=float, default=0.0)
    parser.add_argument("--utonia_object_scene_gate_init", type=float, default=0.0)
    parser.add_argument(
        "--expected_utonia_object_scene_trainable_params",
        type=int,
        default=-1,
    )
    parser.add_argument("--resume_utonia_object_scene_checkpoint", default=None)
    parser.add_argument(
        "--utonia_subtraction_variant",
        choices=("motion", "geometry", "utonia"),
        default="motion",
    )
    parser.add_argument("--utonia_subtraction_hidden_dim", type=int, default=256)
    parser.add_argument("--utonia_subtraction_local_neighbors", type=int, default=64)
    parser.add_argument("--utonia_subtraction_distance_scale_m", type=float, default=0.25)
    parser.add_argument("--utonia_subtraction_block_indices", default="9,19,29")
    parser.add_argument("--utonia_subtraction_scene_lora_rank", type=int, default=8)
    parser.add_argument("--utonia_subtraction_scene_lora_alpha", type=float, default=8.0)
    parser.add_argument("--utonia_subtraction_scene_lora_dropout", type=float, default=0.0)
    parser.add_argument("--utonia_subtraction_scene_gate_init", type=float, default=0.0)
    parser.add_argument("--expected_utonia_subtraction_trainable_params", type=int, default=-1)
    parser.add_argument("--resume_utonia_subtraction_checkpoint", default=None)
    return parser


def _validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.resume_global_step < 0:
        parser.error("--resume_global_step must be non-negative")
    if args.resume_lora_checkpoint:
        resume_path = Path(args.resume_lora_checkpoint).expanduser()
        if not resume_path.is_dir():
            parser.error(f"--resume_lora_checkpoint does not exist: {resume_path}")
    if args.freeze_main_lora and not args.resume_lora_checkpoint:
        parser.error("--freeze_main_lora requires --resume_lora_checkpoint")
    if args.resume_global_step >= args.max_train_steps:
        parser.error("--resume_global_step must be smaller than --max_train_steps")
    for name in ("model_id", "pybullet_root", "vae_cache_dir", "prompt_cache_dir"):
        if not Path(getattr(args, name)).expanduser().exists():
            parser.error(f"{name} does not exist: {getattr(args, name)}")
    required_files = ["physrvg_dit_checkpoint"]
    if args.auxiliary_loss in VJEPA_MODES | LPOS_MODES:
        required_files.append("tiny_vae_checkpoint")
    if args.auxiliary_loss in VJEPA_MODES:
        required_files.append("vjepa_checkpoint")
    if args.auxiliary_loss == "xssc_object":
        required_files.extend(("xssc_config", "xssc_checkpoint", "dinov3_checkpoint", "sam2_config", "sam2_checkpoint"))
    for name in required_files:
        if not Path(getattr(args, name)).expanduser().is_file():
            parser.error(f"{name} does not exist: {getattr(args, name)}")
    if args.auxiliary_loss in LATENT_MASK_MODES and not Path(args.latent_mask_cache_root).expanduser().is_dir():
        parser.error(f"latent mask cache does not exist: {args.latent_mask_cache_root}")
    if args.auxiliary_loss in LPOS_MODES:
        if not Path(args.lpos_mask_cache_root).expanduser().is_dir():
            parser.error(f"L_pos mask cache does not exist: {args.lpos_mask_cache_root}")
        if args.lpos_loss_weight < 0.0:
            parser.error("--lpos_loss_weight must be non-negative")
        if args.lpos_temperature <= 0.0:
            parser.error("--lpos_temperature must be positive")
        _parse_vjepa_input_size(args.lpos_input_size)
    if args.collision_cache_root:
        collision_root = Path(args.collision_cache_root).expanduser()
        if not collision_root.is_dir():
            parser.error(f"collision cache does not exist: {args.collision_cache_root}")
        if args.collision_weight_strength < 0.0:
            parser.error("--collision_weight_strength must be non-negative")
        if args.collision_weight_cap < 1.0:
            parser.error("--collision_weight_cap must be at least 1")
    if args.context_timestep_mode == "force_zero" and args.auxiliary_loss not in {
        "raw_all_noise_future_loss",
        "raw_full_frame",
    }:
        parser.error(
            "--context_timestep_mode force_zero is restricted to the raw "
            "all-frame-noise timestep ablation"
        )
    if args.context_noise_scale is not None and not 0.0 <= args.context_noise_scale <= 1.0:
        parser.error("--context_noise_scale must be in [0, 1]")
    if args.context_noise_scale is not None and args.auxiliary_loss not in RAW_MODES:
        parser.error("--context_noise_scale is restricted to raw flow-matching modes")
    if args.enable_saved_tensor_cpu_offload and args.disable_saved_tensor_cpu_offload:
        parser.error(
            "--enable_saved_tensor_cpu_offload and "
            "--disable_saved_tensor_cpu_offload are mutually exclusive"
        )
    if args.auxiliary_loss == "vjepa_vggt_scene":
        if not args.vggt_scene_cache_dir:
            parser.error("--vggt_scene_cache_dir is required for vjepa_vggt_scene")
        if not Path(args.vggt_scene_cache_dir).expanduser().is_dir():
            parser.error(f"VGGT scene cache does not exist: {args.vggt_scene_cache_dir}")
        if tuple(args.vggt_scene_patch_size) != (4, 4, 4):
            parser.error("formal VGGT scene training requires --vggt_scene_patch_size 4 4 4")
        if args.vggt_scene_grid_height != args.height // 16:
            parser.error("VGGT scene grid height must equal the cached VAE latent height")
        if args.vggt_scene_grid_width != args.width // 16:
            parser.error("VGGT scene grid width must equal the cached VAE latent width")
        if args.context_frames % args.vggt_scene_patch_size[0]:
            parser.error("context frames must be divisible by the VGGT temporal patch size")
        if args.vggt_scene_grid_height % args.vggt_scene_patch_size[1]:
            parser.error("VGGT scene grid height must be divisible by its patch size")
        if args.vggt_scene_grid_width % args.vggt_scene_patch_size[2]:
            parser.error("VGGT scene grid width must be divisible by its patch size")
        if (
            args.vggt_scene_lora_rank != 32
            or args.vggt_scene_lora_alpha != 32.0
            or args.vggt_scene_lora_dropout != 0.0
        ):
            parser.error("formal VGGT scene cross-attention requires LoRA rank=32 alpha=32 dropout=0")
    if args.auxiliary_loss in UTONIA_SCENE_MODES:
        if not args.utonia_scene_cache_dir:
            parser.error(f"--utonia_scene_cache_dir is required for {args.auxiliary_loss}")
        if not Path(args.utonia_scene_cache_dir).expanduser().is_dir():
            parser.error(f"Utonia scene cache does not exist: {args.utonia_scene_cache_dir}")
        if args.utonia_scene_token_count <= 0 or args.utonia_scene_feature_dim != 1386:
            parser.error("formal Utonia scene training requires a positive token count and 1386 input features")
        if args.utonia_scene_prequery_token_limit <= 0:
            parser.error("--utonia_scene_prequery_token_limit must be positive")
        if args.utonia_scene_query_chunk_size <= 0:
            parser.error("--utonia_scene_query_chunk_size must be positive")
        if (
            args.utonia_scene_lora_rank != 32
            or args.utonia_scene_lora_alpha != 32.0
            or args.utonia_scene_lora_dropout != 0.0
        ):
            parser.error("formal Utonia scene cross-attention requires LoRA rank=32 alpha=32 dropout=0")
    if args.auxiliary_loss in UTONIA_LOCAL_SCENE_MODES:
        required_local_paths = (
            "utonia_local_cache_dir",
            "utonia_local_bridge_root",
            "utonia_local_preflight_manifest",
            "utonia_local_normalization_stats",
        )
        for name in required_local_paths:
            value = getattr(args, name)
            if not value or not Path(value).expanduser().exists():
                parser.error(
                    f"--{name} is required and must exist for "
                    f"utonia_local_scene: {value}"
                )
        if args.utonia_local_feature_dim != 1386:
            parser.error("local Utonia requires feature_dim=1386")
        if args.utonia_local_neighbors != 64:
            parser.error("the audited local Utonia design requires exactly 64 neighbors")
        if args.utonia_local_gate_hidden_dim != 64:
            parser.error("the first local Utonia experiment fixes gate hidden_dim=64")
        if (
            args.utonia_local_scene_lora_rank != 32
            or args.utonia_local_scene_lora_alpha != 32.0
            or args.utonia_local_scene_lora_dropout != 0.0
        ):
            parser.error("local Utonia scene attention requires LoRA rank=32 alpha=32 dropout=0")
        if args.utonia_local_scene_gate_init != 0.0:
            parser.error("local Utonia residual gate must be zero initialized")
        if args.expected_utonia_local_trainable_params != 28106453:
            parser.error(
                "the audited 30-block local Utonia branch requires "
                "28,106,453 trainable parameters"
            )
    if args.auxiliary_loss in UTONIA_OBJECT_SCENE_MODES:
        for name in (
            "utonia_object_scene_cache_dir",
            "utonia_object_scene_oracle_bridge_dir",
            "utonia_object_motion_stats",
        ):
            value = getattr(args, name)
            if not value or not Path(value).expanduser().exists():
                parser.error(f"--{name} is required for object-conditioned scene attention: {value}")
        if args.utonia_object_query_slots != 1:
            parser.error("the first object-conditioned scene experiment fixes one query per frame/object")
        if args.utonia_object_selector_dim != 256:
            parser.error("the first object-conditioned scene experiment fixes selector_dim=256")
        if args.utonia_object_distance_scale_m <= 0.0:
            parser.error("--utonia_object_distance_scale_m must be positive")
        if (
            args.utonia_object_scene_lora_rank != 32
            or args.utonia_object_scene_lora_alpha != 32.0
            or args.utonia_object_scene_lora_dropout != 0.0
        ):
            parser.error("object-conditioned scene attention requires LoRA rank=32 alpha=32 dropout=0")
        if args.utonia_object_scene_gate_init != 0.0:
            parser.error("object-conditioned scene residual gates must be zero initialized")
        if not args.smoke and args.expected_utonia_object_scene_trainable_params <= 0:
            parser.error("formal training requires an audited positive scene-trainable parameter count")
    if args.auxiliary_loss in UTONIA_SUBTRACTION_MODES:
        for name in (
            "utonia_object_scene_cache_dir",
            "utonia_object_scene_oracle_bridge_dir",
            "utonia_object_motion_stats",
        ):
            value = getattr(args, name)
            if not value or not Path(value).expanduser().exists():
                parser.error(f"--{name} is required for the subtraction experiment: {value}")
        if not args.freeze_main_lora:
            parser.error("the controlled subtraction experiment requires --freeze_main_lora")
        if args.utonia_subtraction_hidden_dim != 256:
            parser.error("the first subtraction comparison fixes hidden_dim=256")
        if args.utonia_subtraction_local_neighbors != 64:
            parser.error("the first subtraction comparison fixes 64 local scene rows")
        if args.utonia_subtraction_distance_scale_m <= 0.0:
            parser.error("--utonia_subtraction_distance_scale_m must be positive")
        try:
            block_indices = tuple(
                int(item.strip())
                for item in args.utonia_subtraction_block_indices.split(",")
                if item.strip()
            )
        except ValueError:
            parser.error("--utonia_subtraction_block_indices must be comma-separated integers")
        if not block_indices:
            parser.error("--utonia_subtraction_block_indices must not be empty")
        if tuple(sorted(set(block_indices))) != block_indices:
            parser.error(
                "--utonia_subtraction_block_indices must be sorted and contain no duplicates"
            )
        if block_indices[0] < 0 or block_indices[-1] >= args.expected_num_blocks:
            parser.error(
                "--utonia_subtraction_block_indices must be within "
                f"[0,{args.expected_num_blocks})"
            )
        if (
            args.utonia_subtraction_scene_lora_rank != 8
            or args.utonia_subtraction_scene_lora_alpha != 8.0
            or args.utonia_subtraction_scene_lora_dropout != 0.0
        ):
            parser.error("the subtraction branch requires scene LoRA rank=8 alpha=8 dropout=0")
        if args.utonia_subtraction_scene_gate_init != 0.0:
            parser.error("the subtraction residual gate must be zero initialized")
        if not args.smoke and args.expected_utonia_subtraction_trainable_params <= 0:
            parser.error("formal subtraction training requires an audited positive parameter count")
    if args.resume_vggt_scene_checkpoint:
        resume_scene = Path(args.resume_vggt_scene_checkpoint).expanduser()
        if not resume_scene.exists():
            parser.error(f"VGGT scene resume checkpoint does not exist: {resume_scene}")
    if args.resume_utonia_scene_checkpoint:
        resume_scene = Path(args.resume_utonia_scene_checkpoint).expanduser()
        if not resume_scene.exists():
            parser.error(f"Utonia scene resume checkpoint does not exist: {resume_scene}")
    if args.resume_utonia_local_checkpoint:
        resume_local = Path(args.resume_utonia_local_checkpoint).expanduser()
        if not resume_local.exists():
            parser.error(f"local Utonia resume checkpoint does not exist: {resume_local}")
    if args.resume_utonia_object_scene_checkpoint:
        resume_object_scene = Path(args.resume_utonia_object_scene_checkpoint).expanduser()
        if not resume_object_scene.exists():
            parser.error(
                "object-conditioned Utonia resume checkpoint does not exist: "
                f"{resume_object_scene}"
            )
    if args.resume_utonia_subtraction_checkpoint:
        resume_subtraction = Path(args.resume_utonia_subtraction_checkpoint).expanduser()
        if not resume_subtraction.exists():
            parser.error(
                "subtraction resume checkpoint does not exist: "
                f"{resume_subtraction}"
            )
    world_size_hint = int(os.environ.get("WORLD_SIZE", "1"))
    required_local_product = 8 // world_size_hint
    if args.train_batch_size * args.gradient_accumulation_steps != required_local_product:
        parser.error(
            f"effective batch 8 with WORLD_SIZE={world_size_hint} requires "
            f"train_batch_size * gradient_accumulation_steps = {required_local_product}"
        )
    allowed_batch_sizes = (
        {1, 2, 4, 8}
        if (
            args.auxiliary_loss == "raw_all_noise_future_loss"
            or (args.auxiliary_loss == "raw" and args.self_attention_mode == "context_only")
        )
        else {1, 2, 4}
    )
    if args.train_batch_size not in allowed_batch_sizes:
        parser.error(
            f"train_batch_size must be one of {sorted(allowed_batch_sizes)} "
            f"for auxiliary_loss={args.auxiliary_loss}"
        )
    if args.num_frames != 49 or args.context_latent_frames != 2:
        parser.error("this cached experiment requires 49 frames and 2 context latent frames")
    if args.smoke and args.max_train_steps > 10:
        parser.error("smoke runs must use at most 10 optimizer steps")
    if args.lora_profile == "rl":
        if args.lora_rank != 32 or args.lora_alpha != 64.0 or args.lora_dropout != 0.0:
            parser.error(
                "RL LoRA profile requires rank=32, alpha=64, and dropout=0"
            )


def run(
    default_mode: str,
    default_dataset_format: str = "manifest",
    default_lora_profile: str = "full_sa",
) -> None:
    parser = _build_parser(default_mode, default_dataset_format, default_lora_profile)
    args = parser.parse_args()
    _validate(args, parser)
    if args.self_attention_mode == "context_only" and args.auxiliary_loss != "raw":
        parser.error("context_only is restricted to the raw Flow control")
    from fastvideo.models.wan_v2v.context_only_attention import load_attention_policy
    if args.resume_lora_checkpoint:
        policy = load_attention_policy(Path(args.resume_lora_checkpoint))
        if policy["mode"] != args.self_attention_mode:
            parser.error("resume checkpoint attention policy differs from requested training mode")

    if args.auxiliary_loss != default_mode:
        parser.error(f"{Path(sys.argv[0]).name} is fixed to auxiliary_loss={default_mode}")
    if args.lora_profile != default_lora_profile:
        parser.error(
            f"{Path(sys.argv[0]).name} is fixed to lora_profile={default_lora_profile}"
        )

    from fastvideo.models.wan_v2v.model_wan_v2v import WanTransformer3DModel

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        process_group_timeout = (
            timedelta(minutes=30)
            if args.auxiliary_loss
            in UTONIA_OBJECT_SCENE_MODES
            | UTONIA_SUBTRACTION_MODES
            | {"latent_mask_full_frame"}
            else timedelta(minutes=10)
        )
        dist.init_process_group("nccl", timeout=process_group_timeout)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    run_dir = Path(args.output_dir).expanduser().resolve() / args.run_name
    checkpoint_root = run_dir / "checkpoints"
    if rank == 0:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    if args.dataset_format == "pybullet0613_cached":
        if args.caption_field or args.dataset_sources:
            raise ValueError("caption_field/dataset_sources requires dataset_format=manifest")
        from pybullet_0613_cached_dataset import PyBullet0613CachedDataset

        dataset = PyBullet0613CachedDataset(
            root=args.pybullet_root,
            vae_cache_dir=args.vae_cache_dir,
            prompt_cache_dir=args.prompt_cache_dir,
            split="train",
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
        )
    else:
        dataset = base.CachedPyBulletDataset(
            root=args.pybullet_root,
            vae_cache_dir=args.vae_cache_dir,
            prompt_cache_dir=args.prompt_cache_dir,
            prompt_cache_use_cache_text=args.prompt_cache_use_cache_text,
            sample_list=args.sample_list,
            case_prompt_variant=args.case_prompt_variant,
            caption_field=args.caption_field,
            dataset_sources=args.dataset_sources,
            split="train",
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
        )
    if args.auxiliary_loss == "vjepa_vggt_scene":
        from vggt_scene_cache import CachedVGGTSceneDataset

        dataset = CachedVGGTSceneDataset(
            dataset,
            args.vggt_scene_cache_dir,
            context_frames=args.context_frames,
            grid_height=args.vggt_scene_grid_height,
            grid_width=args.vggt_scene_grid_width,
            allow_partial=args.smoke,
        )
    elif args.auxiliary_loss in UTONIA_SCENE_MODES:
        scene_cache_module_name = os.environ.get(
            "PHYSRVG_UTONIA_SCENE_CACHE_MODULE",
            "utonia_scene_cache",
        )
        scene_cache_module = importlib.import_module(scene_cache_module_name)
        CachedUtoniaSceneDataset = scene_cache_module.CachedUtoniaSceneDataset

        dataset = CachedUtoniaSceneDataset(
            dataset,
            args.utonia_scene_cache_dir,
            token_count=args.utonia_scene_token_count,
            feature_dim=args.utonia_scene_feature_dim,
            allow_partial=args.smoke,
        )
    elif args.auxiliary_loss in UTONIA_LOCAL_SCENE_MODES:
        from utonia_local_motion_scene_dataset import CachedUtoniaLocalMotionSceneDataset

        dataset = CachedUtoniaLocalMotionSceneDataset(
            dataset,
            cache_dir=args.utonia_local_cache_dir,
            bridge_root=args.utonia_local_bridge_root,
            preflight_manifest=args.utonia_local_preflight_manifest,
            normalization_stats=args.utonia_local_normalization_stats,
            feature_dim=args.utonia_local_feature_dim,
            local_neighbors=args.utonia_local_neighbors,
            allow_partial=args.smoke,
        )
    elif args.auxiliary_loss in UTONIA_OBJECT_SCENE_MODES | UTONIA_SUBTRACTION_MODES:
        from utonia_object_scene_attention_dataset import (
            CachedUtoniaObjectSceneAttentionDataset,
        )

        dataset = CachedUtoniaObjectSceneAttentionDataset(
            dataset,
            cache_dir=args.utonia_object_scene_cache_dir,
            oracle_bridge_dir=args.utonia_object_scene_oracle_bridge_dir,
            normalization_stats=args.utonia_object_motion_stats,
            allow_partial=args.smoke,
        )

    direct_physrvg_gpu_load = (
        bool(args.physrvg_dit_checkpoint)
        and args.auxiliary_loss
        in UTONIA_OBJECT_SCENE_MODES
        | UTONIA_SUBTRACTION_MODES
        | {"latent_mask_full_frame"}
    )
    if args.physrvg_dit_checkpoint:
        # PhysRVG is a complete single-file state dict. Audited direct-load
        # modes build the exact architecture on meta and strict-load PhysRVG
        # onto the GPU, avoiding a temporary full FP32 CPU model.
        model_config_path = Path(args.model_id).expanduser().resolve() / "transformer" / "config.json"
        model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
        if direct_physrvg_gpu_load:
            # ConfigMixin ignores ``torch_dtype`` for this model class. A
            # normal from_config call therefore creates a full FP32 CPU model
            # before loading BF16 PhysRVG weights. Construct on meta instead;
            # the strict assign-load below materializes only the final BF16
            # parameters directly on this rank's GPU.
            with torch.device("meta"):
                transformer = WanTransformer3DModel.from_config(model_config)
        else:
            transformer = WanTransformer3DModel.from_config(
                model_config, torch_dtype=torch.bfloat16
            )
    else:
        transformer = WanTransformer3DModel.from_pretrained(args.model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
    if len(transformer.blocks) != args.expected_num_blocks:
        raise RuntimeError(f"Expected {args.expected_num_blocks} blocks, found {len(transformer.blocks)}")
    patch_size = tuple(map(int, transformer.config.patch_size))
    if args.self_attention_mode == "context_only":
        if patch_size[0] != 1:
            raise ValueError("context_only requires temporal patch size 1")
        transformer.context_only_latent_frames = args.context_latent_frames

    latent_mask_dataset_audit = None
    if args.auxiliary_loss in LATENT_MASK_MODES:
        latent_shape = tuple(map(int, dataset.expected_latent_shape))
        grid = tuple(
            latent_shape[axis] // patch_size[axis - 1]
            for axis in range(1, 4)
        )
        latent_mask_dataset_audit = _filter_latent_mask_records(
            dataset,
            Path(args.latent_mask_cache_root).expanduser().resolve(),
            grid=grid,
            query_frame=args.latent_mask_query_frame,
            object_index=args.latent_mask_object_index,
            rank=rank,
            device=device,
            manifest_path=(
                None
                if args.latent_mask_preflight_manifest is None
                else Path(args.latent_mask_preflight_manifest).expanduser().resolve()
            ),
        )
    collision_dataset_audit = None
    if args.collision_cache_root:
        collision_dataset_audit = _audit_collision_cache(
            dataset,
            Path(args.collision_cache_root).expanduser().resolve(),
        )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed, drop_last=True)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        collate_fn=dataset.collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if direct_physrvg_gpu_load:
        # Reading the same ~10 GB checkpoint concurrently from four ranks
        # caused severe storage contention in earlier runs. Every rank owns a
        # distinct GPU model, but only one rank reads/materializes it at a
        # time. The 30-minute process-group timeout covers this deterministic
        # startup phase without introducing a retry or fallback path.
        physrvg_dit = None
        for loader_rank in range(world_size):
            if rank == loader_rank:
                print(f"[rank{rank}] direct PhysRVG load -> {device}", flush=True)
                physrvg_dit = base.load_physrvg_dit(
                    transformer,
                    args.physrvg_dit_checkpoint,
                    device=device,
                    assign=True,
                )
                physrvg_dit["meta_load_buffers"] = (
                    base.materialize_wan_rope_after_meta_load(transformer, device)
                )
            if dist.is_initialized():
                dist.barrier()
        if physrvg_dit is None:
            raise RuntimeError(f"rank {rank} did not materialize its PhysRVG DiT")
    else:
        physrvg_dit = base.load_physrvg_dit(
            transformer,
            args.physrvg_dit_checkpoint,
        )
    if args.lora_profile == "rl":
        targets = list(RL_LORA_TARGET_MODULES)
        lora_kwargs = rl_lora_config_kwargs()
    else:
        targets = base.full_sa_target_modules(args.expected_num_blocks)
        lora_kwargs = {
            "r": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": targets,
            "init_lora_weights": True,
            "bias": "none",
        }
    if args.resume_lora_checkpoint:
        transformer = PeftModel.from_pretrained(transformer, args.resume_lora_checkpoint, is_trainable=True)
    else:
        transformer = get_peft_model(
            transformer,
            LoraConfig(**lora_kwargs),
        )
    if args.lora_profile == "rl":
        expected_params = (
            None if args.expected_trainable_params < 0 else args.expected_trainable_params
        )
        lora_stats = base.validate_rl_lora(
            transformer,
            expected_targets=targets,
            expected_trainable_params=expected_params,
        )
    else:
        lora_stats = base.validate_full_sa_lora(
            transformer,
            expected_targets=targets,
            expected_trainable_params=args.expected_trainable_params,
        )
    if args.freeze_main_lora:
        frozen_main_tensors = 0
        frozen_main_parameters = 0
        for name, parameter in transformer.named_parameters():
            if parameter.requires_grad:
                frozen_main_tensors += 1
                frozen_main_parameters += parameter.numel()
                parameter.requires_grad_(False)
        if frozen_main_parameters != int(lora_stats["trainable_params"]):
            raise RuntimeError(
                "main LoRA freeze mismatch: "
                f"found={frozen_main_parameters:,} expected={int(lora_stats['trainable_params']):,}"
            )
        lora_stats = {
            **lora_stats,
            "frozen_for_condition_ablation": True,
            "frozen_tensors": frozen_main_tensors,
            "frozen_params": frozen_main_parameters,
        }
    else:
        lora_stats = {**lora_stats, "frozen_for_condition_ablation": False}
    object_stats = None
    xssc_helpers = None
    vggt_scene_stats = None
    vggt_scene_helpers = None
    utonia_scene_stats = None
    utonia_scene_helpers = None
    utonia_local_stats = None
    utonia_local_helpers = None
    utonia_object_scene_stats = None
    utonia_object_scene_helpers = None
    utonia_subtraction_stats = None
    utonia_subtraction_helpers = None
    if args.auxiliary_loss == "xssc_object":
        import fastvideo_xssc_object as xssc_helpers
        xssc_helpers.inject_object_branch(transformer, args)
        object_stats = xssc_helpers.trainable_stats(transformer)
    elif args.auxiliary_loss == "vjepa_vggt_scene":
        import fastvideo_vggt_scene as vggt_scene_helpers

        vggt_scene_helpers.inject_scene_branch(transformer, args)
        if args.resume_vggt_scene_checkpoint:
            vggt_scene_helpers.load_trainable_state(
                transformer,
                args.resume_vggt_scene_checkpoint,
            )
        vggt_scene_stats = vggt_scene_helpers.trainable_stats(transformer)
    elif args.auxiliary_loss in UTONIA_SCENE_MODES:
        scene_helper_module_name = os.environ.get(
            "PHYSRVG_UTONIA_SCENE_HELPERS",
            "fastvideo_utonia_scene",
        )
        utonia_scene_helpers = importlib.import_module(scene_helper_module_name)

        utonia_scene_helpers.inject_scene_branch(transformer, args)
        if args.resume_utonia_scene_checkpoint:
            utonia_scene_helpers.load_trainable_state(
                transformer,
                args.resume_utonia_scene_checkpoint,
            )
        utonia_scene_stats = utonia_scene_helpers.trainable_stats(transformer)
    elif args.auxiliary_loss in UTONIA_LOCAL_SCENE_MODES:
        import fastvideo_utonia_local_motion_scene as utonia_local_helpers

        utonia_local_helpers.inject_scene_branch(transformer, args)
        if args.resume_utonia_local_checkpoint:
            utonia_local_helpers.load_trainable_state(
                transformer,
                args.resume_utonia_local_checkpoint,
            )
        utonia_local_stats = utonia_local_helpers.trainable_stats(transformer)
        local_trainable = int(utonia_local_stats["utonia_local_trainable_params"])
        if local_trainable != args.expected_utonia_local_trainable_params:
            raise RuntimeError(
                "local Utonia trainable parameter mismatch: "
                f"found={local_trainable:,} "
                f"expected={args.expected_utonia_local_trainable_params:,}"
            )
        total_trainable = sum(
            parameter.numel() for parameter in transformer.parameters() if parameter.requires_grad
        )
        expected_total = int(lora_stats["trainable_params"]) + local_trainable
        if total_trainable != expected_total:
            raise RuntimeError(
                "combined RL-LoRA/local Utonia trainable parameter mismatch: "
                f"found={total_trainable:,} expected={expected_total:,}"
            )
    elif args.auxiliary_loss in UTONIA_OBJECT_SCENE_MODES:
        import fastvideo_utonia_object_scene_attention as utonia_object_scene_helpers

        utonia_object_scene_helpers.inject_scene_branch(transformer, args)
        if args.resume_utonia_object_scene_checkpoint:
            utonia_object_scene_helpers.load_trainable_state(
                transformer,
                args.resume_utonia_object_scene_checkpoint,
            )
        utonia_object_scene_stats = utonia_object_scene_helpers.trainable_stats(transformer)
        scene_trainable = int(
            utonia_object_scene_stats["utonia_object_scene_trainable_params"]
        )
        if (
            args.expected_utonia_object_scene_trainable_params > 0
            and scene_trainable != args.expected_utonia_object_scene_trainable_params
        ):
            raise RuntimeError(
                "object-conditioned scene trainable parameter mismatch: "
                f"found={scene_trainable:,} "
                f"expected={args.expected_utonia_object_scene_trainable_params:,}"
            )
        total_trainable = sum(
            parameter.numel() for parameter in transformer.parameters() if parameter.requires_grad
        )
        expected_main_trainable = (
            0 if args.freeze_main_lora else int(lora_stats["trainable_params"])
        )
        expected_total = expected_main_trainable + scene_trainable
        if total_trainable != expected_total:
            raise RuntimeError(
                "combined RL-LoRA/object-scene trainable parameter mismatch: "
                f"found={total_trainable:,} expected={expected_total:,}"
            )
    elif args.auxiliary_loss in UTONIA_SUBTRACTION_MODES:
        import fastvideo_utonia_subtraction as utonia_subtraction_helpers

        utonia_subtraction_helpers.inject_scene_branch(transformer, args)
        if args.resume_utonia_subtraction_checkpoint:
            utonia_subtraction_helpers.load_trainable_state(
                transformer,
                args.resume_utonia_subtraction_checkpoint,
            )
        utonia_subtraction_stats = utonia_subtraction_helpers.trainable_stats(transformer)
        branch_trainable = int(
            utonia_subtraction_stats["utonia_subtraction_trainable_params"]
        )
        if (
            args.expected_utonia_subtraction_trainable_params > 0
            and branch_trainable != args.expected_utonia_subtraction_trainable_params
        ):
            raise RuntimeError(
                "subtraction trainable parameter mismatch: "
                f"found={branch_trainable:,} "
                f"expected={args.expected_utonia_subtraction_trainable_params:,}"
            )
        total_trainable = sum(
            parameter.numel() for parameter in transformer.parameters() if parameter.requires_grad
        )
        if total_trainable != branch_trainable:
            raise RuntimeError(
                "controlled subtraction must train only its branch: "
                f"total={total_trainable:,} branch={branch_trainable:,}"
            )
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    transformer.to(device).train()
    # The object-conditioned branch has one mandatory valid object and one
    # mandatory valid scene-token set per sample, and smoke gradients verify
    # that every trainable group participates on every forward.  Enabling DDP
    # unused-parameter discovery here only adds an extra autograd traversal.
    # Retain the historical setting for older scene modes whose optional
    # branches have not been audited under that stronger contract.
    ddp_find_unused = (
        args.auxiliary_loss in SCENE_CONDITION_MODES
        and args.auxiliary_loss not in UTONIA_OBJECT_SCENE_MODES | UTONIA_SUBTRACTION_MODES
    )
    model = (
        DDP(
            transformer,
            device_ids=[local_rank],
            find_unused_parameters=ddp_find_unused,
            gradient_as_bucket_view=args.ddp_gradient_as_bucket_view,
        )
        if world_size > 1
        else transformer
    )
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = base.build_optimizer(args, trainable_parameters)
    from diffusers.optimization import get_scheduler
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer, num_warmup_steps=args.lr_warmup_steps, num_training_steps=args.max_train_steps)
    flow_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    flow_scheduler.set_timesteps(args.num_train_timesteps, device=device)

    if args.auxiliary_loss in LATENT_MASK_MODES:
        selected_heads, head_weights, head_audit = _load_head_selection(
            args.latent_mask_head_config,
            args.latent_mask_pck_weight_power,
        )
    else:
        selected_heads, head_weights, head_audit = {}, torch.empty(0), None
    probe = None
    vjepa = None
    lpos = None
    xssc_system = None
    if args.auxiliary_loss in LATENT_MASK_MODES:
        if args.auxiliary_loss == "latent_mask_full_frame" and dist.is_initialized():
            for loader_rank in range(world_size):
                if rank == loader_rank:
                    print(f"[rank{rank}] frozen probe load -> {device}", flush=True)
                    probe = FrozenProbe(args, device, torch.bfloat16, selected_heads)
                dist.barrier()
            if probe is None:
                raise RuntimeError(f"rank {rank} did not load its frozen probe")
        else:
            probe = FrozenProbe(args, device, torch.bfloat16, selected_heads)
    elif args.auxiliary_loss in VJEPA_MODES:
        vjepa = VJEPAAuxiliary(args, device, torch.bfloat16)
    elif args.auxiliary_loss in LPOS_MODES:
        lpos = LPosAuxiliary(args, device, torch.bfloat16)
    elif args.auxiliary_loss == "xssc_object":
        xssc_system = xssc_helpers.FastVideoXSSCSystem(args, device, torch.bfloat16)

    effective_batch = args.train_batch_size * world_size * args.gradient_accumulation_steps
    if effective_batch != 8:
        raise RuntimeError(f"effective batch must be 8, got {effective_batch}")
    # The subtraction campaign is an all-frame denoising experiment. Reuse the
    # same established path as raw_full_frame: all 13 latent frames are noised
    # and contribute to flow-matching MSE. Context token timesteps normally
    # follow whether those tokens were actually noised.
    full_frame_flow = (
        args.full_frame_flow
        or args.auxiliary_loss in {"raw_full_frame", "latent_mask_full_frame"}
        or args.auxiliary_loss in UTONIA_SUBTRACTION_MODES
    )
    all_frame_noise = (
        full_frame_flow
        or args.auxiliary_loss == "raw_all_noise_future_loss"
    )
    saved_tensor_cpu_offload = (
        args.enable_saved_tensor_cpu_offload
        or (
            args.auxiliary_loss == "raw_all_noise_future_loss"
            and not args.disable_saved_tensor_cpu_offload
        )
    )
    if args.context_noise_scale is None or args.context_noise_scale == 0.0:
        configured_noised_range = "future_latents_2_through_12"
        configured_context_tokens = "zero_condition_timestep"
    elif args.context_noise_scale == 1.0:
        configured_noised_range = "all_13_latents"
        configured_context_tokens = "sampled_noise_timestep"
    else:
        configured_noised_range = "all_13_latents_with_context_sigma_interpolation"
        configured_context_tokens = "scheduler_nearest_paired_context_timestep"
    if rank == 0:
        config = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "arguments": vars(args),
            "dataset": dataset.stats,
            "physrvg_dit": physrvg_dit,
            "full_sa": lora_stats,
            "object_xssc": object_stats,
            "vggt_scene": vggt_scene_stats,
            "utonia_scene": utonia_scene_stats,
            "utonia_local_scene": utonia_local_stats,
            "utonia_object_scene_attention": utonia_object_scene_stats,
            "utonia_subtraction": utonia_subtraction_stats,
            "effective_batch_size": effective_batch,
            "world_size": world_size,
            "auxiliary_loss": args.auxiliary_loss,
            "flow_training": {
                "objective": (
                    "full_frame_flow_matching_loss"
                    if full_frame_flow
                    else (
                        "all_frame_noise_future_frame_flow_matching_loss"
                        if all_frame_noise
                        else "future_frame_flow_matching_loss"
                    )
                ),
                "noised_latent_range": (
                    configured_noised_range
                ),
                "loss_latent_range": (
                    "all_13_latents"
                    if full_frame_flow
                    else "future_latents_2_through_12"
                ),
                "context_timestep_mode": args.context_timestep_mode,
                "context_noise_scale": args.context_noise_scale,
                "context_noise_mapping": (
                    "endpoint_exact_and_nearest_paired_training_sigma"
                    if args.context_noise_scale is not None
                    else "legacy"
                ),
                "context_token_timestep": (
                    configured_context_tokens
                ),
                "saved_tensor_cpu_offload": saved_tensor_cpu_offload,
            },
            "latent_mask_heads": head_audit,
            "latent_mask_dataset": latent_mask_dataset_audit,
            "lpos": {
                "enabled": args.auxiliary_loss in LPOS_MODES,
                "loss_weight": float(args.lpos_loss_weight),
                "input_size": str(args.lpos_input_size),
                "mask_cache_root": str(Path(args.lpos_mask_cache_root).expanduser().resolve()),
                "context_frame": int(args.lpos_context_frame),
                "temperature": float(args.lpos_temperature),
                "object_index": int(args.lpos_object_index),
                "implementation": "differentiable_soft_template_centroid",
            },
            "raw_control": {
                "enabled": args.auxiliary_loss in RAW_MODES,
                "objective": (
                    "full_frame_flow_matching_loss_only"
                    if full_frame_flow
                    else (
                        "all_frame_noise_future_frame_flow_matching_loss_only"
                        if all_frame_noise
                        else "flow_matching_loss_only"
                    )
                ),
                "noised_latent_range": (
                    configured_noised_range
                ),
                "loss_latent_range": (
                    "all_13_latents"
                    if full_frame_flow
                    else "future_latents_2_through_12"
                ),
                "context_timestep_mode": args.context_timestep_mode,
                "context_noise_scale": args.context_noise_scale,
                "context_noise_mapping": (
                    "endpoint_exact_and_nearest_paired_training_sigma"
                    if args.context_noise_scale is not None
                    else "legacy"
                ),
                "context_token_timestep": (
                    configured_context_tokens
                ),
            },
            "collision_dataset": collision_dataset_audit,
            "wandb": {
                "mode": os.environ.get("WANDB_MODE", "disabled"),
                "project": os.environ.get("WANDB_PROJECT", "physrvg-full-sa"),
                "name": os.environ.get("WANDB_NAME", args.run_name),
            },
        }
        (run_dir / "resolved_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(config, indent=2, ensure_ascii=True, sort_keys=True), flush=True)

    optimizer.zero_grad(set_to_none=True)
    optimizer_steps_per_epoch = max(
        1, len(dataloader) // args.gradient_accumulation_steps
    )
    resume_epoch = args.resume_global_step // optimizer_steps_per_epoch
    if resume_epoch >= args.num_epochs:
        raise RuntimeError(
            "resume checkpoint is beyond the configured epoch budget: "
            f"resume_step={args.resume_global_step}, "
            f"resume_epoch={resume_epoch}, num_epochs={args.num_epochs}"
        )
    progress = tqdm(
        total=args.max_train_steps,
        initial=args.resume_global_step,
        desc=f"{args.auxiliary_loss} optimizer steps",
        disable=rank != 0,
    )
    metrics_path = run_dir / "train_metrics.jsonl"
    global_step = args.resume_global_step
    micro_step = 0
    context_noise_audit_count = 0
    accumulated: dict[str, torch.Tensor] = {}
    started = time.monotonic()
    fixed_index = None
    if args.force_auxiliary_loss:
        target_sigma = 0.5
        fixed_index = int(torch.argmin((flow_scheduler.sigmas.float() - target_sigma).abs()).item())

    completed_normally = False
    wandb_run = None
    interrupt_requested = False
    interrupt_signal = None

    def save_checkpoint(checkpoint_dir: Path) -> None:
        """Persist the LoRA plus whichever auxiliary branch is trainable."""
        base.save_lora_checkpoint(model, checkpoint_dir)
        if args.self_attention_mode == "context_only":
            (checkpoint_dir / "self_attention_policy.json").write_text(
                json.dumps({"mode": "context_only", "context_latent_frames": args.context_latent_frames}, indent=2) + "\n",
                encoding="utf-8",
            )

        if args.auxiliary_loss == "xssc_object":
            xssc_helpers.save_trainable_state(model, checkpoint_dir)
        elif args.auxiliary_loss == "vjepa_vggt_scene":
            vggt_scene_helpers.save_trainable_state(model, checkpoint_dir)
        elif args.auxiliary_loss in UTONIA_SCENE_MODES:
            utonia_scene_helpers.save_trainable_state(model, checkpoint_dir)
        elif args.auxiliary_loss in UTONIA_LOCAL_SCENE_MODES:
            utonia_local_helpers.save_trainable_state(model, checkpoint_dir)
        elif args.auxiliary_loss in UTONIA_OBJECT_SCENE_MODES:
            utonia_object_scene_helpers.save_trainable_state(model, checkpoint_dir)
        elif args.auxiliary_loss in UTONIA_SUBTRACTION_MODES:
            utonia_subtraction_helpers.save_trainable_state(model, checkpoint_dir)

    def request_interrupt(signum, _frame) -> None:
        """Defer checkpoint I/O until the current optimizer operation is safe."""
        nonlocal interrupt_requested, interrupt_signal
        if not interrupt_requested:
            interrupt_requested = True
            interrupt_signal = int(signum)
            print(
                f"[interrupt] signal={signum}; saving after current optimizer step",
                flush=True,
            )

    signal.signal(signal.SIGINT, request_interrupt)
    signal.signal(signal.SIGTERM, request_interrupt)

    try:
        if rank == 0:
            wandb_run = _init_wandb(run_dir, config)
        for epoch in range(resume_epoch, args.num_epochs):
            sampler.set_epoch(epoch)
            for batch in dataloader:
                should_sync = (micro_step + 1) % args.gradient_accumulation_steps == 0
                sync_context = nullcontext() if should_sync or world_size == 1 else model.no_sync()
                latents = batch["latents"].to(device=device, dtype=torch.bfloat16, non_blocking=True)
                prompt_embeds = batch["prompt_embeds"].to(device=device, dtype=torch.bfloat16, non_blocking=True)
                scene_world_points = None
                scene_confidence = None
                scene_features = None
                scene_feature_mask = None
                if args.auxiliary_loss == "vjepa_vggt_scene":
                    scene_world_points = batch["scene_world_points"].to(
                        device=device,
                        dtype=torch.bfloat16,
                        non_blocking=True,
                    )
                    scene_confidence = batch["scene_confidence"].to(
                        device=device,
                        dtype=torch.bfloat16,
                        non_blocking=True,
                    )
                elif args.auxiliary_loss in UTONIA_SCENE_MODES:
                    scene_features = batch["utonia_scene_features"].to(
                        device=device,
                        dtype=torch.bfloat16,
                        non_blocking=True,
                    )
                    scene_feature_mask = batch["utonia_scene_mask"].to(
                        device=device,
                        dtype=torch.float32,
                        non_blocking=True,
                    )
                elif args.auxiliary_loss in UTONIA_LOCAL_SCENE_MODES:
                    scene_features, scene_feature_mask = _prepare_utonia_local_scene_inputs(
                        batch,
                        device,
                    )
                elif args.auxiliary_loss in UTONIA_OBJECT_SCENE_MODES:
                    scene_features = _prepare_utonia_object_scene_inputs(batch, device)
                elif args.auxiliary_loss in UTONIA_SUBTRACTION_MODES:
                    scene_features = _prepare_utonia_subtraction_inputs(
                        batch,
                        device,
                        args.utonia_subtraction_variant,
                    )
                save_on_cpu_pin_memory = os.environ.get(
                    "PHYSRVG_SAVE_ON_CPU_PIN_MEMORY", "1"
                ).strip().lower() not in {"0", "false", "no", "off"}
                use_saved_tensor_cpu_offload = (
                    args.auxiliary_loss
                    in LATENT_MASK_MODES
                    | VJEPA_MODES
                    | LPOS_MODES
                    | {"xssc_object"}
                    or saved_tensor_cpu_offload
                )
                offload_context = (
                    torch.autograd.graph.save_on_cpu(
                        pin_memory=save_on_cpu_pin_memory
                    )
                    if use_saved_tensor_cpu_offload
                    else nullcontext()
                )
                with sync_context, offload_context:
                    xssc_bundle = None
                    if args.auxiliary_loss == "xssc_object":
                        xssc_bundle = xssc_system.prepare_target(latents)
                    flow_loss, state = _flow_forward(
                        model,
                        flow_scheduler,
                        latents,
                        prompt_embeds,
                        context_latent_frames=args.context_latent_frames,
                        patch_size=patch_size,
                        fixed_index=fixed_index,
                        object_slots=None if xssc_bundle is None else xssc_bundle["object_slots"],
                        object_keep_mask=None if xssc_bundle is None else xssc_bundle["object_keep"],
                        scene_world_points=scene_world_points,
                        scene_confidence=scene_confidence,
                        scene_features=scene_features,
                        scene_feature_mask=scene_feature_mask,
                        logical_keys=batch["logical_keys"],
                        collision_cache_root=(
                            None
                            if args.collision_cache_root is None
                            else Path(args.collision_cache_root).expanduser().resolve()
                        ),
                        collision_weight_strength=args.collision_weight_strength,
                        collision_weight_cap=args.collision_weight_cap,
                        all_frame_noise=all_frame_noise,
                        full_frame_flow=full_frame_flow,
                        context_timestep_mode=args.context_timestep_mode,
                        context_noise_scale=args.context_noise_scale,
                    )
                    if (
                        rank == 0
                        and args.context_noise_scale is not None
                        and context_noise_audit_count < 4
                    ):
                        context_noise_audit_count += 1
                        future_sigma = float(state["future_sigma"].detach().item())
                        context_sigma = float(state["context_sigma"].detach().item())
                        ratio = context_sigma / future_sigma if future_sigma > 0 else 0.0
                        print(
                            json.dumps(
                                {
                                    "context_noise_audit": context_noise_audit_count,
                                    "future_timestep": float(state["future_timestep"].detach().item()),
                                    "future_sigma": future_sigma,
                                    "lambda": float(args.context_noise_scale),
                                    "target_context_sigma": float(state["context_sigma_target"].detach().item()),
                                    "context_timestep": float(state["context_timestep"].detach().item()),
                                    "context_sigma": context_sigma,
                                    "context_sigma_over_future_sigma": ratio,
                                    "context_token_timestep": float(state["context_timestep"].detach().item()),
                                    "future_token_timestep": float(state["future_timestep"].detach().item()),
                                    "context_latent_noised": bool(context_sigma > 0.0),
                                    "noise_tensor_reused": bool(state["context_noise_reused"].item() == 1.0),
                                    "loss_latent_range": "future_latents_2_through_12",
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                    if args.auxiliary_loss in LATENT_MASK_MODES:
                        aux_loss, aux_metrics = _latent_mask_loss(
                            args,
                            probe,
                            state,
                            batch["logical_keys"],
                            head_weights,
                            Path(args.latent_mask_cache_root).expanduser().resolve(),
                            selected_heads,
                            patch_size,
                        )
                    elif args.auxiliary_loss in VJEPA_MODES:
                        aux_loss, aux_metrics = vjepa.compute(
                            _pred_x0(state, args.context_latent_frames),
                            latents.detach(),
                            state["sigma"],
                            force=args.force_auxiliary_loss,
                        )
                        if scene_confidence is not None:
                            aux_metrics["vggt_scene_confidence_mean"] = float(
                                scene_confidence.float().mean().detach().item()
                            )
                        if scene_feature_mask is not None:
                            aux_metrics["utonia_scene_mask_mean"] = float(
                                scene_feature_mask.float().mean().detach().item()
                            )
                    elif args.auxiliary_loss in LPOS_MODES:
                        aux_loss, aux_metrics = lpos.compute(
                            _pred_x0(state, args.context_latent_frames),
                            latents.detach(),
                            batch["logical_keys"],
                            context_frames=args.context_frames,
                        )
                    elif args.auxiliary_loss in RAW_MODES:
                        aux_loss = flow_loss.detach() * 0.0
                        aux_metrics = {
                            "raw_auxiliary_loss": 0.0,
                        }
                        if args.auxiliary_loss in UTONIA_LOCAL_SCENE_MODES:
                            aux_metrics.update(
                                {
                                    "utonia_local_valid_fraction": float(
                                        scene_feature_mask.float().mean().detach().item()
                                    ),
                                    "utonia_local_token_count": float(scene_features.shape[1]),
                                    "utonia_local_object_count_mean": float(
                                        batch["utonia_local_object_counts"].float().mean().item()
                                    ),
                                }
                            )
                        elif args.auxiliary_loss in UTONIA_OBJECT_SCENE_MODES:
                            aux_metrics.update(
                                {
                                    "utonia_object_scene_token_count_mean": float(
                                        batch["utonia_object_scene_counts"].float().mean().item()
                                    ),
                                    "utonia_object_count_mean": float(
                                        batch["utonia_object_counts"].float().mean().item()
                                    ),
                                    "utonia_object_valid_fraction": float(
                                        batch["utonia_object_valid"].float().mean().item()
                                    ),
                                }
                            )
                        elif args.auxiliary_loss in UTONIA_SUBTRACTION_MODES:
                            aux_metrics.update(
                                {
                                    "utonia_subtraction_variant_id": float(
                                        {"motion": 0, "geometry": 1, "utonia": 2}[
                                            args.utonia_subtraction_variant
                                        ]
                                    ),
                                    "utonia_object_count_mean": float(
                                        batch["utonia_object_counts"].float().mean().item()
                                    ),
                                    "utonia_object_valid_fraction": float(
                                        batch["utonia_object_valid"].float().mean().item()
                                    ),
                                }
                            )
                    else:
                        aux_loss, aux_metrics = xssc_system.loss(
                            _pred_x0(state, args.context_latent_frames),
                            latents.detach(),
                            xssc_bundle,
                        )
                    loss = flow_loss + aux_loss
                    (loss / args.gradient_accumulation_steps).backward()
                accumulated.setdefault("loss", torch.zeros((), device=device))
                accumulated["loss"] += loss.detach().float()
                for key, value in aux_metrics.items():
                    accumulated.setdefault(key, torch.zeros((), device=device))
                    accumulated[key] += torch.as_tensor(value, device=device, dtype=torch.float32)
                micro_step += 1
                if not should_sync:
                    continue
                if args.auxiliary_loss in UTONIA_SCENE_MODES:
                    utonia_gradient_metrics = _utonia_gradient_diagnostics(model)
                elif args.auxiliary_loss in UTONIA_LOCAL_SCENE_MODES:
                    utonia_gradient_metrics = utonia_local_helpers.gradient_diagnostics(model)
                elif args.auxiliary_loss in UTONIA_OBJECT_SCENE_MODES:
                    utonia_gradient_metrics = (
                        utonia_object_scene_helpers.gradient_diagnostics(model)
                    )
                    if args.smoke:
                        utonia_object_scene_helpers.assert_smoke_gradient_contract(
                            global_step + 1,
                            utonia_gradient_metrics,
                        )
                elif args.auxiliary_loss in UTONIA_SUBTRACTION_MODES:
                    utonia_gradient_metrics = utonia_subtraction_helpers.gradient_diagnostics(
                        model
                    )
                    if args.smoke:
                        utonia_subtraction_helpers.assert_smoke_gradient_contract(
                            global_step + 1,
                            utonia_gradient_metrics,
                        )
                else:
                    utonia_gradient_metrics = {}
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                metric_tensors = {}
                for key, value in accumulated.items():
                    reduced = value.detach().clone()
                    if dist.is_initialized():
                        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                        reduced /= dist.get_world_size()
                    metric_tensors[key] = reduced / args.gradient_accumulation_steps
                accumulated.clear()
                if rank == 0:
                    metric = {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": float(metric_tensors["loss"].item()),
                        "loss_flow": float((metric_tensors["loss"] - metric_tensors.get("loss_latent_mask", metric_tensors.get("vjepa_weighted_contribution", metric_tensors.get("lpos_weighted_contribution", metric_tensors.get("xssc_weighted_contribution", torch.zeros((), device=device)))))).item()),
                        "grad_norm": float(grad_norm),
                        "learning_rate": float(lr_scheduler.get_last_lr()[0]),
                        "elapsed_seconds": time.monotonic() - started,
                        **utonia_gradient_metrics,
                        **{key: float(value.item()) for key, value in metric_tensors.items() if key != "loss"},
                    }
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(metric, sort_keys=True) + "\n")
                    if wandb_run is not None:
                        try:
                            wandb_run.log(metric, step=global_step, commit=True)
                        except Exception as exc:
                            print(f"[wandb] log failed at step {global_step}: {exc!r}", flush=True)
                    progress.update(1)
                    progress.set_postfix(loss=f"{metric['loss']:.5f}", aux=f"{float(metric_tensors.get('loss_latent_mask', metric_tensors.get('vjepa_weighted_contribution', metric_tensors.get('lpos_weighted_contribution', metric_tensors.get('xssc_weighted_contribution', torch.zeros((), device=device)))).item())):.5f}")
                if global_step % args.checkpointing_steps == 0:
                    if rank == 0:
                        save_checkpoint(checkpoint_root / f"step-{global_step:06d}")
                        base.prune_checkpoints(checkpoint_root, args.checkpoints_total_limit)
                if interrupt_requested:
                    if rank == 0:
                        interrupt_dir = checkpoint_root / f"interrupt-step-{global_step:06d}"
                        if interrupt_dir.exists():
                            interrupt_dir = checkpoint_root / f"interrupt-step-{global_step:06d}-{int(time.time())}"
                        save_checkpoint(interrupt_dir)
                        (interrupt_dir / "interrupt.json").write_text(
                            json.dumps(
                                {
                                    "signal": interrupt_signal,
                                    "global_step": global_step,
                                    "resume_global_step": global_step,
                                    "checkpoint_type": "interrupt",
                                },
                                indent=2,
                                sort_keys=True,
                            )
                            + "\n",
                            encoding="utf-8",
                        )
                        print(f"[interrupt] checkpoint_saved={interrupt_dir}", flush=True)
                    if dist.is_initialized():
                        dist.barrier()
                    break
                    if dist.is_initialized():
                        dist.barrier()
                if global_step >= args.max_train_steps:
                    break
            if args.save_every_epoch and global_step > 0 and rank == 0:
                # ``checkpointing_steps`` may be deliberately large in an
                # epoch-budgeted run; this is the authoritative epoch save.
                epoch_checkpoint = checkpoint_root / f"step-{global_step:06d}"
                if not epoch_checkpoint.exists():
                    save_checkpoint(epoch_checkpoint)
                    base.prune_checkpoints(checkpoint_root, args.checkpoints_total_limit)
            if interrupt_requested:
                break
            if args.save_every_epoch and dist.is_initialized():
                dist.barrier()
            if args.reset_accumulation_each_epoch:
                # The 60-example train split yields 30 examples per rank.
                # Two trailing micro-batches do not form a valid effective
                # batch of 8, so discard them and start the next epoch cleanly.
                if micro_step % args.gradient_accumulation_steps:
                    optimizer.zero_grad(set_to_none=True)
                    accumulated.clear()
                micro_step = 0
            if global_step >= args.max_train_steps:
                break
        completed_normally = True
    except Exception:
        print(f"[rank{rank}] training exception", flush=True)
        traceback.print_exc()
        raise
    finally:
        if rank == 0 and global_step >= args.max_train_steps:
            save_checkpoint(run_dir / "final_lora")
            progress.close()
        if rank == 0 and wandb_run is not None:
            try:
                wandb_run.finish(exit_code=0 if completed_normally else 1)
            except Exception as exc:
                print(f"[wandb] finish failed: {exc!r}", flush=True)
        if dist.is_initialized():
            if completed_normally:
                dist.barrier()
            dist.destroy_process_group()


def main(
    default_mode: str,
    default_dataset_format: str = "manifest",
    default_lora_profile: str = "full_sa",
) -> None:
    run(default_mode, default_dataset_format, default_lora_profile)
