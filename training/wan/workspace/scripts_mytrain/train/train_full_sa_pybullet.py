from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from pybullet_cached_dataset import CachedPyBulletDataset
except ModuleNotFoundError:
    from scripts_mytrain.data.pybullet_cached_dataset import CachedPyBulletDataset


SELF_ATTN_PROJECTIONS = ("to_q", "to_k", "to_v", "to_out.0")


def _init_wandb(run_dir: Path, config: dict) -> object | None:
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


def full_sa_target_modules(num_blocks: int) -> list[str]:
    return [
        f"blocks.{block_id}.attn1.{projection}"
        for block_id in range(num_blocks)
        for projection in SELF_ATTN_PROJECTIONS
    ]


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def validate_full_sa_lora(
    model,
    *,
    expected_targets: list[str],
    expected_trainable_params: int,
) -> dict:
    lora_modules = [
        name
        for name, module in model.named_modules()
        if hasattr(module, "lora_A") and hasattr(module, "lora_B")
    ]
    missing_targets = [
        target
        for target in expected_targets
        if not any(name.endswith(target) for name in lora_modules)
    ]
    unexpected_modules = [
        name
        for name in lora_modules
        if not any(name.endswith(target) for target in expected_targets)
    ]
    if missing_targets or unexpected_modules or len(lora_modules) != len(expected_targets):
        raise RuntimeError(
            "Full-SA LoRA module mismatch: "
            f"found={len(lora_modules)} expected={len(expected_targets)} "
            f"missing={missing_targets[:5]} unexpected={unexpected_modules[:5]}"
        )

    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    invalid_trainables = [
        name
        for name, _ in trainable
        if ".attn1." not in name or (".lora_A." not in name and ".lora_B." not in name)
    ]
    if invalid_trainables:
        raise RuntimeError(f"Non-Full-SA parameters are trainable: {invalid_trainables[:8]}")
    trainable_params = sum(parameter.numel() for _, parameter in trainable)
    if trainable_params != expected_trainable_params:
        raise RuntimeError(
            "Full-SA trainable parameter mismatch: "
            f"found={trainable_params:,} expected={expected_trainable_params:,}"
        )
    return {
        "lora_modules": len(lora_modules),
        "trainable_tensors": len(trainable),
        "trainable_params": trainable_params,
    }


def validate_rl_lora(
    model,
    *,
    expected_targets: list[str],
    expected_trainable_params: int | None = None,
) -> dict:
    """Validate the suffix-based LoRA profile used by the RL trainer.

    Unlike the historical Full-SA profile, the RL target list intentionally
    matches every occurrence of each suffix (including attn1, attn2, and FFN
    modules). Therefore the number of injected modules is architecture
    dependent and must not be compared with ``len(expected_targets)``.
    """
    lora_modules = [
        name
        for name, module in model.named_modules()
        if hasattr(module, "lora_A") and hasattr(module, "lora_B")
    ]
    missing_targets = [
        target
        for target in expected_targets
        if not any(name.endswith(target) for name in lora_modules)
    ]
    unexpected_modules = [
        name
        for name in lora_modules
        if not any(name.endswith(target) for target in expected_targets)
    ]
    if missing_targets or unexpected_modules:
        raise RuntimeError(
            "RL LoRA module mismatch: "
            f"found={len(lora_modules)} missing={missing_targets[:5]} "
            f"unexpected={unexpected_modules[:5]}"
        )

    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    invalid_trainables = [
        name
        for name, _ in trainable
        if ".lora_A." not in name and ".lora_B." not in name
    ]
    if invalid_trainables:
        raise RuntimeError(
            f"Non-RL-LoRA parameters are trainable: {invalid_trainables[:8]}"
        )
    trainable_params = sum(parameter.numel() for _, parameter in trainable)
    if (
        expected_trainable_params is not None
        and expected_trainable_params >= 0
        and trainable_params != expected_trainable_params
    ):
        raise RuntimeError(
            "RL LoRA trainable parameter mismatch: "
            f"found={trainable_params:,} expected={expected_trainable_params:,}"
        )
    return {
        "profile": "rl",
        "target_suffixes": list(expected_targets),
        "lora_modules": len(lora_modules),
        "lora_module_names": lora_modules,
        "trainable_tensors": len(trainable),
        "trainable_params": trainable_params,
        "expected_trainable_params": expected_trainable_params,
    }


def load_physrvg_dit(
    transformer,
    checkpoint: str,
    *,
    device: str | torch.device = "cpu",
    assign: bool = False,
) -> dict:
    """Strict-load the complete PhysRVG DiT state dict.

    ``assign=True`` is intended for a model constructed on the meta device.
    In that mode safetensors materializes each tensor directly on ``device``
    and ``load_state_dict`` adopts those tensors without first allocating a
    second full CPU model. Existing callers retain the historical CPU-copy
    behavior through the defaults above.
    """

    target_device = str(device)
    state_dict = load_file(checkpoint, device=target_device)
    transformer.load_state_dict(state_dict, strict=True, assign=assign)
    parameter = next(transformer.parameters())
    return {
        "checkpoint": checkpoint,
        "strict": True,
        "assign": bool(assign),
        "load_device": target_device,
        "parameter_device": str(parameter.device),
        "parameter_dtype": str(parameter.dtype),
        "source_tensors": len(state_dict),
    }


def materialize_wan_rope_after_meta_load(transformer, device: torch.device) -> dict:
    """Rebuild Wan's non-persistent RoPE buffers after a meta/assign load.

    PhysRVG's strict state dict contains every persistent parameter and buffer,
    but Wan registers ``rope.freqs_cos`` and ``rope.freqs_sin`` with
    ``persistent=False``.  A model constructed on ``meta`` therefore needs
    exactly this small module rebuilt through its original constructor.  No
    model parameter or checkpoint tensor is replaced here.
    """

    rope = getattr(transformer, "rope", None)
    if rope is None:
        raise AttributeError("Wan transformer lacks its rope module")
    buffers = dict(rope.named_buffers(recurse=False))
    expected = {"freqs_cos", "freqs_sin"}
    if set(buffers) != expected:
        raise RuntimeError(f"unexpected Wan RoPE buffers: {sorted(buffers)}")
    meta_names = [name for name, value in buffers.items() if value.is_meta]
    if meta_names:
        if set(meta_names) != expected:
            raise RuntimeError(f"partially materialized Wan RoPE buffers: {meta_names}")
        with torch.device(device):
            replacement = type(rope)(
                rope.attention_head_dim,
                rope.patch_size,
                rope.max_seq_len,
            )
        transformer.rope = replacement

    remaining_meta_parameters = [
        name for name, parameter in transformer.named_parameters() if parameter.is_meta
    ]
    remaining_meta_buffers = [
        name for name, value in transformer.named_buffers() if value.is_meta
    ]
    if remaining_meta_parameters or remaining_meta_buffers:
        raise RuntimeError(
            "meta tensors remain after PhysRVG load: "
            f"parameters={remaining_meta_parameters[:8]} "
            f"buffers={remaining_meta_buffers[:8]}"
        )
    materialized = dict(transformer.rope.named_buffers(recurse=False))
    return {
        "rebuilt_nonpersistent_buffers": sorted(meta_names),
        "rope_device": str(materialized["freqs_cos"].device),
        "rope_dtype": str(materialized["freqs_cos"].dtype),
    }


def cached_flow_matching_loss(
    model,
    scheduler: FlowMatchEulerDiscreteScheduler,
    clean_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    *,
    context_latent_frames: int,
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    batch_size, _, latent_frames, latent_height, latent_width = clean_latents.shape
    if not 0 < context_latent_frames < latent_frames:
        raise ValueError(
            f"context_latent_frames must be in [1, {latent_frames - 1}], "
            f"got {context_latent_frames}"
        )

    timestep_index = torch.randint(
        0, len(scheduler.timesteps), (1,), device=clean_latents.device
    )
    timestep = scheduler.timesteps[timestep_index]
    noise = torch.randn_like(clean_latents)
    noisy_latents = scheduler.scale_noise(clean_latents, timestep, noise)
    target = noise - clean_latents

    # Match PhysRVG's expand-timesteps path: keep context clean and train only future latents.
    condition_mask = torch.ones(
        1,
        1,
        latent_frames,
        latent_height,
        latent_width,
        device=clean_latents.device,
        dtype=clean_latents.dtype,
    )
    condition_mask[:, :, :context_latent_frames] = 0
    model_input = (1 - condition_mask) * clean_latents + condition_mask * noisy_latents

    _, patch_height, patch_width = patch_size
    token_timesteps = (
        condition_mask[0, 0, :, ::patch_height, ::patch_width] * timestep
    ).flatten()
    token_timesteps = token_timesteps.unsqueeze(0).expand(batch_size, -1)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        prediction = model(
            hidden_states=model_input,
            timestep=token_timesteps,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_image=None,
            return_dict=False,
        )[0]
    prediction = prediction[:, :, context_latent_frames:].float()
    target = target[:, :, context_latent_frames:].float()
    return F.mse_loss(prediction, target)


def build_optimizer(args, parameters):
    if args.optimizer == "paged_adamw8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as error:
            raise RuntimeError(
                "--optimizer paged_adamw8bit requires bitsandbytes"
            ) from error
        return bnb.optim.PagedAdamW8bit(
            parameters,
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
            eps=1e-8,
        )
    return torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )


def save_lora_checkpoint(model, checkpoint_dir: Path) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    unwrap_model(model).save_pretrained(checkpoint_dir)


def prune_checkpoints(checkpoint_root: Path, keep: int) -> None:
    if keep <= 0:
        return
    checkpoints = sorted(
        path
        for path in checkpoint_root.glob("step-*")
        if path.is_dir() and path.name[5:].isdigit()
    )
    for checkpoint in checkpoints[:-keep]:
        shutil.rmtree(checkpoint)


def distributed_mean(value: torch.Tensor) -> torch.Tensor:
    reduced = value.detach().clone()
    if dist.is_initialized():
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced /= dist.get_world_size()
    return reduced


def main(args: argparse.Namespace) -> None:
    from fastvideo.models.wan_v2v.model_wan_v2v import WanTransformer3DModel

    if not torch.cuda.is_available():
        raise RuntimeError("Full-SA training requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    run_dir = Path(args.output_dir).expanduser().resolve() / args.run_name
    checkpoint_root = run_dir / "checkpoints"
    if rank == 0:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    dataset = CachedPyBulletDataset(
        root=args.pybullet_root,
        vae_cache_dir=args.vae_cache_dir,
        prompt_cache_dir=args.prompt_cache_dir,
        split="train",
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        collate_fn=dataset.collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.dataloader_num_workers > 0,
    )

    transformer = WanTransformer3DModel.from_pretrained(
        args.model_id,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    if len(transformer.blocks) != args.expected_num_blocks:
        raise RuntimeError(
            f"Expected {args.expected_num_blocks} DiT blocks, found {len(transformer.blocks)}"
        )
    physrvg_dit = load_physrvg_dit(transformer, args.physrvg_dit_checkpoint)

    patch_size = tuple(transformer.config.patch_size)
    targets = full_sa_target_modules(args.expected_num_blocks)
    if args.resume_lora_checkpoint:
        transformer = PeftModel.from_pretrained(
            transformer, args.resume_lora_checkpoint, is_trainable=True
        )
    else:
        transformer = get_peft_model(
            transformer,
            LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=targets,
                init_lora_weights=True,
                bias="none",
            ),
        )
    lora_stats = validate_full_sa_lora(
        transformer,
        expected_targets=targets,
        expected_trainable_params=args.expected_trainable_params,
    )
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    transformer.to(device)
    transformer.train()
    model = DDP(transformer, device_ids=[local_rank]) if world_size > 1 else transformer

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = build_optimizer(args, trainable_parameters)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    flow_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.model_id, subfolder="scheduler"
    )
    flow_scheduler.set_timesteps(args.num_train_timesteps, device=device)

    effective_batch = args.train_batch_size * world_size * args.gradient_accumulation_steps
    wandb_run = None
    if rank == 0:
        config = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "arguments": vars(args),
            "dataset": dataset.stats,
            "physrvg_dit": physrvg_dit,
            "full_sa": lora_stats,
            "effective_batch_size": effective_batch,
            "world_size": world_size,
        }
        (run_dir / "resolved_config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(config, indent=2, ensure_ascii=True, sort_keys=True), flush=True)
        wandb_run = _init_wandb(run_dir, config)

    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(
        total=args.max_train_steps,
        desc="Optimizer steps",
        disable=rank != 0,
    )
    global_step = 0
    micro_step = 0
    accumulated_loss = torch.zeros((), device=device, dtype=torch.float32)
    started = time.monotonic()
    metrics_path = run_dir / "train_metrics.jsonl"

    for epoch in range(args.num_epochs):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            should_sync = (micro_step + 1) % args.gradient_accumulation_steps == 0
            sync_context = nullcontext() if should_sync or world_size == 1 else model.no_sync()
            latents = batch["latents"].to(device=device, dtype=torch.bfloat16, non_blocking=True)
            prompt_embeds = batch["prompt_embeds"].to(
                device=device, dtype=torch.bfloat16, non_blocking=True
            )
            with sync_context:
                loss = cached_flow_matching_loss(
                    model,
                    flow_scheduler,
                    latents,
                    prompt_embeds,
                    context_latent_frames=args.context_latent_frames,
                    patch_size=patch_size,
                )
                (loss / args.gradient_accumulation_steps).backward()
            accumulated_loss += loss.detach().float()
            micro_step += 1

            if not should_sync:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters, args.max_grad_norm
            )
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            mean_loss = distributed_mean(
                accumulated_loss / args.gradient_accumulation_steps
            )
            accumulated_loss.zero_()
            if rank == 0:
                elapsed = time.monotonic() - started
                metric = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": float(mean_loss.item()),
                    "grad_norm": float(grad_norm),
                    "learning_rate": float(lr_scheduler.get_last_lr()[0]),
                    "elapsed_seconds": elapsed,
                }
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metric, sort_keys=True) + "\n")
                if wandb_run is not None:
                    wandb_run.log(metric, step=global_step, commit=True)
                progress.update(1)
                progress.set_postfix(loss=f"{mean_loss.item():.5f}")

            if global_step % args.checkpointing_steps == 0:
                if rank == 0:
                    save_lora_checkpoint(
                        model, checkpoint_root / f"step-{global_step:06d}"
                    )
                    prune_checkpoints(checkpoint_root, args.checkpoints_total_limit)
                if dist.is_initialized():
                    dist.barrier()

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    if rank == 0:
        save_lora_checkpoint(model, run_dir / "final_lora")
        progress.close()
        if wandb_run is not None:
            wandb_run.finish()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train 30-block self-attention Q/K/V/O LoRA on cached PyBullet data"
    )
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--physrvg_dit_checkpoint", required=True)
    parser.add_argument("--resume_lora_checkpoint", default=None)
    parser.add_argument("--pybullet_root", required=True)
    parser.add_argument("--vae_cache_dir", required=True)
    parser.add_argument("--prompt_cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--context_frames", type=int, default=8)
    parser.add_argument("--context_latent_frames", type=int, default=2)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=20000)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--optimizer",
        choices=("adamw", "paged_adamw8bit"),
        default="paged_adamw8bit",
    )
    parser.add_argument("--lr_scheduler", default="constant_with_warmup")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--expected_num_blocks", type=int, default=30)
    parser.add_argument("--expected_trainable_params", type=int, default=23592960)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.context_frames != 8:
        parser.error("--context_frames must be 8 for this experiment")
    if args.context_latent_frames != 2:
        parser.error("--context_latent_frames must be 2 for the 8-frame Wan context")
    if args.num_frames != 49:
        parser.error("--num_frames must be 49 for the existing caches")
    if args.lora_rank != 32 or args.lora_alpha != 32.0 or args.lora_dropout != 0.0:
        parser.error("Full-SA requires LoRA rank=32, alpha=32, dropout=0")
    for name in (
        "train_batch_size",
        "gradient_accumulation_steps",
        "num_epochs",
        "max_train_steps",
        "checkpointing_steps",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    for name in (
        "model_id",
        "pybullet_root",
        "vae_cache_dir",
        "prompt_cache_dir",
    ):
        if not Path(getattr(args, name)).expanduser().exists():
            parser.error(f"--{name} does not exist: {getattr(args, name)}")
    if not Path(args.physrvg_dit_checkpoint).expanduser().is_file():
        parser.error(
            "--physrvg_dit_checkpoint does not exist: "
            f"{args.physrvg_dit_checkpoint}"
        )
    if args.resume_lora_checkpoint and not Path(args.resume_lora_checkpoint).expanduser().is_dir():
        parser.error(
            f"--resume_lora_checkpoint does not exist: {args.resume_lora_checkpoint}"
        )


if __name__ == "__main__":
    argument_parser = build_parser()
    arguments = argument_parser.parse_args()
    validate_args(arguments, argument_parser)
    main(arguments)
