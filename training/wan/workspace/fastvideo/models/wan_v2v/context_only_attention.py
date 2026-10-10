"""Block visibility: context reads context; future reads the whole sequence."""

import json
from pathlib import Path

import torch


def context_only_attention(query, key, value, context_tokens, dispatch, **kwargs):
    """Inputs follow Wan's [batch, sequence, heads, head_dim] convention."""
    if not 0 < context_tokens < query.shape[1] or query.shape[1] != key.shape[1]:
        raise ValueError("expected self-attention with nonempty context and future")
    options = dict(attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs)
    context = dispatch(
        query[:, :context_tokens], key[:, :context_tokens], value[:, :context_tokens], **options
    )
    future = dispatch(query[:, context_tokens:], key, value, **options)
    return torch.cat((context, future), dim=1)


def load_attention_policy(checkpoint):
    """Legacy checkpoints retain full attention; new checkpoints carry a sidecar."""
    if checkpoint is None:
        return {"mode": "full"}
    checkpoint = Path(checkpoint)
    sidecar = checkpoint / "self_attention_policy.json"
    if sidecar.is_file():
        policy = json.loads(sidecar.read_text())
        if policy != {"mode": "context_only", "context_latent_frames": 2}:
            raise ValueError(f"Unsupported attention policy: {policy}")
        return policy
    # Fail closed if the sidecar was lost inside an otherwise identifiable run.
    # Step checkpoints live in run/checkpoints/step-*, final_lora in run/.
    for resolved in (
        checkpoint.parent / "resolved_config.json",
        checkpoint.parent.parent / "resolved_config.json",
    ):
        if not resolved.is_file():
            continue
        arguments = json.loads(resolved.read_text()).get("arguments", {})
        if arguments.get("self_attention_mode", "full") != "full":
            raise ValueError(f"Missing attention sidecar: {sidecar}")
        break
    return {"mode": "full"}
