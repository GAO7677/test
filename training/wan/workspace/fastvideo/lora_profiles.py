"""Shared LoRA profiles used by the PhysRVG training entry points."""

RL_LORA_TARGET_MODULES = (
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "net.0.proj",
    "net.2",
)


def rl_lora_config_kwargs() -> dict:
    """Return the exact LoRA settings used by ``scripts/finetune/train_rl.sh``."""
    return {
        "r": 32,
        "lora_alpha": 64,
        "init_lora_weights": "gaussian",
        "target_modules": list(RL_LORA_TARGET_MODULES),
    }
