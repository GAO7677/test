"""PhysRVG RL-LoRA control training with the original flow-matching loss only."""

from train_full_sa_pybullet_aux import main


if __name__ == "__main__":
    main("raw", default_lora_profile="rl")
