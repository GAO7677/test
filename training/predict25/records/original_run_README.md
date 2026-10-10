# N75 official post-trained, local weights only

Fresh restart from step0, authorized by user; previous run: /data/gaoya/agent-data/outputs_v1/cosmos_predict25/20261004_n75_mb4_official.
GPU[0, 1]; per-GPU batch4, accumulation1, effective batch8; total3000 steps; local checkpoint every500.
Fixed verified batch2 encoder cache. Native official trainer plus independent extension.
W&B metrics online, LoRA upload disabled. No previous adapter or optimizer resumed.

Verified running: both ranks passed gradient/frozen parameter checks; runtime upload_checkpoints=false, local_save_every=500, W&B online.

W&B: https://wandb.ai/875222004-gy/physrvg-full-sa/runs/w3ysvt06

Native results: `/data/gaoya/agent-data/outputs_v1/cosmos_predict25/20261004_n75_mb4_localweights/physrvg-full-sa/predict25-official-posttrained-acn75/train_N75_mb4`
