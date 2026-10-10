#!/usr/bin/env bash
set -euo pipefail
cd /home/gaoya/Code_Video/Code_MODEL/cosmos-predict2.5
source projects/generic_acn75/env.sh
export IMAGINAIRE_OUTPUT_ROOT=/data/gaoya/agent-data/outputs_v1/cosmos_predict25/20261004_n75_mb4_localweights
export COSMOS_PHASE=train COSMOS_MECHANISM=N75 WANDB_MODE=online
run_dir="$IMAGINAIRE_OUTPUT_ROOT/physrvg-full-sa/predict25-official-posttrained-acn75/train_N75_mb4"
if [[ -e "$run_dir/checkpoints/latest_checkpoint.txt" ]]; then
    echo 'Existing checkpoint found: use an explicit resume plan.' >&2
    exit 1
fi
mkdir -p "$IMAGINAIRE_OUTPUT_ROOT"
cp projects/generic_acn75/n75_mb4_settings.json "$IMAGINAIRE_OUTPUT_ROOT/settings.json"
python -m projects.generic_acn75.verify_cache > "$IMAGINAIRE_OUTPUT_ROOT/cache_preflight.log" 2>&1
python -m projects.generic_acn75.summarize_memory_probe > "$IMAGINAIRE_OUTPUT_ROOT/memory_preflight.log" 2>&1
python -m torch.distributed.run --standalone --nproc_per_node=2 scripts/train.py --config=projects/generic_acn75/n75_mb4_config.py > "$IMAGINAIRE_OUTPUT_ROOT/train_N75.log" 2>&1
