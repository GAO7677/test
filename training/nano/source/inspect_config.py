"""Inspect actual LoRA matches on a meta model without loading weights or using GPUs."""
import argparse
import json
import torch
from diffusers import Cosmos3OmniTransformer
from common import CFG, CONFIG_PATH, MODEL, inject_lora


def inspect():
    with torch.device('meta'):
        model = Cosmos3OmniTransformer.from_config(Cosmos3OmniTransformer.load_config(MODEL/'transformer'))
        model.requires_grad_(False)
        names = inject_lora(model)
    return dict(config=str(CONFIG_PATH), lora=CFG['lora'], matched_modules=names,
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),
                effective_batch=CFG['training']['per_gpu_batch']*len(CFG['runtime']['train_gpus'])*CFG['training']['grad_accum'])

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default=CONFIG_PATH)
    parser.parse_args()
    print(json.dumps(inspect(),indent=2))
