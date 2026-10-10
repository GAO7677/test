"""Shared data and flow contracts for the generic2175 Cosmos A/C/N75 experiment."""
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from settings import CFG, CONFIG_PATH
MODEL, DATA, SELECTION, CACHE, OUTPUT = (Path(CFG['paths'][k]) for k in ['model','data','selection','cache','output'])
TARGETS = CFG['lora']['target_modules']
SCALES = CFG['noise']['context_scales']
CONTEXT = CFG['data']['context_latents']


class LoRALinear(torch.nn.Module):
    """Zero-initialized low-rank residual; no dependency on optional quantization packages."""
    def __init__(self, base, rank=None, alpha=None, dropout=None):
        super().__init__()
        rank = CFG['lora']['rank'] if rank is None else rank
        alpha = CFG['lora']['alpha'] if alpha is None else alpha
        dropout = CFG['lora']['dropout'] if dropout is None else dropout
        self.dropout = torch.nn.Dropout(dropout)
        self.base_layer = base
        self.lora_A = torch.nn.Linear(base.in_features, rank, bias=False,
                                     device=base.weight.device, dtype=base.weight.dtype)
        self.lora_B = torch.nn.Linear(rank, base.out_features, bias=False,
                                     device=base.weight.device, dtype=base.weight.dtype)
        torch.nn.init.normal_(self.lora_A.weight, std=1/rank)
        torch.nn.init.zeros_(self.lora_B.weight)
        self.scale = alpha/rank

    def forward(self, x):
        return self.base_layer(x) + self.lora_B(self.lora_A(self.dropout(x)))*self.scale


def inject_lora(model, rank=None, alpha=None, targets=None, dropout=None):
    targets = TARGETS if targets is None else targets
    matches = {target: [] for target in targets}
    selected = []
    for name, module in model.named_modules():
        hit = [t for t in targets if name == t or name.endswith('.' + t)]
        if hit:
            if len(hit) != 1 or not isinstance(module, torch.nn.Linear):
                raise ValueError(f'Ambiguous or non-linear target: {name}')
            matches[hit[0]].append(name)
            selected.append((name, module))
    if any(not names for names in matches.values()):
        raise ValueError(f'Unmatched LoRA targets: {[t for t, names in matches.items() if not names]}')
    for name, module in selected:
        parent_name, attr = name.rsplit('.', 1)
        setattr(model.get_submodule(parent_name), attr, LoRALinear(module, rank, alpha, dropout=dropout))
    return [name for name, _ in selected]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def context_pair(future_sigma, scale, paired_sigmas):
    """Wan experiment endpoint-exact / nearest-paired-sigma policy, in Cosmos time units."""
    if not math.isfinite(scale) or not 0 <= scale <= 1:
        raise ValueError(scale)
    if scale == 0:
        return torch.zeros_like(future_sigma)
    if scale == 1:
        return future_sigma.clone()
    grid = paired_sigmas.to(future_sigma)
    return grid[(grid - future_sigma * scale).abs().argmin()].reshape_as(future_sigma)


def sample_sigma(generator, shift=None):
    shift = CFG["noise"]["shift"] if shift is None else shift
    # Native Cosmos TrainTimeSampler('waver'), _WAVER_MODE_S=1.29.
    u = torch.rand((), generator=generator)
    t = 1 - u - CFG["noise"]["waver_s"] * (torch.cos(torch.pi / 2 * u).square() - 1 + u)
    return shift * t / (1 + (shift - 1) * t)


def noise_input(clean, noise, sigma, context_sigma):
    # All arithmetic before model input casting is FP32, including BF16 cached targets.
    clean, noise = clean.float(), noise.float()
    levels = sigma.to(clean).expand(clean.shape[2]).clone()
    levels[:CONTEXT] = context_sigma.to(clean)
    noisy = (1 - levels[None, None, :, None, None]) * clean + levels[None, None, :, None, None] * noise
    return noisy, noise - clean, levels


def future_loss(prediction, target):
    # Mean over only the eleven future latent frames, independent of context noise.
    return (prediction[:, :, CONTEXT:].float() - target[:, :, CONTEXT:].float()).square().mean()


def pack(model, tokens, latent, levels, mode):
    """Use official Diffusers Cosmos packing; one sample, no cross-sample attention."""
    from diffusers import Cosmos3OmniPipeline
    helper = SimpleNamespace(transformer=model, vae_scale_factor_temporal=4)
    text = Cosmos3OmniPipeline._prepare_text_segment(helper, tokens, latent.device)
    cond = list(range(CONTEXT)) if SCALES[mode] == 0 else []
    vision = Cosmos3OmniPipeline._prepare_vision_segment(
        helper, latent, bool(cond), text['vision_start_temporal_offset'],
        float(CFG['data']['fps']), text['und_len'], latent.device, condition_frame_indexes=cond,
    )
    frame_indices = vision['vision_noisy_frame_indexes'][0]
    stride = vision['vision_token_shapes'][0][1] * vision['vision_token_shapes'][0][2]
    kwargs = {k: text[k] for k in ['input_ids', 'text_indexes', 'und_len']}
    kwargs.update({k: v for k, v in vision.items() if k.startswith('vision_') and k != 'vision_mrope_ids'})
    kwargs.update(
        position_ids=torch.cat([text['text_mrope_ids'], vision['vision_mrope_ids']], dim=1),
        sequence_length=text['und_len'] + vision['num_vision_tokens'],
        vision_tokens=[latent],
        vision_timesteps=(levels[frame_indices] * CFG['noise']['num_train_timesteps']).repeat_interleave(stride),
        return_dict=False,
    )
    return kwargs
