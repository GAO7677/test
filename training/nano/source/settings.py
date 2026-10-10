"""Shared JSON configuration. --config selects a file for every entry point."""
import argparse
import copy
import json
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).with_name('experiment.json')

def load_config(path):
    cfg = json.loads(Path(path).read_text())
    template = {'paths': ['model', 'data', 'selection', 'cache', 'output'], 'lora': ['rank', 'alpha', 'dropout', 'target_modules'], 'data': ['count', 'frames', 'height', 'width', 'fps', 'caption_field', 'context_latents', 'context_rgb_frames'], 'training': ['seed', 'max_steps', 'per_gpu_batch', 'grad_accum', 'save_every', 'master_dtype', 'compute_dtype', 'gradient_checkpointing', 'learning_rate', 'betas', 'eps', 'weight_decay', 'max_grad_norm', 'lr_schedule', 'warmup_steps', 'smoke_samples'], 'noise': ['shift', 'waver_s', 'num_train_timesteps', 'context_scales'], 'runtime': ['train_gpus', 'prepare_gpu', 'evaluation_gpu', 'cpu_threads', 'cuda_allocator'], 'evaluation': ['steps', 'seed', 'guidance_scale', 'flow_shift', 'benchmarks']}
    template['wandb'] = ['enabled','project','entity','group','upload_checkpoints']
    for section, values in template.items():
        if section not in cfg or not isinstance(cfg[section], dict):
            raise ValueError(f'Missing configuration section: {section}')
        if set(cfg[section]) != set(values):
            raise ValueError(f'Invalid keys in {section}: expected {sorted(values)}')
    if set(cfg) != set(template):
        raise ValueError('Unknown configuration sections')
    lora, t, data = cfg['lora'], cfg['training'], cfg['data']
    if not isinstance(lora['rank'], int) or lora['rank'] <= 0 or lora['alpha'] <= 0:
        raise ValueError('LoRA rank and alpha must be positive')
    targets = lora['target_modules']
    if not targets or len(targets) != len(set(targets)) or not all(isinstance(x, str) and x for x in targets):
        raise ValueError('target_modules must be nonempty unique module suffixes')
    if not 0 <= lora['dropout'] < 1:
        raise ValueError('LoRA dropout must be in [0, 1)')
    if t['per_gpu_batch'] != 1:
        raise ValueError('Packed training currently supports per_gpu_batch=1; change grad_accum for effective batch')
    if t['master_dtype'] not in ['float32', 'bfloat16'] or t['compute_dtype'] != 'bfloat16':
        raise ValueError('Supported precision: FP32/BF16 master, BF16 compute')
    if t['lr_schedule'] not in ['cosine','constant']:
        raise ValueError('lr_schedule must be cosine or constant')
    for key in ['max_steps','grad_accum','save_every','smoke_samples']:
        if not isinstance(t[key], int) or t[key] < 1: raise ValueError(key)
    if t['warmup_steps'] < 0 or t['learning_rate'] <= 0 or t['max_grad_norm'] <= 0:
        raise ValueError('Invalid optimizer/schedule settings')
    if len(t['betas']) != 2 or any(not 0 <= x < 1 for x in t['betas']): raise ValueError('betas')
    if data['frames'] % 4 != 1 or data['height'] % 32 or data['width'] % 32:
        raise ValueError('Expected 4n+1 frames, dimensions divisible by 32')
    if not 0 < data['context_latents'] < (data['frames']-1)//4+1: raise ValueError('context_latents')
    runtime = cfg['runtime']
    gpus = runtime['train_gpus']
    if not gpus or len(gpus) != len(set(gpus)): raise ValueError('train_gpus')
    if any(not isinstance(g, int) or g < 0 or g == 4 for g in gpus + [runtime['prepare_gpu'], runtime['evaluation_gpu']]):
        raise ValueError('GPU IDs must be nonnegative; physical GPU 4 is forbidden')
    if not cfg['noise']['context_scales'] or any(not 0 <= x <= 1 for x in cfg['noise']['context_scales'].values()):
        raise ValueError('context_scales')
    for key in ['cache','output']:
        p = Path(cfg['paths'][key]).resolve()
        if not p.is_relative_to('/data/gaoya'): raise ValueError(f'{key} must be under /data/gaoya')
    return copy.deepcopy(cfg)

_selector = argparse.ArgumentParser(add_help=False)
_selector.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
CONFIG_PATH = _selector.parse_known_args()[0].config.resolve()
CFG = load_config(CONFIG_PATH)
