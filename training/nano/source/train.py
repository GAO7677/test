"""Two-rank FSDP2 Cosmos LoRA training with adapter-only, resumable checkpoints."""
import argparse
import gc
import faulthandler
import importlib.metadata
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor, distribute_tensor
from diffusers import Cosmos3OmniTransformer, FlowMatchEulerDiscreteScheduler
from safetensors.torch import load_file, save_file

from common import CFG, CONFIG_PATH, MODEL, CACHE, OUTPUT, TARGETS, SCALES, sha, write_json
from common import context_pair, sample_sigma, noise_input, future_loss, pack, inject_lora


def adapter_parameters(model):
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def local_tensor(t):
    return (t.to_local() if isinstance(t, DTensor) else t).detach().cpu().clone()


def save_checkpoint(model, optimizer, generator, root, step, config, rank, world):
    folder = root / f'step-{step:06d}'
    folder.mkdir(parents=True, exist_ok=True)
    params = adapter_parameters(model)
    # All ranks participate in gathering adapters; frozen 15B base weights are never saved.
    gathered = {}
    for name, param in params.items():
        value = param.full_tensor() if isinstance(param, DTensor) else param
        if rank == 0:
            gathered[name] = value.detach().cpu().contiguous()
    if rank == 0:
        save_file(gathered, str(folder / 'adapters.safetensors'))
        write_json(folder / 'adapter_config.json', dict(rank=CFG['lora']['rank'], alpha=CFG['lora']['alpha'], dropout=CFG['lora']['dropout'], target_modules=TARGETS,
                    base_model=str(MODEL), format='named_lora_parameters', inference_loader='evaluate.py'))
    states = {}
    for name, p in params.items():
        states[name] = {k: {'value': local_tensor(v), 'sharded': isinstance(v, DTensor)}
                       if torch.is_tensor(v) else v for k, v in optimizer.state[p].items()}
    payload = dict(step=step, world_size=world, optimizer=states,
                   cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                   sample_rng=generator.get_state())
    tmp = folder / f'rank-{rank}.tmp'
    torch.save(payload, tmp)
    tmp.replace(folder / f'rank-{rank}.pt')
    dist.barrier()
    if rank == 0:
        write_json(folder / 'complete.json', dict(step=step, world_size=world, config=config))
        write_json(root / 'latest.json', dict(path=str(folder), step=step))
    dist.barrier()
    return folder


def restore(model, optimizer, generator, folder, config, rank, world):
    meta = json.loads((folder / 'complete.json').read_text())
    for key in ['mode', 'master_dtype', 'grad_accum', 'seed', 'manifest_sha256', 'max_steps', 'learning_rate', 'experiment', 'cache_config_sha256']:
        if meta['config'][key] != config[key]:
            raise ValueError(f'Resume mismatch: {key}')
    if meta['world_size'] != world:
        raise ValueError('Optimizer resume requires the original world size')
    weights = load_file(str(folder / 'adapters.safetensors'), device=f'cuda:{rank}')
    payload = torch.load(folder / f'rank-{rank}.pt', map_location=f'cuda:{rank}', weights_only=True)
    params = adapter_parameters(model)
    assert set(weights) == set(params)
    with torch.no_grad():
        for name, p in params.items():
            t = weights[name].to(device='cuda', dtype=p.dtype)
            if isinstance(p, DTensor):
                t = distribute_tensor(t, p.device_mesh, p.placements)
            p.copy_(t)
    for name, p in params.items():
        restored = {}
        for key, value in payload['optimizer'][name].items():
            if not isinstance(value, dict):
                restored[key] = value
            elif value['sharded']:
                restored[key] = DTensor.from_local(value['value'].to('cuda'), p.device_mesh, p.placements,
                                                  shape=p.shape, stride=p.stride())
            else:
                restored[key] = value['value'].to('cuda') if key != 'step' else value['value'].cpu()
        optimizer.state[p] = restored
    torch.set_rng_state(payload['cpu_rng'].cpu())
    torch.cuda.set_rng_state(payload['cuda_rng'].cpu())
    generator.set_state(payload['sample_rng'].cpu())
    dist.barrier()
    return payload['step']


def global_grad_norm(params):
    # FSDP already averages gradients; sum squares of distinct local shards across ranks.
    total = torch.zeros((), device='cuda', dtype=torch.float64)
    for p in params:
        if p.grad is not None:
            grad = p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad
            total += grad.double().square().sum()
    dist.all_reduce(total)
    norm = total.sqrt().float()
    if not torch.isfinite(norm):
        raise FloatingPointError('non-finite gradient norm')
    scale = (CFG['training']['max_grad_norm'] / (norm + 1e-6)).clamp(max=1)
    for p in params:
        if p.grad is not None:
            p.grad.mul_(scale.item())
    return norm.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=CONFIG_PATH)
    parser.add_argument('--mode', choices=list(SCALES), required=True)
    parser.add_argument('--max-steps', type=int, default=CFG['training']['max_steps'])
    parser.add_argument('--stop-after', type=int, default=0)
    parser.add_argument('--grad-accum', type=int, default=CFG['training']['grad_accum'])
    parser.add_argument('--save-every', type=int, default=CFG['training']['save_every'])
    parser.add_argument('--master-dtype', choices=['float32', 'bfloat16'], default=CFG['training']['master_dtype'])
    parser.add_argument('--run-name')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--verify-roundtrip', action='store_true')
    args = parser.parse_args()
    if args.smoke or args.stop_after:
        faulthandler.dump_traceback_later(90, repeat=True)
    physical = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    assert physical == list(map(str, CFG['runtime']['train_gpus'])), physical
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    assert world == len(physical)
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(CFG['runtime']['cpu_threads'])
    dist.init_process_group('nccl')
    mesh = init_device_mesh('cuda', (world,))
    torch.manual_seed(CFG['training']['seed'])
    CFG['training'].update(max_steps=args.max_steps, grad_accum=args.grad_accum,
                           save_every=args.save_every, master_dtype=args.master_dtype)
    t = CFG['training']
    config = dict({k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                  experiment=CFG, seed=t['seed'], learning_rate=t['learning_rate'], warmup_steps=t['warmup_steps'],
                  manifest_sha256=sha(CACHE / 'manifest.json'), cache_config_sha256=sha(CACHE / 'config.json'),
                  base_model=str(MODEL), weight_loading='meta_fsdp_direct_cuda', context_latents=CFG['data']['context_latents'], noise_sampler='cosmos_waver',
                  context_mapping='endpoint_exact_nearest_paired_sigma',
                  loss='future_latents_mean', lora_rank=CFG['lora']['rank'], lora_alpha=CFG['lora']['alpha'],
                  packages={p: importlib.metadata.version(p) for p in ['torch','diffusers','transformers','peft']})
    root = OUTPUT / (args.run_name or args.mode)
    if root.exists() and (root / 'latest.json').exists() and not args.resume:
        raise RuntimeError('Existing checkpoint: use --resume or a new run name')
    root.mkdir(parents=True, exist_ok=True)
    cache_meta = json.loads((CACHE / 'config.json').read_text())
    for key in ['count','frames','height','width','fps']:
        if cache_meta[key] != CFG['data'][key]:
            raise ValueError(f'Cache does not match configured data.{key}; prepare a new cache directory')
    if cache_meta['prompt_field'] != CFG['data']['caption_field']:
        raise ValueError('Cache caption field mismatch')
    if cache_meta['selection_sha256'] != sha(Path(CFG['paths']['selection'])):
        raise ValueError('Cache sample selection mismatch')
    records = json.loads((CACHE / 'manifest.json').read_text())
    if args.smoke:
        records = sorted(records, key=lambda r: len(r['input_ids']), reverse=True)[:t['smoke_samples']]
    else:
        complete = json.loads((CACHE / 'complete.json').read_text())
        assert complete['count'] == len(records) == CFG['data']['count']
        assert complete['manifest_sha256'] == config['manifest_sha256']
    run = None
    if rank == 0 and CFG['wandb']['enabled']:
        import wandb
        wb_dir = root / 'wandb'
        wb_dir.mkdir(exist_ok=True)
        id_path = root / 'wandb_run.json'
        run_id = json.loads(id_path.read_text())['id'] if id_path.exists() else wandb.util.generate_id()
        write_json(id_path, dict(id=run_id))
        run = wandb.init(project=CFG['wandb']['project'], entity=CFG['wandb']['entity'],
                         group=CFG['wandb']['group'], name=f"cosmos3-nano-{args.mode}-r{CFG['lora']['rank']}-seed{t['seed']}",
                         id=run_id, resume='allow', allow_val_change=True, mode='online', dir=str(wb_dir), config=config,
                         settings=wandb.Settings(init_timeout=120, x_disable_stats=True, x_disable_meta=True))
        write_json(id_path, dict(id=run.id, url=run.url, project=run.project, entity=run.entity))
        print(f'WANDB_URL={run.url}', flush=True)
    dist.barrier()
    print(f'rank={rank} loading safetensors directly into CUDA FSDP shards', flush=True)
    dtype = getattr(torch, args.master_dtype)
    from cuda_loader import load_sharded_cuda
    model = load_sharded_cuda(MODEL / 'transformer', mesh, local_rank, dtype)
    if t['gradient_checkpointing']:
        model.enable_gradient_checkpointing()
    if rank == 0:
        write_json(root / 'trainable_modules.json', dict(lora=CFG['lora'], trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad), frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad), names=list(adapter_parameters(model))))
    model.train()
    gc.collect()
    torch.cuda.empty_cache()
    params = list(adapter_parameters(model).values())
    optimizer = torch.optim.AdamW(params, lr=t['learning_rate'], betas=tuple(t['betas']), eps=t['eps'], weight_decay=t['weight_decay'], foreach=False)
    generator = torch.Generator().manual_seed(t['seed'] + rank)
    paired = FlowMatchEulerDiscreteScheduler(num_train_timesteps=CFG['noise']['num_train_timesteps'], shift=CFG['noise']['shift']).sigmas[:CFG['noise']['num_train_timesteps']]
    completed = 0
    if args.resume:
        folder = Path(json.loads((root / 'latest.json').read_text())['path'])
        completed = restore(model, optimizer, generator, folder, config, rank, world)
        print(f'rank={rank} resumed step={completed}', flush=True)
    if rank == 0:
        write_json(root / 'resolved_config.json', config)
    stop = min(args.max_steps, completed + args.stop_after) if args.stop_after else args.max_steps
    perm_epoch, permutation = -1, None
    torch.cuda.reset_peak_memory_stats()
    for step in range(completed, stop):
        start = time.time()
        warmup = t['warmup_steps']
        factor = min((step+1)/warmup, 1.0) if warmup else 1.0
        if t['lr_schedule'] == 'cosine':
            factor *= 0.5*(1+math.cos(math.pi*max(0,step-warmup)/max(1,args.max_steps-warmup)))
        lr = t['learning_rate'] * factor
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        losses, used = [], []
        for micro in range(args.grad_accum):
            index = (step*args.grad_accum+micro)*world+rank
            epoch, offset = divmod(index, len(records))
            if perm_epoch != epoch:
                permutation = torch.randperm(len(records), generator=torch.Generator().manual_seed(t['seed']+epoch)).tolist()
                perm_epoch = epoch
            record = records[permutation[offset]]
            used.append(record['key'])
            clean = load_file(str(CACHE / record['latent_file']))['latent'][None].to('cuda').float()
            sigma = sample_sigma(generator)
            context_sigma = context_pair(sigma, SCALES[args.mode], paired)
            noise = torch.randn(clean.shape, generator=generator, dtype=torch.float32).to('cuda')
            noisy, target, levels = noise_input(clean, noise, sigma, context_sigma)
            kwargs = pack(model, record['input_ids'], noisy.bfloat16(), levels, args.mode)
            pred = model(**kwargs)[0][0]
            loss = future_loss(pred, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'non-finite loss at {step}/{micro}')
            (loss / args.grad_accum).backward()
            losses.append(loss.detach())
            del pred, kwargs, clean, noise, noisy, target, loss
        norm = global_grad_norm(params)
        assert all(p.grad is not None for p in params), 'Missing adapter gradient'
        optimizer.step()
        average = torch.stack(losses).mean()
        dist.all_reduce(average)
        average /= world
        torch.cuda.synchronize()
        log = dict(peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30, step=step+1, loss=average.item(), grad_norm=norm, lr=lr,
                   seconds=time.time()-start, peak_GiB=torch.cuda.max_memory_allocated()/2**30,
                   allocated_GiB=torch.cuda.memory_allocated()/2**30, samples=used, mode=args.mode)
        print(json.dumps(log), flush=True)
        if rank == 0:
            with (root / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(log)+'\n')
            write_json(root / 'progress.json', log)
            if run is not None:
                run.log({k:v for k,v in log.items() if isinstance(v, (int,float))}, step=step+1)
        if (step+1) % args.save_every == 0 or step+1 == stop:
            folder = save_checkpoint(model, optimizer, generator, root, step+1, config, rank, world)
            if rank == 0 and run is not None and CFG['wandb']['upload_checkpoints']:
                artifact = wandb.Artifact(f'cosmos3-nano-{args.mode}-{run.id}', type='model', metadata={'step':step+1, 'lora':CFG['lora']})
                for name in ['adapters.safetensors','adapter_config.json','complete.json']:
                    artifact.add_file(str(folder/name), name=name)
                run.log_artifact(artifact, aliases=[f'step-{step+1:06d}', 'latest'])
            if args.verify_roundtrip:
                old = {n:local_tensor(p) for n,p in adapter_parameters(model).items()}
                restore(model, optimizer, generator, folder, config, rank, world)
                assert all(torch.equal(old[n], local_tensor(p)) for n,p in adapter_parameters(model).items())
                if rank == 0:
                    write_json(root / 'roundtrip_pass.json', dict(step=step+1, exact=True))
    if rank == 0:
        write_json(root / 'run_status.json', dict(completed_steps=stop, target=args.max_steps,
                   status='complete' if stop==args.max_steps else 'paused'))
    if run is not None:
        run.finish()
    faulthandler.cancel_dump_traceback_later()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
