"""Cosmos sampling with shared Wan RGB/prompt inputs and matched output format."""
import argparse
import ast
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def load_wan_reader(path):
    # Reuse the exact four loader functions without importing the Wan DiT stack.
    import imageio
    from PIL import Image
    from torchvision.transforms import functional as transform_functional, InterpolationMode
    names = {'crop_and_resize', 'open_video_reader', 'video_length', 'load_context_video'}
    source = ast.parse(Path(path).read_text())
    nodes = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    namespace = dict(imageio=imageio, Image=Image, transform_functional=transform_functional,
                     InterpolationMode=InterpolationMode, Path=Path, Any=Any)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['load_context_video']


@contextmanager
def exact_context(pipe, frames, config, audit):
    import numpy as np
    import torch
    rgb = np.stack([np.asarray(frame) for frame in frames])
    assert rgb.shape == (8, 512, 896, 3)
    audit['rgb8_sha256'] = hashlib.sha256(rgb.tobytes()).hexdigest()
    context = torch.from_numpy(rgb.copy()).permute(3, 0, 1, 2).unsqueeze(0).float() / 127.5 - 1
    original = pipe._encode_video
    calls = []

    def encode(_native_padded_input):
        latent = original(context.to(_native_padded_input.device))
        assert latent.shape[2] == 2, latent.shape
        total = (config['num_frames'] - 1) // int(pipe.vae.config.scale_factor_temporal) + 1
        padded = latent.new_zeros(latent.shape[:2] + (total,) + latent.shape[3:])
        padded[:, :, :2] = latent
        calls.append(dict(rgb_frames=8, condition_latents=2, total_latents=total,
                          latent_sha256=hashlib.sha256(latent.detach().float().cpu().numpy().tobytes()).hexdigest()))
        return padded.to(_native_padded_input.dtype)

    pipe._encode_video = encode
    try:
        yield
        assert len(calls) == 1, calls
        audit['encode_calls'] = calls
    finally:
        pipe._encode_video = original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--case', action='append')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    root = Path(cfg['output_root'])
    source = Path(cfg['test100_root'])
    resolved = json.loads((source / 'resolved_config.json').read_text())
    active = json.loads((source / 'active_cases.json').read_text())['cases']
    assert len(active) == cfg['expected_cases'] == 89
    assert (cfg['context_frames'], cfg['context_latent_indexes']) == (8, [0, 1])
    assert (cfg['num_frames'], cfg['height'], cfg['width'], cfg['fps']) == (49, 512, 896, 24)
    selected = args.case or active
    assert len(set(selected)) == len(selected) and set(selected) <= set(active)
    inputs = {}
    for case in selected:
        row = resolved['cases'][case]
        assert isinstance(row['prompt'], str) and row['prompt'].strip()
        for key in ['context_video', 'gt_video']:
            assert Path(row[key]).is_file(), row[key]
        inputs[case] = dict(prompt=row['prompt'], context_video=row['context_video'], gt_video=row['gt_video'],
                            context_sha256=sha(row['context_video']),
                            prompt_sha256=hashlib.sha256(row['prompt'].encode()).hexdigest())
    provenance = dict(config_sha256=sha(args.config), script_sha256=sha(__file__),
                      loader_sha256=sha(cfg['wan_loader']),
                      active_cases_sha256=sha(source/'active_cases.json'),
                      negative_prompt_sha256=sha(cfg['negative_prompt_file']))
    fingerprint = hashlib.sha256(json.dumps(dict(config=cfg, provenance=provenance), sort_keys=True).encode()).hexdigest()
    save(root / ('config_check.json' if args.check_only else 'resolved_config.json'),
         dict(config=cfg, cases=inputs, provenance=provenance, fingerprint=fingerprint,
              status='STATIC_PASS_GPU_NOT_RUN' if args.check_only else 'STARTING'))
    if args.check_only:
        print(f'STATIC_PASS: {len(inputs)} cases; GPU not loaded')
        return
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    assert visible in ['0','1','2','3','5','6'], 'Set exactly one authorized physical GPU; GPU4 forbidden'
    import numpy as np
    import torch
    from diffusers import Cosmos3OmniPipeline, UniPCMultistepScheduler
    from diffusers.utils import export_to_video
    torch.set_num_threads(2)
    reader = load_wan_reader(cfg['wan_loader'])
    pipe = Cosmos3OmniPipeline.from_pretrained(cfg['model_path'], torch_dtype=torch.bfloat16,
        safety_checker=None, enable_safety_checker=False, low_cpu_mem_usage=True).to('cuda')
    pipe.vae.to(dtype=torch.float32)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=cfg['flow_shift'])
    negative = json.dumps(json.loads(Path(cfg['negative_prompt_file']).read_text()), ensure_ascii=True, separators=(',', ':'))
    completed = []
    for case, row in inputs.items():
        dest = root / 'results' / case
        receipt = dest / 'result.json'
        if receipt.exists():
            old = json.loads(receipt.read_text())
            if old.get('fingerprint') == fingerprint and old.get('input') == row and sha(dest/'video.mp4') == old['video_sha256']:
                completed.append(case)
                continue
            raise RuntimeError(f'Existing result has a different contract: {case}')
        frames = reader(Path(row['context_video']), cfg['height'], cfg['width'], max_frames=8)
        audit = {}
        start = time.time()
        save(root/'progress.json', dict(status='RUNNING', current=case, selected_cases=selected,
             completed=len(completed), total=len(selected), completed_cases=completed, gpu=visible))
        with torch.inference_mode(), exact_context(pipe, frames, cfg, audit):
            result = pipe(prompt=row['prompt'], negative_prompt=negative, video=frames,
                condition_frame_indexes_vision=cfg['context_latent_indexes'], condition_video_keep='first',
                num_frames=cfg['num_frames'], height=cfg['height'], width=cfg['width'], fps=cfg['fps'],
                num_inference_steps=cfg['num_inference_steps'], guidance_scale=cfg['guidance_scale'],
                enable_sound=cfg['enable_sound'], add_resolution_template=cfg['add_resolution_template'],
                add_duration_template=cfg['add_duration_template'],
                generator=torch.Generator(device='cuda').manual_seed(cfg['seed']))
        array = np.asarray(result.video)
        assert array.shape == (49,512,896,3) and np.isfinite(array).all(), array.shape
        dest.mkdir(parents=True, exist_ok=True)
        export_to_video(result.video, str(dest/'video.partial.mp4'), fps=24, macro_block_size=1)
        import av
        with av.open(str(dest/'video.partial.mp4')) as container:
            stream = container.streams.video[0]
            assert (stream.width,stream.height,stream.frames,float(stream.average_rate)) == (896,512,49,24.0)
        (dest/'video.partial.mp4').replace(dest/'video.mp4')
        save(receipt, dict(status='COMPLETE',case=case,model='Cosmos3-Nano matched IO',fingerprint=fingerprint,
             input=row,context_audit=audit,protocol=cfg,seed=cfg['seed'],provenance=provenance,
             scheduler=dict(pipe.scheduler.config),output_video=str(dest/'video.mp4'),
             video_sha256=sha(dest/'video.mp4'),seconds=time.time()-start))
        completed.append(case)
        if len(completed) == 1:
            save(root/'validation.json', dict(status='REAL_GPU_OUTPUT_CONTRACT_PASS', case=case,
                 fingerprint=fingerprint, context_audit=audit, output_shape=list(array.shape),
                 num_inference_steps=cfg['num_inference_steps'], gpu=visible,
                 limitation='8 RGB frames encoded into 2 context latents; equal latent count does not prove equal effective temporal support across models.'))
        save(root/'progress.json', dict(status='RUNNING', completed=len(completed), total=len(selected),
             completed_cases=completed, current=case, gpu=visible))
        print('COMPLETE', case, len(completed), '/', len(selected), flush=True)
    save(root/'progress.json',dict(status='COMPLETE',selected_cases=selected,completed=len(selected),
         total=len(selected),completed_cases=completed,gpu=visible))


if __name__ == '__main__':
    main()
