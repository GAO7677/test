"""Test89/Test70 inference for native post-trained baseline and exported LoRA."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from projects.generic_acn75.inference_io import (
    check_adapter_metadata, read_prefix, sha, video_info, write_json,
)

HERE = Path(__file__).resolve().parent


def plan(args):
    from cosmos_predict2.config import InferenceArguments
    settings = json.loads(args.settings.read_text())
    assert settings['input'] == dict(context_rgb_frames=8,context_latents=2,
                                     policy='encode_prefix8_then_take_first2_latents')
    assert settings['output']['frames'] >= 9 and settings['output']['frames'] % 4 == 1
    assert all(settings['output'][key] % 16 == 0 for key in ['height','width'])
    bench = settings['benchmarks'][args.benchmark]
    manifest = json.loads(Path(bench['manifest']).read_text())
    assert len(manifest['cases']) == bench['count'] and manifest['fps'] == bench['fps']
    rows = manifest['cases']
    if args.case:
        rows = {key: rows[key] for key in args.case}
    if args.limit:
        rows = dict(list(rows.items())[:args.limit])
    training = None
    adapter = None
    if args.method == 'baseline':
        if args.step is not None or args.adapter is not None:
            raise ValueError('Baseline must not receive an adapter or checkpoint step')
        name = 'Predict2.5-2B-posttrained-baseline-native'
    else:
        if args.step is None or args.step <= 0:
            raise ValueError('LoRA inference requires an explicit positive --step')
        path = args.training_settings or Path(settings['training_settings'][args.method])
        training = json.loads(path.read_text())
        assert args.method in training['modes']
        for key in ['native_checkpoint', 'native_checkpoint_sha256']:
            assert training[key] == settings['assets'][key], f'Training/inference mismatch: {key}'
        run_name = training.get('run_name', f'train_{args.method}')
        run = Path(training['output']) / training['wandb_project'] / training['wandb_group'] / run_name
        adapter = args.adapter or run / 'adapters' / f'step{args.step:06d}'
        # Prefer the immutable run snapshot for rank/alpha/targets. Mutable
        # launch settings identify the active output directory only.
        snapshot = next((parent / 'settings.json' for parent in list(adapter.parents)[:5]
                         if (parent / 'settings.json').is_file() and
                         'native_checkpoint' in json.loads((parent/'settings.json').read_text())), None)
        if snapshot is not None:
            training = json.loads(snapshot.read_text())
            assert args.method in training['modes']
            for key in ['native_checkpoint', 'native_checkpoint_sha256']:
                assert training[key] == settings['assets'][key]
        name = f'Predict2.5-2B-posttrained-{args.method}-step{args.step}-native'
    output = args.output or Path(settings['output_base']) / (
        datetime.now(timezone.utc).strftime('%Y%m%d') + '_native_bench') / args.benchmark / name
    if output.resolve().is_relative_to(Path('/home/gaoya')):
        raise ValueError('Store generated videos under /data/gaoya, not /home/gaoya')
    adapter_ready = adapter is None or ((adapter / 'metadata.json').exists() and
                                       (adapter / 'adapter.safetensors').exists())
    if adapter is not None and adapter_ready:
        check_adapter_metadata(adapter, args.method, args.step, settings['assets']['native_checkpoint'])
    if not adapter_ready and not args.dry_run:
        raise FileNotFoundError(f'Checkpoint not yet available: {adapter}')
    samples = []
    for key, row in rows.items():
        if Path(key).name != key:
            raise ValueError(f'Unsafe case name: {key}')
        # Never pass the original full GT video to the native video reader.
        # The runtime supplies an explicitly extracted first-eight-frame tensor.
        sample = InferenceArguments(name=key, inference_type='video2world',
            input_path=Path(row['context_video']), prompt=row['prompt'],
            resolution=f"{settings['output']['height']},{settings['output']['width']}",
            num_output_frames=settings['output']['frames'])
        if settings.get('negative_prompt_policy') == 'empty':
            sample = sample.model_copy(update={'negative_prompt': ''})
        elif settings.get('negative_prompt_policy') != 'official_default':
            raise ValueError('Resolve negative_prompt_policy before inference')
        samples.append(sample)
    info = dict(method=name, benchmark=args.benchmark, case_count=len(rows), cases=list(rows),
                manifest_sha256=sha(bench['manifest']),
                output=str(output), adapter=str(adapter) if adapter else None,
                checkpoint_available=adapter_ready, settings=settings,
                official_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                implementation_sha256={name:sha(HERE/name) for name in
                                       ['infer_bench.py','inference_io.py','inference_runtime.py']},
                sample_defaults=samples[0].model_dump(mode='json'),
                training_settings=training, protocol='native_prefix8_online_v1')
    canonical = {k: v for k, v in info.items() if k not in
                 ['output', 'checkpoint_available', 'cases', 'case_count', 'sample_defaults']}
    canonical['sampling'] = {k: v for k, v in info['sample_defaults'].items()
                             if k not in ['name', 'prompt', 'prompt_path', 'input_path']}
    info['protocol_sha256'] = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
    return settings, rows, samples, training, adapter, output, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', type=Path, default=HERE / 'inference_settings.json')
    parser.add_argument('--benchmark', choices=['test89', 'test70', 'pcve'], required=True)
    parser.add_argument('--method', choices=['baseline', 'A', 'C', 'N75'], required=True)
    parser.add_argument('--step', type=int)
    parser.add_argument('--adapter', type=Path, help='Native adapters/stepXXXXXX directory')
    parser.add_argument('--training-settings', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--case', action='append')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--dry-run', action='store_true', help='Validate inputs without loading models or using CUDA')
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be positive')
    settings, rows, samples, training, adapter, output, info = plan(args)
    if args.dry_run:
        checks = []
        for key, row in rows.items():
            rgb = read_prefix(row['context_video'], settings['input']['context_rgb_frames'])
            if row.get('context_video_sha256'):
                assert sha(row['context_video']) == row['context_video_sha256']
            checks.append(dict(case=key, prefix_shape=list(rgb.shape), gt=video_info(row['gt_video'])))
        write_json(output / 'dry_run.json', dict(status='PASS', plan=info, checks=checks))
        print(json.dumps(dict(status='PASS',cases=len(checks),checkpoint_available=info['checkpoint_available'],
                              output=str(output),sampling=info['sample_defaults']),ensure_ascii=False))
        return
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible.isdigit() or int(visible) == 4:
        raise ValueError('Set CUDA_VISIBLE_DEVICES to exactly one physical GPU index; GPU4 is prohibited')
    if os.environ.get('SMOKE'):
        raise ValueError('Unset SMOKE: it changes official sampling defaults')
    import torch
    from cosmos_predict2._src.predict2.inference.video2world import resize_input
    from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video
    from projects.generic_acn75.inference_runtime import BenchmarkInference
    torch.cuda.set_device(0)
    torch.set_grad_enabled(False)
    output.mkdir(parents=True, exist_ok=True)
    old = output / 'inference_config.json'
    if old.exists() and json.loads(old.read_text())['protocol_sha256'] != info['protocol_sha256']:
        raise ValueError('Output directory contains a different inference protocol; select a new output directory')
    write_json(old, info)
    pipe = BenchmarkInference(settings, training, adapter, settings['benchmarks'][args.benchmark]['fps'], samples[0].seed)
    write_json(output / 'weight_load_audit.json', pipe.weight_audit)
    for index, sample in enumerate(samples):
        row = rows[sample.name]
        dest = output / 'results' / sample.name
        receipt = dest / 'result.json'
        if receipt.exists():
            prev = json.loads(receipt.read_text())
            assert prev['protocol_sha256'] == info['protocol_sha256']
            assert prev['adapter_sha256'] == pipe.weight_audit.get('adapter_sha256')
            assert prev['video_sha256'] == sha(dest / 'video.mp4')
            continue
        start = time.time()
        write_json(output / 'status.json', dict(stage='GENERATING',case=sample.name,index=index,total=len(samples)))
        if row.get('context_video_sha256'):
            assert sha(row['context_video']) == row['context_video_sha256']
        rgb = read_prefix(row['context_video'], settings['input']['context_rgb_frames'])
        video = torch.from_numpy(rgb.copy()).permute(0, 3, 1, 2)
        video = resize_input(video, [settings['output']['height'], settings['output']['width']])
        # Padding specifies requested output shape only. BenchmarkModel.encode
        # encodes the eight observed frames separately and pads in latent space.
        pad = video[-1:].expand(settings['output']['frames'] - len(video), -1, -1, -1)
        video = torch.cat([video, pad]).permute(1, 0, 2, 3).unsqueeze(0).cuda()
        generated = pipe.generate_vid2world(prompt=sample.prompt, input_path=video,
            guidance=sample.guidance, seed=sample.seed, negative_prompt=sample.negative_prompt,
            num_steps=sample.num_steps, num_video_frames=sample.num_output_frames,
            num_latent_conditional_frames=sample.num_input_frames, resolution=sample.resolution)
        expected = [1, 3, settings['output']['frames'], settings['output']['height'], settings['output']['width']]
        assert list(generated.shape) == expected and torch.isfinite(generated).all()
        dest.mkdir(parents=True, exist_ok=True)
        save_img_or_video((1 + generated[0]) / 2, str(dest / 'video.partial'), fps=pipe.fps)
        fmt = video_info(dest / 'video.partial.mp4')
        assert (fmt['width'],fmt['height'],fmt['frames'],fmt['fps']) == (
            settings['output']['width'],settings['output']['height'],settings['output']['frames'],pipe.fps)
        (dest / 'video.partial.mp4').replace(dest / 'video.mp4')
        write_json(receipt, dict(status='COMPLETE',method=info['method'],case=sample.name,
            protocol_sha256=info['protocol_sha256'],adapter_sha256=pipe.weight_audit.get('adapter_sha256'),
            video_sha256=sha(dest/'video.mp4'),output_format=fmt,online_text=True,online_vae=True,
            training_cache_used=False,context=pipe.model.last_context_audit,
            input_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            sampling=sample.model_dump(mode='json'),seconds=time.time()-start))
        print('COMPLETE',sample.name,flush=True)
        del generated,video
    write_json(output / 'status.json', dict(stage='COMPLETE',total=len(samples)))


if __name__ == '__main__':
    main()
