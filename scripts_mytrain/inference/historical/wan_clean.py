"""wan clean: historical inference, no RePaint or offset variants.

Run with --help for the reproducible CLI. The original core is below;
bootstrap redirects output and freezes dependencies without changing sampling.
"""
if __name__ == "__main__":
    from reproduce import launch
    launch("wan", "clean", __file__)
    raise SystemExit(0)

"""Real-model endpoint gate and serial, complete-sequence pilot worker."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import argparse
import contextlib
import json
import sys
import time
import traceback
import fcntl
from pathlib import Path
import numpy as np
import torch
import imageio.v2 as imageio
from common import ROOT, OLD, save, sha, training_scheduler
"""Scoped inferv2 extension; no shared source edits or attention replacement."""
import math
import torch
from common import training_pair_function


class ContextNoise:
    def __init__(self, pipe, scheduler, scale, patch_size, digest=None):
        if not math.isfinite(scale) or not 0 <= scale <= 1:
            raise ValueError('inference context noise scale must be finite and in [0,1]')
        self.pipe, self.scheduler, self.scale = pipe, scheduler, scale
        self.patch = tuple(patch_size)
        self.digest = digest
        self.pair, _ = training_pair_function()
        self.index = 0
        self.trace = []
        self.step_count = 0

    def prepare(self, *args, **kwargs):
        result = self.original_prepare(*args, **kwargs)
        latent, condition, mask, noise = result
        self.condition = condition.detach().clone()
        self.c = int((mask[0, 0, :, 0, 0] == 0).sum())
        assert self.c == 2 and self.patch[0] == 1
        assert torch.all(mask[:, :, :self.c] == 0) and torch.all(mask[:, :, self.c:] == 1)
        self.noise = noise[:, :, :self.c].detach().clone()
        self.latent_dtype = latent.dtype
        return result

    def before(self, module, args, kwargs):
        assert not args
        latent, timestep = kwargs['hidden_states'], kwargs['timestep']
        assert self.index == self.step_count, 'Exactly one conditional forward per sampling step required'
        sigma = self.pipe.scheduler.sigmas[self.index].to(device=latent.device, dtype=self.latent_dtype)
        # Training's lookup returns in its input timestep dtype; do not truncate
        # its FP32 table through UniPC's integer scheduler timestep storage.
        future_t = self.pipe.scheduler.timesteps[self.index].to(device=latent.device, dtype=self.scheduler.timesteps.dtype)
        ct, cs, target = self.pair(self.scheduler, future_t, sigma, self.scale, device=latent.device)
        grid = tuple(n // p for n, p in zip(latent.shape[2:], self.patch))
        space = grid[1] * grid[2]
        assert grid[0] == 13 and timestep.shape == (latent.shape[0], grid[0] * space)
        assert torch.all(timestep[:, :self.c * space] == 0)
        assert torch.all(timestep[:, self.c * space:] == future_t)
        assert torch.equal(latent[:, :, :self.c], self.condition[:, :, :self.c].to(latent))
        x, ts = latent.clone(), timestep.clone()
        context = (1 - cs) * self.condition[:, :, :self.c] + cs * self.noise
        x[:, :, :self.c] = context.to(x)
        ts[:, :self.c * space] = ct.to(ts)
        assert torch.equal(x[:, :, self.c:], latent[:, :, self.c:])
        assert torch.equal(ts[:, self.c * space:], timestep[:, self.c * space:])
        assert torch.isfinite(x).all()
        row = dict(step=self.index, future_sigma=float(sigma), future_timestep=float(future_t),
                   target_context_sigma=float(target), actual_context_sigma=float(cs), context_timestep=float(ct),
                   grid=list(grid), future_unchanged=True)
        if self.digest:
            row['context_sha256'] = self.digest(x[:, :, :self.c])['sha256']
            row['timestep_sha256'] = self.digest(ts)['sha256']
        self.trace.append(row)
        self.index += 1
        return args, dict(kwargs, hidden_states=x, timestep=ts)

    def step(self, model_output, timestep, sample, *args, **kwargs):
        assert not args and kwargs.get('return_dict') is False
        assert self.index == self.step_count + 1
        future = self.original_step(model_output[:, :, self.c:], timestep, sample[:, :, self.c:], **kwargs)[0]
        self.step_count += 1
        return (torch.cat([self.condition[:, :, :self.c], future], dim=2),)

    def __enter__(self):
        self.original_prepare = self.pipe.prepare_latents
        self.original_step = self.pipe.scheduler.step
        self.pipe.prepare_latents = self.prepare
        self.pipe.scheduler.step = self.step
        self.hook = self.pipe.transformer.register_forward_pre_hook(self.before, with_kwargs=True)
        return self

    def __exit__(self, *exc):
        self.hook.remove()
        self.pipe.prepare_latents = self.original_prepare
        self.pipe.scheduler.step = self.original_step

sys.path.insert(0, str(OLD))
import run as replay


TEST70_CACHE_ROOT = Path(os.environ.get(
    'PHYSRVG_TEST70_CACHE_ROOT',
    '/data/gaoya/AAA_test_video/physv_v2v_0819/test70_context_cache_20260929',
))
TEST70_CACHE_MANIFEST = TEST70_CACHE_ROOT / 'manifest.json'


def difference(a, b):
    a, b = a.float(), b.float()
    d = b - a
    return dict(max_abs=float(d.abs().max()), relative_l2=float(d.norm() / a.norm().clamp_min(1e-12)), rms=float(d.square().mean().sqrt()))


def check(a, b, tolerance):
    value = difference(a, b)
    assert all(np.isfinite(x) for x in value.values()), value
    assert value['max_abs'] <= tolerance['max_abs'] and value['relative_l2'] <= tolerance['relative_l2'], value
    return value


class Experiment(replay.Experiment):
    def _refresh_test70_cache(self):
        self.cache_enabled = False
        self.cache_manifest = None
        self.context_cache = {}
        if os.environ.get('PHYSRVG_DISABLE_TEST70_CACHE') == '1' or not TEST70_CACHE_MANIFEST.is_file():
            return
        manifest = json.loads(TEST70_CACHE_MANIFEST.read_text())
        assert manifest['status'] == 'COMPLETE'
        assert manifest['cache_contract'] == 'normalized_context_latent_v1'
        assert manifest['model_id'] == str(self.cfg['base']['model_id'])
        self.cache_manifest = manifest
        self.cache_enabled = True

    def _load_cached_prompt(self, case):
        from safetensors.torch import load_file
        entry = self.cache_manifest['cases'][case]['prompt']
        path = TEST70_CACHE_ROOT / entry['file']
        assert sha(path) == entry['file_sha256']
        tensor = load_file(str(path), device='cpu')['prompt_embeds']
        assert list(tensor.shape) == entry['shape'] and str(tensor.dtype) == entry['dtype']
        return tensor.to('cuda:0')

    def _load_cached_context(self, case):
        from safetensors.torch import load_file
        if case not in self.context_cache:
            entry = self.cache_manifest['cases'][case]['context_latent']
            path = TEST70_CACHE_ROOT / entry['file']
            assert sha(path) == entry['file_sha256']
            tensor = load_file(str(path), device='cpu')['context_latent']
            assert list(tensor.shape) == entry['shape'] and str(tensor.dtype) == entry['dtype']
            self.context_cache[case] = tensor.contiguous()
        return self.context_cache[case]

    def prepare_case(self, case):
        from infer_full_sa_lora_json_list import load_context_video
        if case in self.prompts:
            return
        if self.cache_enabled:
            emb = self._load_cached_prompt(case)
        else:
            with torch.inference_mode():
                emb, _ = self.pipe.encode_prompt(
                    prompt=self.cfg['cases'][case]['prompt'], negative_prompt='',
                    do_classifier_free_guidance=False, device=torch.device('cuda:0'),
                )
        self.prompts[case] = emb.detach()
        self.inputs[case] = load_context_video(
            Path(self.cfg['cases'][case]['context_video']), 512, 896, max_frames=8,
        )
        save(replay.ROOT / 'replay_states' / f'{case}_prompt.json', dict(
            prompt=self.cfg['cases'][case]['prompt'], embedding=replay.digest(emb),
            context_video_sha256=sha(self.cfg['cases'][case]['context_video']),
            cache_enabled=self.cache_enabled,
        ))

    def load_adapters(self):
        from peft import PeftModel, LoraConfig, get_peft_model_state_dict
        from safetensors.torch import load_file
        matches = {}
        for method in self.active_methods:
            adapter = 'analysis_' + method
            path = self.cfg['models'][method]['source_checkpoint']
            expected = load_file(str(Path(path) / 'adapter_model.safetensors'))
            assert all('.lora_A.' in k or '.lora_B.' in k for k in expected)
            if self.peft is None:
                self.peft = PeftModel.from_pretrained(self.raw, path, adapter_name=adapter, is_trainable=False)
            else:
                self.peft.add_adapter(adapter, LoraConfig.from_pretrained(path))
                for module in self.peft.modules():
                    for attr in ('lora_A', 'lora_B'):
                        bank = getattr(module, attr, None)
                        if bank is not None and adapter in bank:
                            bank[adapter].float()
                self.peft.load_adapter(path, adapter_name=adapter, is_trainable=False)
            actual = get_peft_model_state_dict(self.peft, adapter_name=adapter)
            assert set(actual) == set(expected)
            assert all(torch.equal(actual[k].detach().cpu().float(), v.float()) for k, v in expected.items())
            targets = {n: dict(r=m.r[adapter], alpha=m.lora_alpha[adapter], scaling=m.scaling[adapter], merged=bool(m.merged))
                       for n, m in self.peft.named_modules() if hasattr(m, 'lora_A') and adapter in m.lora_A}
            assert targets and not any(t['merged'] for t in targets.values())
            matches[method] = dict(tensors_exact=True, keys_exact=True, targets=targets, extra_non_lora=False)
        self.pipe.transformer = self.peft
        self.peft.eval()
        self.raw = self.peft.get_base_model()
        save(replay.ROOT / 'adapter_loading.json', matches)


def sample(exp, task, train, legacy=None, capture_first=False):
    case, seed = task['case'], task['seed']
    exp.prepare_case(case)
    scale = task['inference_context_noise_scale']
    adapter = ContextNoise(exp.pipe, train, scale, exp.raw.config.patch_size, replay.digest) if legacy is None else None
    inputs, actual = {}, []
    condition = None
    original = exp.pipe.prepare_latents

    def prepare(*a, **kw):
        nonlocal condition
        if exp.cache_enabled:
            kw['precomputed_video_latent'] = exp._load_cached_context(case).to('cuda:0')
        result = original(*a, **kw)
        condition = result[1].detach().cpu()
        inputs.update(initial_latents=replay.digest(result[0]), condition=replay.digest(result[1]),
                      condition_mask=replay.digest(result[2]), noise=replay.digest(result[3]), text=replay.digest(exp.prompts[case]))
        return result

    def capture(module, args, kw):
        actual.append(dict(step=len(actual), context=replay.digest(kw['hidden_states'][:, :, :2]), timestep=replay.digest(kw['timestep'])))
        if capture_first and len(actual) == 1:
            inputs['first_forward'] = replay.cpu(kw)

    exp.pipe.prepare_latents = prepare
    torch.manual_seed(seed)
    try:
        with exp.method(task['method']), torch.inference_mode(), (adapter or contextlib.nullcontext()):
            # Registered after intervention: observes the inputs actually consumed by DiT.
            hook = exp.pipe.transformer.register_forward_pre_hook(capture, with_kwargs=True)
            try:
                out = exp.pipe(video=exp.inputs[case], device=torch.device('cuda:0'), prompt_embeds=exp.prompts[case],
                               negative_prompt=None, height=512, width=896, num_frames=49, num_inference_steps=40,
                               guidance_scale=5., do_cfg=False, generator=torch.Generator(device='cuda').manual_seed(seed),
                               output_type='latent', dynamic_condition_mask=False, context_noise_mode=legacy or 'clean').frames.detach().cpu()
            finally:
                hook.remove()
    finally:
        exp.pipe.prepare_latents = original
    assert len(actual) == 40 and torch.isfinite(out).all()
    assert torch.equal(out[:, :, :2], condition[:, :, :2].to(out))
    if adapter:
        assert adapter.index == adapter.step_count == 40
    return out, inputs, actual, adapter.trace if adapter else []


def validate(exp, train, cfg):
    dest = ROOT / 'gate'
    dest.mkdir(exist_ok=True)
    receipt = dict(status='RUNNING', started=time.time(), checks={}, config_sha256=sha(ROOT / 'config.json'))
    save(dest / 'validation.json', receipt)
    task = dict(case='v2v_pendulum_l110', seed=42, method='C', inference_context_noise_scale=0.)
    clean, paired, inputs, _ = sample(exp, task, train, legacy='clean', capture_first=True)
    state = paired.pop('first_forward')
    torch.save(state, dest / 'first_forward.pt')
    reference_inputs = replay.cuda(state)

    def forward(name):
        with exp.method(name), torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            return exp.pipe.transformer(**reference_inputs)[0].detach().cpu()

    base = forward('Base')
    for method in cfg['methods']:
        on = forward(method)
        off = forward('Base')
        again = forward(method)
        receipt['checks'][method + '_restore'] = dict(off=check(base, off, cfg['endpoint_tolerance']),
                                                      on=check(on, again, cfg['endpoint_tolerance']))
    for scale, legacy, reference in [(0., 'clean', clean), (1., 'fixed_noise_match_timestep', None)]:
        task['inference_context_noise_scale'] = scale
        if reference is None:
            reference, source_inputs, source_trace, _ = sample(exp, task, train, legacy=legacy)
        else:
            source_inputs, source_trace = paired, inputs
        changed, new_inputs, new_trace, trace = sample(exp, task, train)
        assert new_inputs == source_inputs == paired
        assert new_trace == source_trace, 'Endpoint context/timestep input hashes differ'
        receipt['checks'][f'endpoint_{scale:g}'] = check(reference, changed, cfg['endpoint_tolerance'])
        torch.save(changed, dest / f'endpoint_{scale:g}.pt')
        save(dest / f'endpoint_{scale:g}_trace.json', trace)
        save(dest / 'validation.json', receipt)
        print('ENDPOINT_PASS', scale, receipt['checks'][f'endpoint_{scale:g}'], flush=True)
    repeat, _, _, _ = sample(exp, task, train)
    receipt['checks']['normal_repeat_scale1'] = check(changed, repeat, cfg['endpoint_tolerance'])
    receipt.update(status='PASS', completed=time.time(), no_decoding=True,
                   full_samples=5, all_6_adapter_restore=True, shared_attention_backend_unchanged=True)
    save(dest / 'validation.json', receipt)


def generate(exp, task, train, cfg):
    dest = ROOT / 'results' / task['id']
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'result.json').exists():
        receipt = json.loads((dest / 'result.json').read_text())
        assert receipt['task'] == task and receipt['config_sha256'] == sha(ROOT / 'config.json')
        assert sha(dest / 'video.mp4') == receipt['video_sha256'] and sha(dest / 'latent.pt') == receipt['latent_sha256']
        from publish import install_result
        install_result(exp.cfg, task)
        return
    start = time.time()
    torch.cuda.reset_peak_memory_stats()
    latent, paired, actual, trace = sample(exp, task, train)
    baseline = ROOT / 'paired_inputs' / f'{task["case"]}__seed{task["seed"]}.json'
    baseline.parent.mkdir(exist_ok=True)
    with baseline.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if baseline.exists():
            assert json.loads(baseline.read_text()) == paired, 'Initial noise/condition/text pairing mismatch'
        else:
            save(baseline, paired)
    torch.save(latent, dest / 'latent.pt')
    exp.pipe.transformer.to('cpu')
    torch.cuda.empty_cache()
    with torch.inference_mode():
        z = latent.to('cuda:0', dtype=exp.pipe.vae_pipeline_dtype)
        mean = torch.tensor(exp.pipe.vae.config.latents_mean).view(1, exp.pipe.vae.config.z_dim, 1, 1, 1).to(z)
        inv_std = 1. / torch.tensor(exp.pipe.vae.config.latents_std).view(1, exp.pipe.vae.config.z_dim, 1, 1, 1).to(z)
        decoded = exp.pipe.vae.decode(z / inv_std + mean, return_dict=False)[0]
        frames = exp.pipe.video_processor.postprocess_video(decoded, output_type='np')[0]
    assert frames.shape == (49, 512, 896, 3) and np.isfinite(frames).all()
    with imageio.get_writer(str(dest / 'video.partial.mp4'), fps=cfg['export_fps'], codec='libx264', quality=8,
                            macro_block_size=None, ffmpeg_params=['-threads', '1']) as writer:
        for frame in frames:
            writer.append_data((np.clip(frame, 0, 1) * 255).astype(np.uint8))
    (dest / 'video.partial.mp4').replace(dest / 'video.mp4')
    save(dest / 'result.json', dict(status='COMPLETE', task=task, paired_input=paired, trace=trace, actual_forward=actual,
         config_sha256=sha(ROOT / 'config.json'), video_sha256=sha(dest / 'video.mp4'), latent_sha256=sha(dest / 'latent.pt'),
         seconds=time.time()-start, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
         source_checkpoint=exp.cfg['models'][task['method']]['source_checkpoint'], clean_context_restored=True,
         dtype=str(next(exp.raw.parameters()).dtype), attention_backend=exp.cfg['runtime']['backend'],
         test70_cache_enabled=bool(exp.cache_enabled),
         test70_cache_manifest=str(TEST70_CACHE_MANIFEST) if exp.cache_enabled else None))
    from publish import install_result
    install_result(exp.cfg, task)
    del z, decoded, frames
    torch.cuda.empty_cache()
    exp.pipe.transformer.to('cuda:0')
    print('COMPLETE', task['id'], round(time.time()-start, 1), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True, choices=[0, 1, 2, 3, 5, 6])
    parser.add_argument('--method', required=True, choices=['C', 'N25', 'N50', 'N75', 'A', 'B'])
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    args = parser.parse_args()
    assert 0 <= args.shard_index < args.shard_count
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(args.gpu)
    resolved = json.loads((ROOT / 'resolved_config.json').read_text())
    cfg = resolved['config']
    assert sha(ROOT / 'config.json') == resolved['config_sha256']
    assert json.loads((ROOT / 'validation_cpu.json').read_text())['status'] == 'PASS'
    for path, expected in resolved['source_hashes'].items():
        if Path(path).suffix == '.py':
            assert sha(path) == expected, f'Shared source changed: {path}'
    folder = ROOT / f'workers/{args.method}'
    if args.shard_count > 1:
        folder = folder / f'shard{args.shard_index}of{args.shard_count}'
    save(folder / 'resolved_config.json', resolved)
    (folder / 'replay_states').mkdir(exist_ok=True)
    replay.ROOT = folder
    exp = Experiment()
    exp._refresh_test70_cache()
    if exp.cache_enabled:
        exp.cfg.setdefault('runtime', {})['cache'] = 'test70_prompt_and_normalized_context_latent'
        exp.cfg['runtime']['cache_manifest'] = str(TEST70_CACHE_MANIFEST)
        exp.cfg['runtime']['text_encoder_forward_per_task'] = False
        save(ROOT / 'runtime_cache.json', {
            'enabled': True, 'manifest': str(TEST70_CACHE_MANIFEST),
            'prompt_embeddings': 'precomputed exact encode_prompt output',
            'context_latent': 'precomputed normalized argmax VAE latent',
            'text_encoder_forward_per_task': False,
        })
    exp.active_methods = [args.method]
    exp.load_adapters()
    exp.pipe.set_progress_bar_config(disable=True)
    train = training_scheduler(resolved, 'cuda:0')
    tasks = [t for t in resolved['tasks'] if t['method'] == args.method]
    # Keep all five noise variants of a case together, with disjoint ownership.
    cases = list(dict.fromkeys(t['case'] for t in tasks))
    selected = set(cases[args.shard_index::args.shard_count])
    tasks = [t for t in tasks if t['case'] in selected]
    previous_case = None
    for index, task in enumerate(tasks):
        if previous_case is not None and previous_case != task['case']:
            exp.prompts.clear()
            exp.inputs.clear()
        previous_case = task['case']
        save(folder / 'status.json', dict(status='RUNNING', gpu=args.gpu, completed=index, total=len(tasks), current=task, time=time.time()))
        generate(exp, task, train, cfg)
    save(folder / 'status.json', dict(status='COMPLETE', gpu=args.gpu, completed=len(tasks), total=len(tasks), time=time.time()))


