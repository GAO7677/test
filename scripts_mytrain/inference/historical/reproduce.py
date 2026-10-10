"""Read-only historical preflight and isolated replay of one saved result.

No benchmark publication, training, RePaint, or context-offset experiments.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BENCH = Path('/data/gaoya/AAA_test_video/Context_Video_bench')
WAN = BENCH / 'Test70/noise/wan'
NANO = BENCH / 'Test70_v2/noise/nano_20cases'
TRIPLET = BENCH / 'Shared/supplement/gap038_triplet'
PREDICT_REPO = Path('/home/gaoya/Code_Video/Code_MODEL/cosmos-predict2.5')
PYTHONS = {
    'wan': '/data/gaoya/agent-data/envs/physrvg-full-sa/bin/python',
    'nano': '/data/gaoya/agent-data/envs/cosmos_nano_acn75/bin/python',
    'predict25': '/data/gaoya/agent-data/envs/cosmos_predict25_official/bin/python',
}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def verify(path, expected):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f'SHA256 mismatch: {path}: {actual} != {expected}')
    return actual


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def default_reference(backbone, arm):
    if backbone == 'wan':
        scale = '0' if arm == 'clean' else '0.75'
        return WAN / f'results/v2v_gap_038__N75__seed42__noise{scale}/result.json'
    if backbone == 'nano':
        variant = 'clean' if arm == 'clean' else 'matched'
        return NANO / f'step1500/{variant}/results/v2v_gap_038/result.json'
    variant = 'clean' if arm == 'clean' else 'matched'
    return TRIPLET / f'predict25/N75/step1500/{variant}/results/v2v_gap_038/result.json'


def find_nano_config(reference):
    # Configs beside result.json are evaluation receipts, not model configs.
    for parent in list(reference.parents)[2:7]:
        candidates = [parent / 'config.json', parent / 'launch_config.json']
        candidates += sorted(parent.glob('launch_gpu*.json'))
        for path in candidates:
            if path.is_file():
                obj = read(path)
                if {'paths', 'lora', 'data', 'evaluation', 'noise'} <= obj.keys():
                    return path, obj
    raise ValueError('Historical Nano launch config not found; supply --config')


def plan(backbone, arm, args):
    ref = (args.reference or default_reference(backbone, arm)).absolute()
    receipt = read(ref)
    verify(ref.with_name('video.mp4'), receipt['video_sha256'])
    result = dict(backbone=backbone, arm=arm, reference=str(ref),
                  reference_sha256=sha(ref), reference_video_sha256=receipt['video_sha256'],
                  python=PYTHONS[backbone], gpu=args.gpu, output=str(args.output) if args.output else None)
    if backbone == 'wan':
        resolved = read(HERE / 'frozen/wan/resolved_config.json')
        task = receipt['task']
        if task not in resolved['tasks']:
            raise ValueError('This entry supports the pinned Test70 Wan campaign; unknown task')
        if (task['inference_context_noise_scale'] == 0) != (arm == 'clean'):
            raise ValueError('Clean/noisy entry does not match reference')
        adapter = Path(receipt['source_checkpoint'])
        for name in ['adapter_model.safetensors', 'adapter_config.json']:
            verify(adapter / name, resolved['source_hashes'][str(adapter / name)])
        # Current Wan core is identical to the campaign snapshot. Fail on drift.
        for path, expected in resolved['source_hashes'].items():
            if path.endswith('.py') and '/PhysRVG-main/' in path:
                verify(path, expected)
        video = resolved['cases'][task['case']]['context_video']
        verify(video, resolved['source_hashes'][video])
        if receipt.get('test70_cache_enabled', True):
            cache = Path('/data/gaoya/AAA_test_video/physv_v2v_0819/test70_context_cache_20260929/manifest.json')
            if not cache.is_file():
                raise FileNotFoundError('Historical Wan context/prompt cache required: ' + str(cache))
            result['cache_manifest'] = str(cache)
        result.update(task=task, case=task['case'], adapter=str(adapter), profile='test70_fixed_initial_noise')
    elif backbone == 'nano':
        c = receipt['provenance']['config']
        variant = c.get('context_variant', 'early' if 'inference_noise_scale' in c else 'clean')
        if variant not in ['clean', 'early', 'matched', 'shared075', 'scaled', 'clean_noisy']:
            raise ValueError('Unknown historical Nano variant: ' + variant)
        scale = c.get('context_scale', c.get('inference_noise_scale', .75 if variant != 'clean' else 0.))
        is_clean = variant == 'clean' or variant == 'early' and scale == 0
        if is_clean != (arm == 'clean'):
            raise ValueError('Clean/noisy entry does not match reference; clean_noisy is a noisy-role arm')
        config_path, cfg = (args.config, read(args.config)) if args.config else find_nano_config(ref)
        adapter = Path(c['adapter'])
        verify(adapter / 'adapters.safetensors', c['adapter_sha256'])
        verify(adapter / 'adapter_config.json', c['adapter_config_sha256'])
        for file, key in [('frozen/nano/run_matched.py', 'matched_reader_sha256'),
                          ('frozen/wan/infer_full_sa_lora_json_list.py', 'wan_reader_sha256'),
                          ('frozen/nano/negative_prompt.json', 'negative_prompt_sha256')]:
            verify(HERE / file, c[key])
        mode = read(adapter / 'complete.json')['config']['mode']
        training_contract = read(adapter / 'complete.json')['config']['experiment']
        if cfg['noise'] != training_contract['noise'] or cfg['lora'] != training_contract['lora']:
            raise ValueError('Nano launch noise/LoRA contract differs from checkpoint')
        source_sha = receipt['provenance']['context_sha256']
        # Locate the exact case in archived manifests, never identify it by name alone.
        candidates = []
        for parent in list(ref.parents)[2:7]:
            candidates += [parent / 'manifest.json', parent / 'clean_manifest.json']
        candidates += [HERE / 'references/nano_test70v2_manifest.json', HERE / 'frozen/predict25/manifest.json']
        if c['benchmark'] == 'test89':
            candidates += [Path('/data/gaoya/agent-data/outputs_v1/geometry_sweep_bench/20260929_test100_clean/resolved_config.json')]
        case = ref.parent.name
        row = None
        for path in candidates:
            if path.is_file():
                candidate = read(path).get('cases', {}).get(case)
                if candidate and sha(candidate['context_video']) == source_sha:
                    row = dict(candidate)
                    break
        if row is None:
            raise ValueError('Cannot identify exact historical input; missing case manifest')
        row.update(prompt=receipt['provenance']['prompt'], context_video_sha256=source_sha,
                   gt_video_sha256=receipt['provenance']['gt_sha256'])
        verify(row['gt_video'], row['gt_video_sha256'])
        if variant == 'early':
            profile = 'early'
        elif variant == 'scaled':
            profile = 'ac_sweep' if mode in ('A', 'C') else 'n75_100' if scale == 1 else 'n75_sweep'
        else:
            profile = 'v2'
        # The runner keeps historical precision, scheduler, CFG and RNG order.
        for key, old_key in [('steps', 'num_inference_steps'), ('seed', 'seed'), ('guidance_scale', 'guidance_scale'), ('flow_shift', 'flow_shift')]:
            if cfg['evaluation'][key] != c[old_key]:
                raise ValueError(f'Launch config differs from receipt: {key}')
        result.update(case=case, adapter=str(adapter), mode=mode, profile=profile,
                      variant=variant, scale=scale, config_source=str(config_path),
                      config=cfg, row=row, sampling=c, fps=c['fps'])
    else:
        root = ref.parents[2]
        saved = read(root / 'inference_config.json')
        audit_path = root / 'triplet_audit.json'
        audit = read(audit_path) if audit_path.exists() else None
        variant = audit['variant'] if audit else 'clean'
        if (variant == 'clean') != (arm == 'clean'):
            raise ValueError('Clean/noisy entry does not match reference')
        if not audit and arm != 'clean':
            raise ValueError('Missing historical denoise audit')
        adapter = Path(saved['adapter']) if saved['adapter'] else None
        if adapter:
            verify(adapter / 'adapter.safetensors', receipt['adapter_sha256'])
            metadata = read(adapter / 'metadata.json')
        else:
            metadata = {}
        # infer_bench selects method/step from explicit saved checkpoint data.
        mode = audit['mode'] if audit else ('baseline' if adapter is None else saved['method'].split('-')[-3])
        if adapter:
            mode = next(m for m in ['N75', 'C', 'A'] if f'-{m}-step' in saved['method'])
            step = int(adapter.name.removeprefix('step'))
        else:
            step = None
        if audit:
            cache = TRIPLET / 'predict_shared_context.pt'
            if not cache.is_file():
                raise FileNotFoundError('Exact triplet reproduction requires existing canonical context cache')
            if receipt['case'] != 'v2v_gap_038':
                raise ValueError('The historical triplet cache is specific to gap038')
            result['canonical_cache'] = str(cache)
        settings = saved['settings']
        manifest_path = Path(settings['benchmarks'][saved['benchmark']]['manifest'])
        if not manifest_path.exists() and audit:
            # The original 70-case directory was moved. The triplet archived
            # the exact gap038 RGB hashes; retain that evidence, not a new case.
            manifest_path = HERE / 'frozen/predict25/manifest.json'
            settings['benchmarks'][saved['benchmark']]['manifest'] = str(manifest_path)
            settings['benchmarks'][saved['benchmark']]['count'] = 1
        manifest = read(manifest_path)
        row = manifest['cases'][receipt['case']]
        if row['prompt'] != receipt['sampling']['prompt'] or row['context_video'] != receipt['sampling']['input_path']:
            raise ValueError('Archived Predict manifest differs from saved sampling input')
        if row.get('context_video_sha256'):
            verify(row['context_video'], row['context_video_sha256'])
        result.update(case=receipt['case'], adapter=str(adapter) if adapter else None,
                      mode=mode, step=step, profile='triplet' if audit else 'native', variant=variant,
                      settings=settings, training=saved['training_settings'], benchmark=saved['benchmark'],
                      implementation_sha256=saved['implementation_sha256'], sampling=receipt['sampling'])
    return result


def setup_nano(p, output, entry):
    cfg = p['config']
    cfg['runtime']['evaluation_gpu'] = p['gpu']
    cfg['paths']['output'] = str(output)
    write(output / 'launch_config.json', cfg)
    write(output / 'manifest.json', dict(cases={p['case']: p['row']}, fps=p['fps']))
    profile = p['profile']
    folder = HERE / 'frozen/nano' / profile
    sys.argv = [str(entry), '--config', str(output / 'launch_config.json'),
                '--adapter', p['adapter'], '--benchmark', p['sampling']['benchmark'],
                '--name', p['mode'], '--case-manifest', str(output / 'manifest.json'),
                '--output-dir', str(output)]
    os.environ['CONTEXT_VARIANT'] = p['variant']
    os.environ['CONTEXT_SCALE'] = str(p['scale'])
    os.environ['PYTORCH_ALLOC_CONF'] = cfg['runtime']['cuda_allocator']
    os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)
    load('settings', folder / 'settings.py')
    load('common', folder / 'common.py')
    if profile == 'early':
        load('noise_hook', folder / 'campaign/noise_hook.py')
        core = load('historical_nano_core', folder / 'campaign/evaluate_noise.py')
        sys.argv += ['--noise-scale', str(p['scale'])]
    else:
        core = load('historical_nano_core', entry)
        # Select the exact saved noise class; no endpoint policy is generalized.
        hooks = load('historical_nano_hooks', folder / 'hooks.py')
        if profile == 'v2':
            core.NanoContext = lambda pipe, variant, mode, cfg, seed, scale: hooks.NanoContext(pipe, variant, mode, cfg, seed)
        else:
            core.NanoContext = hooks.NanoContext
    core.MATCHED = HERE / 'frozen/nano/run_matched.py'
    core.WAN_READER = HERE / 'frozen/wan/infer_full_sa_lora_json_list.py'
    core.NEGATIVE = HERE / 'frozen/nano/negative_prompt.json'
    core.main()
    return output / 'results' / p['case'] / 'result.json'


def setup_wan(p, output, entry):
    folder = HERE / 'frozen/wan'
    sys.path[:0] = [str(REPO), str(REPO / 'scripts_mytrain/inference')]
    common = load('common', folder / 'common.py')
    common.ROOT = output
    common.OLD = folder / 'replay'
    audit = load('audit', folder / 'replay/audit.py')
    audit.ROOT = output
    load('observer', folder / 'replay/observer.py')
    replay = load('run', folder / 'replay/run.py')
    replay.ROOT = output
    core = load('historical_wan_core', entry)
    core.ROOT = output
    # Historical worker published completed videos. Replay writes only new output.
    publish = types.ModuleType('publish')
    publish.install_result = lambda *a, **kw: None
    sys.modules['publish'] = publish
    resolved = read(folder / 'resolved_config.json')
    write(output / 'resolved_config.json', resolved)
    (output / 'replay_states').mkdir()
    (output / 'config.json').write_bytes((folder / 'config.json').read_bytes())
    exp = core.Experiment()
    if 'cache_manifest' in p:
        exp._refresh_test70_cache()
        if not exp.cache_enabled:
            raise RuntimeError('Historical context cache not enabled')
    else:
        exp.cache_enabled = False
    exp.active_methods = [p['task']['method']]
    exp.load_adapters()
    exp.pipe.set_progress_bar_config(disable=True)
    train = common.training_scheduler(resolved, 'cuda:0')
    core.generate(exp, p['task'], train, resolved['config'])
    return output / 'results' / p['task']['id'] / 'result.json'


def setup_predict(p, output, entry):
    folder = HERE / 'frozen/predict25'
    os.chdir(PREDICT_REPO)
    sys.path[:0] = [str(PREDICT_REPO), str(PREDICT_REPO / 'packages/cosmos-oss')]
    import projects
    package = types.ModuleType('projects.generic_acn75')
    package.__path__ = [str(folder)]
    sys.modules['projects.generic_acn75'] = package
    projects.generic_acn75 = package
    write(output / 'settings.json', p['settings'])
    if p['training']:
        write(output / 'training_settings.json', p['training'])
    sys.argv = [str(entry), '--settings', str(output / 'settings.json'),
                '--benchmark', p['benchmark'], '--method', p['mode'], '--case', p['case'], '--output', str(output)]
    if p['adapter']:
        sys.argv += ['--step', str(p['step']), '--adapter', p['adapter'], '--training-settings', str(output / 'training_settings.json')]
    core = load('historical_predict_core', entry)
    core.HERE = folder  # Historical native receipt hash list uses original filenames.
    if p['profile'] == 'triplet':
        core.install_predict(p['variant'], p['mode'], 20261007, str(output / 'triplet_audit.json'))
    core.main()
    return output / 'results' / p['case'] / 'result.json'


def compare(reference, generated):
    old, new = read(reference), read(generated)
    report = dict(reference=str(reference), generated=str(generated),
                  video_bytes_equal=old['video_sha256'] == new['video_sha256'])
    for key in ['paired_input', 'actual_forward', 'trace', 'latent_sha256', 'input_rgb_sha256', 'noise_audit']:
        if key in old and key in new:
            report[key + '_equal'] = old[key] == new[key]
    a, b = old.get('context_audit', {}), new.get('context_audit', {})
    for key in ['rgb8_sha256', 'encode_calls']:
        if key in a and key in b:
            report[key + '_equal'] = a[key] == b[key]
    for key in ['clean_context_sha256', 'context_noise_sha256', 'future_initial_sha256', 'trace']:
        x, y = a.get('triplet', {}), b.get('triplet', {})
        if key in x and key in y:
            report[key + '_equal'] = x[key] == y[key]
    old_audit = Path(reference).parents[2] / 'triplet_audit.json'
    new_audit = Path(generated).parents[2] / 'triplet_audit.json'
    if old_audit.exists():
        if not new_audit.exists():
            report['triplet_audit_equal'] = False
        else:
            report['triplet_audit_equal'] = read(old_audit) == read(new_audit)
    report['status'] = 'BYTE_EXACT' if all(v for k, v in report.items() if k.endswith('_equal')) else 'DIFFERENT_REQUIRES_REVIEW'
    return report


def launch(backbone, arm, entry):
    parser = argparse.ArgumentParser(description=f'Reproduce one historical {backbone} {arm} result, using its saved receipt.')
    parser.add_argument('--reference', type=Path, help='Historical result.json; default: N75 gap038 (Wan step1000, Cosmos step1500)')
    parser.add_argument('--output', type=Path, help='New empty output directory under /data/gaoya')
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--config', type=Path, help='Nano original full launch config if not found beside the reference')
    parser.add_argument('--check-only', action='store_true', help='Read-only source/input/adapter checks; no GPU, no writes')
    args = parser.parse_args()
    if args.gpu not in [0, 1, 2, 3, 5, 6]:
        parser.error('Use one GPU supported by the historical evaluator: 0,1,2,3,5,6')
    for row in read(HERE / 'source_manifest.json'):
        verify(HERE / row['file'], row['sha256'])
    runtime = read(HERE / 'runtime_manifest.json')[backbone]
    for row in runtime['files']:
        verify(row['path'], row['sha256'])
    p = plan(backbone, arm, args)
    if args.check_only:
        print(json.dumps(dict(status='PREFLIGHT_PASS_GPU_NOT_RUN', **p), indent=2, ensure_ascii=False))
        return
    if args.output is None:
        parser.error('--output is required for generation')
    output = args.output.resolve()
    if not output.is_relative_to('/data/gaoya'):
        parser.error('Generated outputs must be under /data/gaoya')
    if output.exists() and any(output.iterdir()):
        parser.error('Output must be new or empty; historical outputs cannot be overwritten or reused')
    if Path(sys.prefix).resolve() != Path(PYTHONS[backbone]).parent.parent.resolve():
        parser.error('Use the pinned Python environment: ' + PYTHONS[backbone])
    import importlib.metadata
    for package, version in runtime['packages'].items():
        if importlib.metadata.version(package) != version:
            parser.error(f'Package version changed: {package}; expected {version}')
    output.mkdir(parents=True, exist_ok=True)
    write(output / 'reproduction_plan.json', p)
    write(output / 'reproduction_sources.json', dict(
        entry=dict(path=str(entry), sha256=sha(entry)), launcher_sha256=sha(__file__),
        frozen_sources=read(HERE / 'source_manifest.json'), runtime=runtime))
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS']:
        os.environ[key] = '1' if backbone == 'wan' else '4'
    generated = {'wan': setup_wan, 'nano': setup_nano, 'predict25': setup_predict}[backbone](p, output, Path(entry).resolve())
    report = compare(Path(p['reference']), generated)
    write(output / 'reproduction_comparison.json', report)
    print(json.dumps(report, indent=2))
    if report['status'] != 'BYTE_EXACT':
        raise SystemExit(2)
