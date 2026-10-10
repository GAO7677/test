"""Exact sigma/time alignment experiment on the pinned historical noisy demos.

Only the context noise level and context timestep change. Source substitutions
are required to match exactly and the executed source is archived per run.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys

HIST = Path(__file__).resolve().parents[1] / 'historical'
spec = importlib.util.spec_from_file_location('historical_reproduce', HIST / 'reproduce.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Source contract changed: ' + old)
    return source.replace(old, new, 1)


def collect_trace(obj):
    if isinstance(obj, dict):
        if isinstance(obj.get('trace'), list):
            return obj['trace']
        for value in obj.values():
            found = collect_trace(value)
            if found is not None:
                return found
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backbone', required=True, choices=list(r.PYTHONS))
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.reference = None
    args.config = None
    output = args.output.resolve()
    if not output.is_relative_to('/data/gaoya') or (output.exists() and any(output.iterdir())):
        parser.error('Use a new or empty output directory under /data/gaoya')
    if Path(sys.prefix).resolve() != Path(r.PYTHONS[args.backbone]).parent.parent.resolve():
        parser.error('Required Python: ' + r.PYTHONS[args.backbone])
    for row in r.read(HIST / 'source_manifest.json'):
        r.verify(HIST / row['file'], row['sha256'])
    runtime = r.read(HIST / 'runtime_manifest.json')[args.backbone]
    for row in runtime['files']:
        r.verify(row['path'], row['sha256'])
    for package, version in runtime['packages'].items():
        assert importlib.metadata.version(package) == version, package
    p = r.plan(args.backbone, 'noisy', args)
    p['experiment'] = dict(name='equal_sigma', context_sigma='exact native future sigma; no training-grid lookup',
                           context_timestep='exact native future timestep',
                           unchanged=['checkpoint', 'input', 'future seed', 'context noise seed/source', 'mask', 'CFG', 'native scheduler'])
    if args.backbone == 'wan':
        p['task']['inference_context_noise_scale'] = 1.0
        p['task']['id'] = p['case'] + '__N75__seed42__equal_sigma'
    output.mkdir(parents=True, exist_ok=True)
    r.write(output / 'experiment_plan.json', p)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS']:
        os.environ[key] = '1' if args.backbone == 'wan' else '4'
    original_load = r.load
    patches = []

    def load(name, path):
        if name not in ['historical_nano_hooks', 'historical_predict_core']:
            return original_load(name, path)
        source = Path(path).read_text()
        original_sha = hashlib.sha256(source.encode()).hexdigest()
        source = replace_once(source,
            "return grid[(grid - .75 * future).abs().argmin()].reshape_as(future)",
            "return future.clone()  # exact equality; no grid quantization")
        if name == 'historical_nano_hooks':
            source = replace_once(source, 'tc=sc*self.n', 'tc=tf.clone()')
            source = replace_once(source, 'context_sha256=digest(context),context_role=',
                'future_timestep=float(tf), context_sha256=digest(context),context_role=')
            source = replace_once(source, 'context=((1-sc)*self.clean.float()+sc*self.noise).to(x)',
                'assert torch.equal(sc, sf) and torch.equal(tc, tf)\n        context=((1-sc)*self.clean.float()+sc*self.noise).to(x)')
        else:
            source = replace_once(source, 'tc=sc*self.rectified_flow.num_train_timesteps', 'tc=tf.clone()')
            source = replace_once(source, 'context_sha256=digest(ctx),mask=',
                'future_timestep=float(tf), context_sha256=digest(ctx),mask=')
            source = replace_once(source, 'ctx=((1-sc)*clean.float()+sc*state[\'noise\']).to(x)',
                "assert torch.equal(sc, sf) and torch.equal(tc, tf)\n            ctx=((1-sc)*clean.float()+sc*state['noise']).to(x)")
        archived = output / 'executed_sources' / (name + '.py')
        archived.parent.mkdir(exist_ok=True)
        archived.write_text(source)
        patches.append(dict(source=str(path), original_sha256=original_sha,
                            executed_source=str(archived), executed_sha256=r.sha(archived)))
        return original_load(name, archived)

    r.load = load
    entry = HIST / (args.backbone + '_noisy.py')
    result = {'wan': r.setup_wan, 'nano': r.setup_nano, 'predict25': r.setup_predict}[args.backbone](p, output, entry)
    receipt = r.read(result)
    audit = r.read(output / 'triplet_audit.json') if args.backbone == 'predict25' else receipt
    trace = collect_trace(audit)
    assert trace
    rows = []
    for row in trace:
        sf = row.get('future_sigma', row.get('sigma_future'))
        sc = row.get('actual_context_sigma', row.get('sigma_context'))
        tf = row['future_timestep']
        tc = row.get('context_timestep', row.get('t_context'))
        assert sf == sc and tf == tc, row
        item = dict(step=row['step'], future_sigma=sf, context_sigma=sc, future_timestep=tf, context_timestep=tc)
        if rows and item['step'] == rows[-1]['step']:
            assert item == rows[-1]
        else:
            rows.append(item)
    video = result.with_name('video.mp4')
    r.verify(video, receipt['video_sha256'])
    r.write(output / 'sigma_trace.json', rows)
    r.write(output / 'experiment_verification.json', dict(status='PASS', backbone=args.backbone,
        scope='exact per-forward sigma/timestep alignment; not historical byte-exact reproduction',
        steps=len(rows), calls=len(trace), video=str(video), video_sha256=r.sha(video),
        runner_sha256=r.sha(__file__), entry_sha256=r.sha(entry), source_patches=patches))
    print('EQUAL_SIGMA_PASS', args.backbone, len(rows), str(video), flush=True)


if __name__ == '__main__':
    main()
