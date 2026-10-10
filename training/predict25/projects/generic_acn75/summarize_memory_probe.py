"""Collect peak statistics across phases; native DeviceMonitor resets peak counters."""
import json
from pathlib import Path


def main():
    settings=json.loads(Path(__file__).with_name('memory_probe_settings.json').read_text())
    root=Path(settings['output'])
    runs=root/'physrvg-full-sa'/'predict25-official-memory-probe'
    result=dict(settings=settings,runs=[],status='PASS')
    for mode in settings['modes']:
        run=runs/f'smoke_mb4_{mode}'
        ranks=[]
        for rank in range(settings['world_size']):
            rows=[json.loads(line) for line in (run/'memory'/f'rank{rank}.jsonl').read_text().splitlines()]
            assert rows[-1]['stage']=='PASS' and rows[-1]['iteration']==settings['optimizer_steps']
            assert [r['iteration'] for r in rows if r['stage']=='optimizer_step_complete']==list(range(1,settings['optimizer_steps']+1))
            audit=json.loads((run/'audit'/f'validation_rank{rank}.json').read_text())
            assert audit['status']=='PASS' and audit['finite_gradients'] and audit['lora_changed'] and audit['frozen_parameters_unchanged']
            assert audit['per_gpu_batch']==settings['per_gpu_batch'] and audit['grad_accum']==settings['grad_accum']
            assert not audit['vae_loaded'] and not audit['text_encoder_loaded']
            ranks.append(dict(rank=rank,optimizer_steps=rows[-1]['iteration'],
                peak_allocated_gib=max(r['peak_allocated_gib'] for r in rows),
                peak_reserved_gib=max(r['peak_reserved_gib'] for r in rows),
                minimum_device_free_gib=min(r['device_free_gib'] for r in rows),
                losses=[r['loss'] for r in rows if 'loss' in r]))
        meta=json.loads((run/'adapters'/f'step{settings["optimizer_steps"]:06d}'/'metadata.json').read_text())
        assert meta['iteration']==settings['optimizer_steps'] and meta['adapter_roundtrip_exact']
        latest=run/'checkpoints'/'latest_checkpoint.txt'
        assert latest.is_file() and (latest.parent/latest.read_text().strip()).is_file()
        result['runs'].append(dict(mode=mode,status='PASS',ranks=ranks,checkpoint=str(latest.parent/latest.read_text().strip())))
    (root/'summary.json').write_text(json.dumps(result,indent=2))
    lines=['# Predict2.5 batch4 GPU0/1 memory test','',
           'Status: PASS. A/C/N75 each completed two optimizer updates through the official training entrypoint.',
           '', 'Per GPU batch4, world size2, accumulation1, effective batch8. Existing verified batch2-encoded cache is used only to measure training memory; batch4 online encoder numerical equivalence is not claimed. No formal 3000-step training was started.',
           '', '| Mode | GPU0 allocated peak GiB | GPU1 allocated peak GiB | Max reserved GiB | Min device free GiB |',
           '|---|---:|---:|---:|---:|']
    for run in result['runs']:
        a,b=run['ranks']
        lines.append(f'| {run["mode"]} | {a["peak_allocated_gib"]:.3f} | {b["peak_allocated_gib"]:.3f} | {max(a["peak_reserved_gib"],b["peak_reserved_gib"]):.3f} | {min(a["minimum_device_free_gib"],b["minimum_device_free_gib"]):.3f} |')
    lines+=['','Peak allocation is the maximum across recorded forward/backward/update phases, because the official DeviceMonitor resets allocator peak counters at each logged update. These are short-run results on the current approximately 48 GiB GPUs, not on standard 24 GiB RTX4090 cards.','',
            'Logs: A.log, C.log, N75.log. Per-rank memory traces, gradient/freeze audits and checkpoints are under physrvg-full-sa/predict25-official-memory-probe/smoke_mb4_{A,C,N75}/.']
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
