"""Fail closed unless the complete official latent and text cache is verified."""
import hashlib
import json
from pathlib import Path
from safetensors import safe_open


def verify():
    settings=json.loads(Path(__file__).with_name('settings.json').read_text())
    root=Path(settings['cache_root'])
    contract=json.loads((root/'contract.json').read_text())
    fingerprint=contract['fingerprint']
    assert contract['contract']['cache_batch_size']==settings['per_gpu_batch']==2
    assert contract['contract']['vae_sha256']==settings['vae_checkpoint_sha256']
    assert contract['contract']['video_size']==[settings['height'],settings['width']]
    assert contract['contract']['frames']==settings['frames']
    rows=json.loads((root/'manifest.json').read_text())
    source=json.loads((Path(settings['dataset'])/'manifest.json').read_text())
    assert len(rows)==len(source)==2175
    checked=set()
    for row,original in zip(rows,source):
        assert row['video']==original['video'] and row['caption']==original['caption'].strip()
        assert isinstance(row['fps'],float) and row['fps']==float(settings['fps'])
        for kind,field,shape in [('latent','latent_file',[16,13,settings['height']//8,settings['width']//8]),
                                  ('embedding','text_file',[512,100352])]:
            if row[field] in checked:continue
            path=root/row[field]
            with safe_open(str(path),framework='pt') as handle:
                assert handle.metadata()['fingerprint']==fingerprint
                assert handle.metadata()['source_sha256']==row['video_sha256' if kind=='latent' else 'caption_sha256']
                assert handle.get_slice(kind).get_shape()==shape
                assert handle.get_slice(kind).get_dtype()=='BF16'
            receipt=root/'reports'/('file_'+hashlib.sha256(row[field].encode()).hexdigest()+'.json')
            report=json.loads(receipt.read_text())
            assert report['fingerprint']==fingerprint and report['roundtrip_exact']
            assert report['encoding_batch_size']==settings['per_gpu_batch']
            digest=hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda:handle.read(8*1024*1024),b''):digest.update(chunk)
            assert digest.hexdigest()==report['sha256'],path
            checked.add(row[field])
    for stage in ['video','text']:
        reports=[json.loads(p.read_text()) for p in (root/'reports').glob(f'{stage}_full_*.json')]
        assert reports and all(r['status']=='PASS' and r['fingerprint']==fingerprint for r in reports)
        shards=reports[0]['shards']
        assert {r['shard'] for r in reports}==set(range(shards))
        assert all(r['shards']==shards for r in reports)
        assert sum(r['completed'] for r in reports)==(len(rows) if stage=='video' else len({r['caption_sha256'] for r in rows}))
    comparison=json.loads((root/'reports'/'online_cached_comparison.json').read_text())
    assert comparison['status']=='PASS' and comparison['fingerprint']==fingerprint
    assert comparison['conditioning_dtype_exact']
    assert comparison['official_low_precision_callback']
    assert len(comparison['checks'])==8
    for stage in ['text','video']:
        small_checks=[json.loads(p.read_text()) for p in (root/'reports').glob(f'{stage}_small_*.json')]
        assert small_checks and all(r['status']=='PASS' and r['fingerprint']==fingerprint and r['repeat_checks'] and r['companion_checks'] for r in small_checks)
    result=dict(status='PASS',samples=len(rows),unique_captions=len({r['caption_sha256'] for r in rows}),
                verified_files=len(checked),fingerprint=fingerprint,cache_root=str(root),encoding_batch_size=settings['per_gpu_batch'])
    temporary=root/'ready.tmp'
    temporary.write_text(json.dumps(result,indent=2))
    temporary.replace(root/'ready.json')
    print(json.dumps(result))
    return result


if __name__=='__main__':verify()
