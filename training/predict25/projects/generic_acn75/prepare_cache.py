"""Materialize the outputs of NVIDIA's unchanged training encoders.

Only asset loading and deterministic sample selection are project-specific.
The reference model omits DiT construction; its normalization, encode and
get_data_and_condition methods are inherited directly from NVIDIA.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.utils.data import default_collate

from cosmos_predict2._src.imaginaire.lazy_config import instantiate
from cosmos_predict2._src.imaginaire.utils.callback import LowPrecisionCallback
from cosmos_predict2._src.predict2.models.video2world_model_rectified_flow import Video2WorldModelRectifiedFlow
from projects.generic_acn75.config import make_config, SETTINGS
from projects.generic_acn75.data import PrefixVideoDataset, CachedVideoDataset
from projects.generic_acn75.model import CudaTextEncoder, MechanismModel


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


class OnlineReference(Video2WorldModelRectifiedFlow):
    def set_up_model(self):
        self.conditioner = instantiate(self.config.conditioner)
        self.net = torch.nn.Identity()
        self._param_count = 0


class CachedReference(OnlineReference):
    _normalize_video_databatch_inplace = MechanismModel._normalize_video_databatch_inplace


def dataset():
    return PrefixVideoDataset(dataset_dir=SETTINGS['dataset'], num_frames=SETTINGS['frames'],
                              video_size=(SETTINGS['height'], SETTINGS['width']), caption_format='text')


def initialize():
    root = Path(SETTINGS['cache_root'])
    root.mkdir(parents=True, exist_ok=True)
    for name in ['text', 'latents', 'reports']:
        (root/name).mkdir(exist_ok=True)
    s = SETTINGS
    assert sha(s['vae_checkpoint']) == s['vae_checkpoint_sha256']
    weights = {p.name: sha(p) for p in sorted(Path(s['text_weights']).iterdir())
               if p.suffix in ['.safetensors', '.json', '.jinja']}
    sources = [
        'cosmos_predict2/_src/predict2/models/text2world_model_rectified_flow.py',
        'cosmos_predict2/_src/predict2/models/video2world_model_rectified_flow.py',
        'cosmos_predict2/_src/predict2/tokenizers/wan2pt1.py',
        'cosmos_predict2/_src/predict2/datasets/local_datasets/dataset_video.py',
        'cosmos_predict2/_src/predict2/datasets/local_datasets/dataset_utils.py',
        'cosmos_predict2/_src/predict2/text_encoders/text_encoder.py',
        'cosmos_predict2/_src/predict2/text_encoders/reason1.py',
        'cosmos_predict2/_src/reason1/tokenizer/processor.py',
        'projects/generic_acn75/data.py', 'projects/generic_acn75/text.py',
    ]
    contract = dict(schema=1, implementation='NVIDIA Cosmos-Predict2.5 official training online encoders',
        official_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        video_size=[s['height'], s['width']], frames=s['frames'], fps=s['fps'],
        temporal_selection=s['temporal_preprocess'], spatial_preprocess=s['spatial_preprocess'],
        vae_sha256=s['vae_checkpoint_sha256'], text_assets=weights,
        video_dtype='bfloat16', text_dtype='bfloat16', text_shape=[512,100352],
        text_projection='not_applied', embedding_concat_strategy='full_concat',
        cache_batch_size=s['cache_batch_size'], tokenizer_config=str(make_config(online_encoding=True).model.config.tokenizer),
        sources={path:sha(path) for path in sources},
        runtime={name:importlib.metadata.version(name) for name in ['torch','torchvision','transformers','flash-attn','transformer-engine']},
        cudnn_version=torch.backends.cudnn.version())
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    if (root/'contract.json').exists():
        assert json.loads((root/'contract.json').read_text())['fingerprint'] == fingerprint, 'Use a new cache version when the contract changes'
    rows = json.loads((Path(s['dataset'])/'manifest.json').read_text())
    output = []
    for index,row in enumerate(rows):
        caption = row['caption'].strip()
        caption_sha = hashlib.sha256(caption.encode()).hexdigest()
        output.append(dict(index=index, logical_key=row['logical_key'], video=row['video'],
            caption=caption, fps=float(s['fps']), video_sha256=sha(row['video']), caption_sha256=caption_sha,
            latent_file=f'latents/{index:05d}.safetensors', text_file=f'text/{caption_sha}.safetensors'))
    atomic_json(root/'manifest.json',output)
    atomic_json(root/'contract.json',dict(fingerprint=fingerprint,contract=contract))
    print('INITIALIZED',len(output),len({r['caption_sha256'] for r in output}),fingerprint,flush=True)


def save_exact(root, row, kind, tensor, fingerprint):
    key = 'embedding' if kind == 'text' else 'latent'
    field = 'text_file' if kind == 'text' else 'latent_file'
    source = row['caption_sha256' if kind == 'text' else 'video_sha256']
    expected = [512,100352] if kind == 'text' else [16,13,SETTINGS['height']//8,SETTINGS['width']//8]
    assert list(tensor.shape) == expected and tensor.dtype == torch.bfloat16
    assert torch.isfinite(tensor).all()
    path = root/row[field]
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    save_file({key:tensor.cpu().contiguous()},str(temporary),metadata=dict(fingerprint=fingerprint,source_sha256=source))
    temporary.replace(path)
    restored = load_file(str(path),device='cuda:0')[key]
    assert torch.equal(restored,tensor), f'Cache serialization changed {path}'
    return dict(file=row[field],sha256=sha(path),roundtrip_exact=True)


def existing(root,row,kind,fingerprint):
    field='text_file' if kind=='text' else 'latent_file'
    receipt=root/'reports'/('file_'+hashlib.sha256(row[field].encode()).hexdigest()+'.json')
    if not receipt.exists():return None
    report=json.loads(receipt.read_text())
    path=root/row[field]
    assert report['fingerprint']==fingerprint and report['roundtrip_exact']
    assert sha(path)==report['sha256']
    return report


def receipt_path(root,row,kind):
    field='text_file' if kind=='text' else 'latent_file'
    return root/'reports'/('file_'+hashlib.sha256(row[field].encode()).hexdigest()+'.json')


def online_batch(ds,rows,root):
    items=[]
    for row in rows:
        assert sha(row['video'])==row['video_sha256'], 'Source video changed after manifest creation'
        item=ds[row['index']]
        assert item['ai_caption']==row['caption'] and abs(item['fps']-row['fps'])<0.01
        item['t5_text_embeddings']=load_file(str(root/row['text_file']))['embedding']
        item['t5_text_mask']=torch.ones(512)
        items.append(item)
    batch=to_cuda(default_collate(items))
    # In official online forward(), this mask is constructed after callbacks.
    batch['t5_text_mask']=torch.ones(len(rows),512,device='cuda')
    return batch


def to_cuda(batch):
    batch={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
    callback=LowPrecisionCallback(config=None,trainer=None,update_iter=1)
    callback.on_train_start(SimpleNamespace(precision=torch.bfloat16))
    callback.on_training_step_start(None,batch)
    return batch


@torch.inference_mode()
def produce(args):
    root=Path(SETTINGS['cache_root'])
    fingerprint=json.loads((root/'contract.json').read_text())['fingerprint']
    rows=json.loads((root/'manifest.json').read_text())
    if args.small:rows=rows[:8]
    if args.stage=='text':
        rows=list({r['caption_sha256']:r for r in rows}.values())
    batch_size=SETTINGS['cache_batch_size']
    assert batch_size==SETTINGS['per_gpu_batch']==2
    groups=[rows[i:i+batch_size] for i in range(0,len(rows),batch_size)]
    selected=[group for index,group in enumerate(groups) if index%args.shards==args.shard]
    if args.stage=='text':
        encoder=CudaTextEncoder(make_config(online_encoding=True).model.config.text_encoder_config,SETTINGS['text_weights'])
    else:
        ds=dataset()
        encoder=OnlineReference(make_config(online_encoding=True).model.config).eval()
    def encode(group):
        if args.stage=='text':
            return encoder.compute_text_embeddings_online({'ai_caption':[r['caption'] for r in group]},'ai_caption')
        _,latent,_=encoder.get_data_and_condition(online_batch(ds,group,root))
        assert torch.equal(latent.to(torch.bfloat16).float(),latent)
        return latent.to(torch.bfloat16)
    checks=[]
    start=time.time();completed=0
    total=sum(len(group) for group in selected)
    for group in selected:
        # Pad only the last odd group. Do not save the duplicate sample twice.
        padded=group+[group[-1]]*(batch_size-len(group))
        reports=[existing(root,row,args.stage,fingerprint) for row in group]
        if not all(reports):
            tensors=encode(padded)
            for offset,row in enumerate(group):
                report=save_exact(root,row,args.stage,tensors[offset],fingerprint)
                report.update(fingerprint=fingerprint,index=row['index'],stage=args.stage,
                              encoding_batch_size=batch_size,companion_indices=[r['index'] for r in padded],
                              cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu=torch.cuda.get_device_name())
                atomic_json(receipt_path(root,row,args.stage),report)
            del tensors
        if args.small:
            repeats=encode(padded)
            for offset,row in enumerate(group):
                key='embedding' if args.stage=='text' else 'latent'
                field='text_file' if args.stage=='text' else 'latent_file'
                restored=load_file(str(root/row[field]),device='cuda:0')[key]
                assert torch.equal(restored,repeats[offset]), 'Independent online repeat differs from cache'
                checks.append(dict(index=row['index'],online_repeat_exact=True,max_abs=0,batch_size=batch_size))
                del restored
            del repeats
        completed+=len(group)
        status=dict(stage=args.stage,shard=args.shard,shards=args.shards,completed=completed,total=total,
                    elapsed_seconds=time.time()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,fingerprint=fingerprint)
        atomic_json(root/'reports'/f'progress_{args.stage}_{args.shard}.json',status)
        print(json.dumps(status),flush=True)
    companion_checks=[]
    if args.small:
        first=selected[0][0]
        field='text_file' if args.stage=='text' else 'latent_file'
        key='embedding' if args.stage=='text' else 'latent'
        baseline=load_file(str(root/first[field]),device='cuda:0')[key]
        for group,position in [([first,first],0),([selected[-1][-1],first],1)]:
            repeat=encode(group)[position]
            assert torch.equal(baseline,repeat), 'Encoding depends on companion sample or position'
            companion_checks.append(dict(indices=[r['index'] for r in group],position=position,exact=True,max_abs=0))
    atomic_json(root/'reports'/f'{args.stage}_{"small" if args.small else "full"}_{args.shard}.json',
                dict(status='PASS',fingerprint=fingerprint,completed=completed,shard=args.shard,shards=args.shards,
                     repeat_checks=checks,companion_checks=companion_checks,encoding_batch_size=batch_size,
                     peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))


@torch.inference_mode()
def compare():
    root=Path(SETTINGS['cache_root'])
    rows=json.loads((root/'manifest.json').read_text())
    ds=dataset()
    cached_ds=CachedVideoDataset(root,limit=8)
    online=OnlineReference(make_config(online_encoding=True).model.config).eval()
    cached=CachedReference(make_config().model.config).eval()
    assert not hasattr(cached.tokenizer,'model') and cached.text_encoder is None
    checks=[]
    for index in range(0,8,SETTINGS['cache_batch_size']):
        indices=list(range(index,index+SETTINGS['cache_batch_size']))
        torch.manual_seed(42)
        _,reference,condition=online.get_data_and_condition(online_batch(ds,[rows[i] for i in indices],root))
        torch.manual_seed(42)
        _,restored,cache_condition=cached.get_data_and_condition(to_cuda(default_collate([cached_ds[i] for i in indices])))
        assert torch.equal(reference,restored)
        for name,value in condition.to_dict().items():
            other=cache_condition.to_dict()[name]
            if isinstance(value,torch.Tensor):
                assert value.dtype==other.dtype,(name,value.dtype,other.dtype)
                assert torch.equal(value,other),name
            else:assert value==other,name
        checks.extend(dict(index=i,latent_exact=True,condition_exact=True,max_abs=0) for i in indices)
    result=dict(status='PASS',fingerprint=cached_ds.contract['fingerprint'],checks=checks,
                cached_vae_loaded=False,cached_text_encoder_loaded=False,conditioning_dtype_exact=True,
                official_low_precision_callback=True,encoding_batch_size=SETTINGS['cache_batch_size'],
                reference='Inherited official Video2WorldModelRectifiedFlow.get_data_and_condition')
    atomic_json(root/'reports'/'online_cached_comparison.json',result)
    print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['initialize','text','video','compare'])
    parser.add_argument('--small',action='store_true')
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    args=parser.parse_args()
    assert 0<=args.shard<args.shards
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=SETTINGS['cudnn_benchmark']
    if args.stage=='initialize':initialize()
    elif args.stage=='compare':compare()
    else:
        if args.stage=='video' and not args.small and args.shard==0:compare()
        produce(args)


if __name__=='__main__':main()
