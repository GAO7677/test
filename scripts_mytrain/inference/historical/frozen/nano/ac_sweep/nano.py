"""Generate independent Test70/Test89 results with fixed clean context for all adapters."""
import argparse
from contextlib import nullcontext
from hooks import NanoContext, CleanContextAudit
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import Cosmos3OmniTransformer, Cosmos3OmniPipeline, AutoencoderKLWan, UniPCMultistepScheduler
from diffusers.utils import export_to_video
from safetensors.torch import load_file
from transformers import Qwen3VLVisionModel

from common import CFG, CONFIG_PATH, MODEL, OUTPUT, inject_lora, write_json, sha

MATCHED = Path('/data/gaoya/agent-data/outputs_v1/cosmos/20260929_test100_matched_io/run_matched.py')
WAN_READER = Path('/home/gaoya/code_V2V_baselines/PhysRVG-main/scripts_mytrain/inference/infer_full_sa_lora_json_list.py')
TEST89 = Path('/data/gaoya/agent-data/outputs_v1/geometry_sweep_bench/20260929_test100_clean')
TEST70 = Path('/data/gaoya/agent-data/physv_v2v_0819/manifests/test70_initial_state_v1/input_list.txt')
NEGATIVE = Path('/data/gaoya/agent-data/outputs_v1/cosmos/20260929_test100/negative_prompt.json')


def cases(benchmark):
    if benchmark == 'test89':
        source = json.loads((TEST89 / 'resolved_config.json').read_text())['cases']
        active = json.loads((TEST89 / 'active_cases.json').read_text())['cases']
        assert len(active) == 89
        return {k:source[k] for k in active}, 24
    rows = {}
    for line in TEST70.read_text().splitlines():
        row = json.loads(Path(line).read_text())
        rows[row['sample_id']] = dict(prompt=row['input_caption'], context_video=row['input_video_8f'],
                                     gt_video=row['source_video'])
    assert len(rows)==70
    return rows, 30


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=CONFIG_PATH)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--benchmark', choices=['test70','test89','test70_v2'], default='test89')
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--steps',type=int,default=CFG['evaluation']['steps'])
    parser.add_argument('--name',required=True)
    parser.add_argument('--case-manifest',type=Path)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(CFG['runtime']['evaluation_gpu'])
    torch.set_num_threads(CFG['runtime']['cpu_threads'])
    if CFG['data']['context_latents'] != 2 or CFG['data']['context_rgb_frames'] != 8:
        raise ValueError('Matched Test70/Test89 evaluation requires 8 RGB / 2 latent context frames')
    root=args.output_dir or OUTPUT/'evaluation'/args.benchmark/args.name
    if not root.resolve().is_relative_to('/data/gaoya'):
        raise ValueError('Evaluation outputs must be under /data/gaoya')
    if args.case_manifest:
        manifest=json.loads(args.case_manifest.read_text())
        rows,fps=manifest['cases'],manifest['fps']
    else:
        rows,fps=cases(args.benchmark)
    if args.limit:rows=dict(list(rows.items())[:args.limit])
    cfg=dict(context_scale=float(os.environ['CONTEXT_SCALE']),context_variant=os.environ['CONTEXT_VARIANT'],context_seed=20261007,num_frames=CFG['data']['frames'],height=CFG['data']['height'],width=CFG['data']['width'],fps=fps,context_latent_indexes=[0,1],
             seed=CFG['evaluation']['seed'],num_inference_steps=args.steps,guidance_scale=CFG['evaluation']['guidance_scale'],flow_shift=CFG['evaluation']['flow_shift'],
             add_resolution_template=False,add_duration_template=False,
             adapter=str(args.adapter) if args.adapter else None,
             adapter_sha256=sha(args.adapter/'adapters.safetensors') if args.adapter else None,
             benchmark=args.benchmark,matched_reader_sha256=sha(MATCHED),wan_reader_sha256=sha(WAN_READER),
             negative_prompt_sha256=sha(NEGATIVE),evaluator_sha256=sha(__file__),
             adapter_config_sha256=sha(args.adapter/'adapter_config.json') if args.adapter else None,
             case_manifest_sha256=sha(args.case_manifest) if args.case_manifest else None)
    if (root/'config.json').exists() and json.loads((root/'config.json').read_text()) != cfg:
        raise ValueError('Existing output directory has a different evaluation contract')
    write_json(root/'config.json',cfg)
    module_spec=importlib.util.spec_from_file_location('cosmos_matched_reader',MATCHED)
    reference=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(reference)
    reader=reference.load_wan_reader(WAN_READER)
    model=Cosmos3OmniTransformer.from_pretrained(MODEL/'transformer',torch_dtype=torch.bfloat16,
                                                local_files_only=True, device_map={'': 'cuda:0'})
    if args.adapter:
        assert (args.adapter/'complete.json').exists()
        receipt=json.loads((args.adapter/'complete.json').read_text())
        if receipt['config']['mode'] != args.name:
            raise ValueError('Checkpoint group does not match requested evaluation name')
        adapter_cfg=json.loads((args.adapter/'adapter_config.json').read_text())
        if Path(adapter_cfg['base_model']).resolve() != MODEL.resolve():
            raise ValueError('Adapter base model differs from configured model')
        model.requires_grad_(False)
        inject_lora(model,rank=adapter_cfg['rank'],alpha=adapter_cfg['alpha'],
                    targets=adapter_cfg['target_modules'],dropout=adapter_cfg.get('dropout',0.0))
        weights=load_file(str(args.adapter/'adapters.safetensors'), device='cuda:0')
        params={n:p for n,p in model.named_parameters() if 'lora_' in n}
        assert set(params)==set(weights)
        with torch.no_grad():
            for n,p in params.items():
                if p.shape != weights[n].shape or not torch.isfinite(weights[n]).all():
                    raise ValueError(f'Invalid adapter tensor: {n}')
                p.copy_(weights[n])
                if not torch.equal(p, weights[n].to(dtype=p.dtype)):
                    raise ValueError(f'Adapter copy verification failed: {n}')
        audit=dict(status='PASS',mode=receipt['config']['mode'],step=receipt['step'],
                   base_model=str(MODEL),adapter=str(args.adapter.resolve()),
                   adapter_sha256=cfg['adapter_sha256'],adapter_config=adapter_cfg,
                   tensor_count=len(params),parameter_count=sum(p.numel() for p in params.values()),
                   all_finite=True,all_loaded_tensors_equal_after_dtype_cast=True,
                   nonzero_B_tensors=sum(bool(torch.count_nonzero(w)) for n,w in weights.items() if 'lora_B' in n),
                   device=str(next(iter(params.values())).device),dtype=str(next(iter(params.values())).dtype))
        write_json(root/'weight_load_audit.json',audit)
        print('WEIGHT_LOAD_AUDIT '+json.dumps(audit),flush=True)
        del weights
    vae=AutoencoderKLWan.from_pretrained(MODEL/'vae', torch_dtype=torch.float32,
        local_files_only=True, device_map={'': 'cuda:0'})
    vision=Qwen3VLVisionModel.from_pretrained(MODEL/'vision_encoder',torch_dtype=torch.bfloat16,
        local_files_only=True,device_map={'': 'cuda:0'})
    pipe=Cosmos3OmniPipeline.from_pretrained(MODEL,transformer=model,vae=vae,torch_dtype=torch.bfloat16,
        vision_encoder=vision,sound_tokenizer=None,safety_checker=None,enable_safety_checker=False,local_files_only=True).to('cuda')
    pipe.vae.to(dtype=torch.float32)
    pipe.transformer.eval().requires_grad_(False)
    pipe.scheduler=UniPCMultistepScheduler.from_config(pipe.scheduler.config,flow_shift=CFG['evaluation']['flow_shift'])
    negative=json.dumps(json.loads(NEGATIVE.read_text()),ensure_ascii=True,separators=(',',':'))
    for index,(key,row) in enumerate(rows.items()):
        dest=root/'results'/key
        provenance=dict(context_sha256=sha(row['context_video']),gt_sha256=sha(row['gt_video']),
                        prompt=row['prompt'],config=cfg)
        assert provenance['context_sha256']==row['context_video_sha256'], 'Source context changed'
        assert provenance['gt_sha256']==row['gt_video_sha256'], 'Source GT changed'
        if (dest/'result.json').exists():
            old=json.loads((dest/'result.json').read_text())
            assert old['provenance']==provenance and old['video_sha256']==sha(dest/'video.mp4')
            continue
        frames=reader(Path(row['context_video']),CFG['data']['height'],CFG['data']['width'],max_frames=CFG['data']['context_rgb_frames'])
        audit={};start=time.time()
        hook=(CleanContextAudit(pipe,args.name) if os.environ['CONTEXT_VARIANT']=='clean' else NanoContext(pipe,os.environ['CONTEXT_VARIANT'],args.name,CFG['noise'],20261007,float(os.environ['CONTEXT_SCALE'])))
        with torch.inference_mode(),reference.exact_context(pipe,frames,cfg,audit),hook:
            result=pipe(prompt=row['prompt'],negative_prompt=negative,video=frames,
                condition_frame_indexes_vision=[0,1],condition_video_keep='first',
                num_frames=CFG['data']['frames'],height=CFG['data']['height'],width=CFG['data']['width'],fps=fps,num_inference_steps=args.steps,
                guidance_scale=CFG['evaluation']['guidance_scale'],enable_sound=False,add_resolution_template=False,add_duration_template=False,
                generator=torch.Generator(device='cuda').manual_seed(CFG['evaluation']['seed']))
        audit['triplet']=hook.audit
        arr=np.asarray(result.video)
        assert arr.shape==(CFG['data']['frames'],CFG['data']['height'],CFG['data']['width'],3) and np.isfinite(arr).all()
        dest.mkdir(parents=True,exist_ok=True)
        export_to_video(result.video,str(dest/'video.partial.mp4'),fps=fps,macro_block_size=1)
        (dest/'video.partial.mp4').replace(dest/'video.mp4')
        write_json(dest/'result.json',dict(provenance=provenance,context_audit=audit,
            video_sha256=sha(dest/'video.mp4'),seconds=time.time()-start,shape=list(arr.shape)))
        status=dict(completed=index+1,total=len(rows),current=key,seconds=time.time()-start)
        write_json(root/'progress.json',status);print(status,flush=True)
    write_json(root/'complete.json',dict(count=len(rows),config=cfg))


if __name__=='__main__':
    main()
