"""Generate independent Test70/Test89 results with fixed clean context for all adapters."""
import argparse
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
    parser.add_argument('--benchmark', choices=['test70','test89'], default='test89')
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--steps',type=int,default=CFG['evaluation']['steps'])
    parser.add_argument('--name',required=True)
    args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(CFG['runtime']['evaluation_gpu'])
    torch.set_num_threads(CFG['runtime']['cpu_threads'])
    if CFG['data']['context_latents'] != 2 or CFG['data']['context_rgb_frames'] != 8:
        raise ValueError('Matched Test70/Test89 evaluation requires 8 RGB / 2 latent context frames')
    root=OUTPUT/'evaluation'/args.benchmark/args.name
    rows,fps=cases(args.benchmark)
    if args.limit:rows=dict(list(rows.items())[:args.limit])
    cfg=dict(num_frames=CFG['data']['frames'],height=CFG['data']['height'],width=CFG['data']['width'],fps=fps,context_latent_indexes=[0,1],
             seed=CFG['evaluation']['seed'],num_inference_steps=args.steps,guidance_scale=CFG['evaluation']['guidance_scale'],flow_shift=CFG['evaluation']['flow_shift'],
             add_resolution_template=False,add_duration_template=False,
             adapter=str(args.adapter) if args.adapter else None,
             adapter_sha256=sha(args.adapter/'adapters.safetensors') if args.adapter else None,
             benchmark=args.benchmark,matched_reader_sha256=sha(MATCHED),wan_reader_sha256=sha(WAN_READER))
    write_json(root/'config.json',cfg)
    module_spec=importlib.util.spec_from_file_location('cosmos_matched_reader',MATCHED)
    reference=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(reference)
    reader=reference.load_wan_reader(WAN_READER)
    model=Cosmos3OmniTransformer.from_pretrained(MODEL/'transformer',torch_dtype=torch.bfloat16,
                                                local_files_only=True, device_map={'': 'cuda:0'})
    if args.adapter:
        assert (args.adapter/'complete.json').exists()
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
            for n,p in params.items():p.copy_(weights[n])
    vae=AutoencoderKLWan.from_pretrained(MODEL/'vae', torch_dtype=torch.float32,
        local_files_only=True, device_map={'': 'cuda:0'})
    pipe=Cosmos3OmniPipeline.from_pretrained(MODEL,transformer=model,vae=vae,torch_dtype=torch.bfloat16,
        sound_tokenizer=None,safety_checker=None,enable_safety_checker=False,local_files_only=True).to('cuda')
    pipe.vae.to(dtype=torch.float32)
    pipe.transformer.eval().requires_grad_(False)
    pipe.scheduler=UniPCMultistepScheduler.from_config(pipe.scheduler.config,flow_shift=CFG['evaluation']['flow_shift'])
    negative=json.dumps(json.loads(NEGATIVE.read_text()),ensure_ascii=True,separators=(',',':'))
    for index,(key,row) in enumerate(rows.items()):
        dest=root/'results'/key
        provenance=dict(context_sha256=sha(row['context_video']),gt_sha256=sha(row['gt_video']),
                        prompt=row['prompt'],config=cfg)
        if (dest/'result.json').exists():
            old=json.loads((dest/'result.json').read_text())
            assert old['provenance']==provenance and old['video_sha256']==sha(dest/'video.mp4')
            continue
        frames=reader(Path(row['context_video']),CFG['data']['height'],CFG['data']['width'],max_frames=CFG['data']['context_rgb_frames'])
        audit={};start=time.time()
        with torch.inference_mode(),reference.exact_context(pipe,frames,cfg,audit):
            result=pipe(prompt=row['prompt'],negative_prompt=negative,video=frames,
                condition_frame_indexes_vision=[0,1],condition_video_keep='first',
                num_frames=CFG['data']['frames'],height=CFG['data']['height'],width=CFG['data']['width'],fps=fps,num_inference_steps=args.steps,
                guidance_scale=CFG['evaluation']['guidance_scale'],enable_sound=False,add_resolution_template=False,add_duration_template=False,
                generator=torch.Generator(device='cuda').manual_seed(CFG['evaluation']['seed']))
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
