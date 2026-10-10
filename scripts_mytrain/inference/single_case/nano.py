"""Single-case Cosmos3-Nano inference with explicit clean/noisy/equal-sigma modes."""
from common import arguments,finish,load,HIST

def main():
    a,c,out=arguments('nano')
    import torch,json,sys,types,numpy as np
    from pathlib import Path
    from diffusers import Cosmos3OmniTransformer,AutoencoderKLWan,Cosmos3OmniPipeline,UniPCMultistepScheduler
    from transformers import Qwen3VLVisionModel
    from diffusers.utils import export_to_video
    from safetensors.torch import load_file
    from sampling import NanoContext,CleanContextAudit
    torch.set_num_threads(4)
    adapter=Path(c['adapter']);cfg=json.loads((adapter/'adapter_config.json').read_text())
    settings=types.ModuleType('settings');settings.CFG=dict(paths={k:c['model_path'] for k in ['model','data','selection','cache','output']},lora=cfg,noise={'context_scales':{'N75':.75}},data={'context_latents':2});settings.CONFIG_PATH=None;sys.modules['settings']=settings
    helpers=load('single_nano_helpers',HIST/'frozen/nano/v2/common.py')
    reference=load('single_nano_input',HIST/'frozen/nano/run_matched.py')
    reader=reference.load_wan_reader(HIST/'frozen/wan/infer_full_sa_lora_json_list.py')
    model_path=Path(c['model_path'])
    model=Cosmos3OmniTransformer.from_pretrained(model_path/'transformer',torch_dtype=torch.bfloat16,local_files_only=True,device_map={'':'cuda:0'})
    model.requires_grad_(False)
    helpers.inject_lora(model,rank=cfg['rank'],alpha=cfg['alpha'],targets=cfg['target_modules'],dropout=cfg.get('dropout',0.))
    weights=load_file(str(adapter/'adapters.safetensors'),device='cuda:0');params={n:p for n,p in model.named_parameters() if 'lora_' in n}
    assert params.keys()==weights.keys()
    with torch.no_grad():
        for n,p in params.items():
            assert p.shape==weights[n].shape and torch.isfinite(weights[n]).all()
            p.copy_(weights[n]);assert torch.equal(p,weights[n].to(p.dtype))
    del weights
    vae=AutoencoderKLWan.from_pretrained(model_path/'vae',torch_dtype=torch.float32,local_files_only=True,device_map={'':'cuda:0'})
    vision=Qwen3VLVisionModel.from_pretrained(model_path/'vision_encoder',torch_dtype=torch.bfloat16,local_files_only=True,device_map={'':'cuda:0'})
    pipe=Cosmos3OmniPipeline.from_pretrained(model_path,transformer=model,vae=vae,torch_dtype=torch.bfloat16,vision_encoder=vision,sound_tokenizer=None,safety_checker=None,enable_safety_checker=False,local_files_only=True).to('cuda')
    pipe.vae.to(dtype=torch.float32);pipe.transformer.eval().requires_grad_(False)
    pipe.scheduler=UniPCMultistepScheduler.from_config(pipe.scheduler.config,flow_shift=c['flow_shift'])
    frames=reader(Path(c['video']),512,896,max_frames=8);audit={}
    noise_cfg=dict(num_train_timesteps=c['training_timesteps'],shift=c['training_shift'])
    hook=CleanContextAudit(pipe,'N75') if c['mode']=='clean' else NanoContext(pipe,c['mode'],'N75',noise_cfg,c['context_seed'])
    with torch.inference_mode(),reference.exact_context(pipe,frames,{'num_frames':49},audit),hook:
        result=pipe(prompt=c['prompt'],negative_prompt=c['negative_prompt'],video=frames,condition_frame_indexes_vision=[0,1],condition_video_keep='first',num_frames=49,height=512,width=896,fps=c['fps'],num_inference_steps=c['steps'],guidance_scale=c['guidance'],enable_sound=False,add_resolution_template=False,add_duration_template=False,generator=torch.Generator(device='cuda').manual_seed(c['seed']))
    arr=np.asarray(result.video);assert arr.shape==(49,512,896,3) and np.isfinite(arr).all()
    export_to_video(result.video,str(out/'video.mp4'),fps=c['fps'],macro_block_size=1)
    audit['triplet']=hook.audit;finish(a,c,out,audit)

if __name__=='__main__':main()
