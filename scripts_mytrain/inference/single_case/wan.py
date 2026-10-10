"""Single-case Wan inference, preserving the historical native pipeline and hooks."""
from common import arguments,finish,load,save,HIST

def main():
    a,c,out=arguments('wan')
    import torch,numpy as np,imageio.v2 as imageio
    from types import SimpleNamespace
    from pathlib import Path
    from peft import PeftModel,get_peft_model_state_dict
    from safetensors.torch import load_file
    from diffusers import FlowMatchEulerDiscreteScheduler
    from sampling import ContextNoise
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    reader=load('single_wan_reader',HIST/'frozen/wan/infer_full_sa_lora_json_list.py')
    pipe,*_=reader.load_pipeline(SimpleNamespace(lora_checkpoint=None,context_frames=8,context_mask_mode='official_fixed2',model_id=Path(c['model_path']),physrvg_dit_checkpoint=Path(c['physrvg_dit_checkpoint']),device='cuda:0',object_xssc_trainable=None))
    pipe.transformer.eval();pipe.vae.eval();pipe.text_encoder.eval()
    pipe.transformer=PeftModel.from_pretrained(pipe.transformer,c['adapter'],adapter_name='analysis_N75',is_trainable=False)
    expected=load_file(str(Path(c['adapter'])/'adapter_model.safetensors'))
    actual=get_peft_model_state_dict(pipe.transformer,adapter_name='analysis_N75')
    assert expected.keys()==actual.keys() and all(torch.equal(actual[k].cpu().float(),v.float()) for k,v in expected.items())
    pipe.transformer.set_adapter('analysis_N75');pipe.transformer.eval()
    frames=reader.load_context_video(Path(c['video']),512,896,max_frames=8)
    if c.get('prompt_cache'):prompt=load_file(c['prompt_cache'])['prompt_embeds'].to('cuda:0')
    else:
        with torch.inference_mode():prompt,_=pipe.encode_prompt(prompt=c['prompt'],negative_prompt='',do_classifier_free_guidance=False,device=torch.device('cuda:0'))
    cache=load_file(c['context_cache'])['context_latent'].to('cuda:0') if c.get('context_cache') else None
    orig=pipe.prepare_latents
    def prepare(*args,**kwargs):
        if cache is not None:kwargs['precomputed_video_latent']=cache
        return orig(*args,**kwargs)
    pipe.prepare_latents=prepare
    train=FlowMatchEulerDiscreteScheduler.from_pretrained(c['model_path'],subfolder='scheduler',local_files_only=True)
    train.set_timesteps(c['training_timesteps'],device='cuda:0')
    scale={'clean':0.,'noisy':.75,'equal_sigma':1.}[c['mode']]
    hook=ContextNoise(pipe,train,scale,pipe.transformer.get_base_model().config.patch_size)
    pipe.set_progress_bar_config(disable=True);torch.manual_seed(c['seed'])
    with torch.inference_mode(),hook:
        latent=pipe(video=frames,device=torch.device('cuda:0'),prompt_embeds=prompt,negative_prompt=None,height=512,width=896,num_frames=49,num_inference_steps=c['steps'],guidance_scale=c['guidance'],do_cfg=False,generator=torch.Generator(device='cuda').manual_seed(c['seed']),output_type='latent',dynamic_condition_mask=False,context_noise_mode='clean').frames.detach().cpu()
    pipe.transformer.to('cpu');torch.cuda.empty_cache()
    with torch.inference_mode():
        z=latent.to('cuda:0',dtype=pipe.vae_pipeline_dtype)
        mean=torch.tensor(pipe.vae.config.latents_mean).view(1,pipe.vae.config.z_dim,1,1,1).to(z)
        inv_std=1./torch.tensor(pipe.vae.config.latents_std).view(1,pipe.vae.config.z_dim,1,1,1).to(z)
        decoded=pipe.vae.decode(z/inv_std+mean,return_dict=False)[0]
        result=pipe.video_processor.postprocess_video(decoded,output_type='np')[0]
    with imageio.get_writer(str(out/'video.mp4'),fps=c['fps'],codec='libx264',quality=8,macro_block_size=None,ffmpeg_params=['-threads','1']) as w:
        for frame in result:w.append_data((np.clip(frame,0,1)*255).astype(np.uint8))
    finish(a,c,out,dict(trace=hook.trace))

if __name__=='__main__':main()
