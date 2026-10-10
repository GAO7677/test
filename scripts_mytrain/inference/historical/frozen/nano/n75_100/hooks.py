"""Isolated inference-only context protocol adapters; no training or shared code changes."""
import hashlib
import torch


def digest(x):
    return hashlib.sha256(x.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def choose_sigma(variant, mode, future, grid, scale=.75):
    if variant == 'clean_noisy':
        return torch.zeros_like(future)
    if mode == 'A' and variant != 'shared075':
        return future.clone()
    return grid[(grid - scale * future).abs().argmin()].reshape_as(future)


class NanoContext:
    def __init__(self, pipe, variant, mode, cfg, seed, context_scale=.75):
        from diffusers import FlowMatchEulerDiscreteScheduler
        self.pipe,self.variant,self.mode,self.seed=pipe,variant,mode,seed
        self.context_scale=float(context_scale)
        assert mode=="N75" and variant=="scaled" and self.context_scale == 1.0
        self.table=FlowMatchEulerDiscreteScheduler(num_train_timesteps=cfg['num_train_timesteps'],shift=cfg['shift'])
        self.n=cfg['num_train_timesteps'];self.calls=0;self.index=0
        self.audit=dict(variant=variant,mode=mode,context_seed=seed,context_scale=self.context_scale,trace=[])
    def prepare(self,*args,**kwargs):
        result=self.prepare_orig(*args,**kwargs)
        self.clean=result[0][:,:,:2].detach().clone()
        self.noise=torch.randn(self.clean.shape,device=self.clean.device,dtype=torch.float32,
            generator=torch.Generator(device=self.clean.device).manual_seed(self.seed))
        self.audit.update(clean_context_sha256=digest(self.clean),context_noise_sha256=digest(self.noise),
                          future_initial_sha256=digest(result[0][:,:,2:]))
        return result
    def before(self,module,args,kw):
        assert not args and self.calls//2==self.index and self.pipe.do_classifier_free_guidance
        x=kw['vision_tokens'][0]; sf=self.pipe.scheduler.sigmas[self.index].to(x.device).float()
        tf=self.pipe.scheduler.timesteps[self.index].to(x.device).float()
        grid=self.table.sigmas[:self.n].to(sf)
        sc=choose_sigma(self.variant,self.mode,sf,grid,self.context_scale)
        tc=sc*self.n
        if self.variant=='matched' and self.mode=='A':tc=tf
        context=((1-sc)*self.clean.float()+sc*self.noise).to(x)
        y=x.clone();y[:,:,:2]=context
        assert torch.equal(y[:,:,2:],x[:,:,2:])
        t,h,w=kw['vision_token_shapes'][0];stride=h*w
        levels=torch.full((t*stride,),float(tf),device=x.device);levels[:2*stride]=tc
        item=dict(step=self.index,sigma_future=float(sf),sigma_context=float(sc),t_context=float(tc),
                  context_sha256=digest(context),context_role='noisy',context_loss=False)
        if self.calls%2==0:self.audit['trace'].append(item)
        else:assert item==self.audit['trace'][-1], 'CFG context mismatch'
        self.calls+=1
        return args,dict(kw,vision_tokens=[y],vision_timesteps=levels,
            vision_noisy_frame_indexes=[torch.arange(t,device=x.device)],
            vision_mse_loss_indexes=kw['vision_sequence_indexes'])
    def step(self,*args,**kwargs):
        assert self.calls==2*(self.index+1)
        out=self.step_orig(*args,**kwargs)
        out[0][...,:2,:,:]=self.clean
        self.index+=1
        return out
    def __enter__(self):
        self.prepare_orig=self.pipe.prepare_latents;self.step_orig=self.pipe.scheduler.step
        self.pipe.prepare_latents=self.prepare;self.pipe.scheduler.step=self.step
        self.handle=self.pipe.transformer.register_forward_pre_hook(self.before,with_kwargs=True)
        return self
    def __exit__(self,typ,value,tb):
        self.handle.remove();self.pipe.prepare_latents=self.prepare_orig;self.pipe.scheduler.step=self.step_orig
        if typ is None:
            assert self.index==len(self.pipe.scheduler.timesteps)
            self.audit.update(status='PASS',calls=self.calls,steps=self.index)


def install_predict(variant,mode,seed,audit_file):
    import json
    from pathlib import Path
    from dataclasses import replace
    from projects.generic_acn75.inference_runtime import BenchmarkModel
    original=BenchmarkModel.denoise;decode_orig=BenchmarkModel.decode
    encode_orig=BenchmarkModel.encode
    # Freeze the actual video latent once across all arms/checkpoints of this case.
    cache=Path('/data/gaoya/AAA_test_video/Context_Video_bench/Shared/supplement/gap038_triplet/predict_shared_context.pt')
    def encode(self,rgb):
        latent=encode_orig(self,rgb)
        input_sha=digest(rgb[:,:,:8])
        if cache.exists():
            saved=torch.load(cache,map_location=latent.device,weights_only=True)
            assert saved['input_sha']==input_sha
            assert saved['vae_sha']==self.assets['vae_checkpoint_sha256']
            canonical=saved['latent'].to(latent)
            assert canonical.shape==latent.shape
            self.last_context_audit['fresh_vs_shared_max_abs']=float((latent-canonical).abs().max())
        else:
            canonical=latent.detach().clone()
            torch.save(dict(latent=canonical,input_sha=input_sha,vae_sha=self.assets['vae_checkpoint_sha256']),cache)
            self.last_context_audit['fresh_vs_shared_max_abs']=0.
        self.last_context_audit.update(shared_context_cache=str(cache),shared_context_sha256=digest(canonical))
        return canonical.clone()
    BenchmarkModel.encode=encode
    audit=dict(variant=variant,mode=mode,context_seed=seed,trace=[])
    state={}
    def denoise(self,noise,x,timesteps,condition):
        clean=condition.gt_frames[:,:,:2].to(x)
        if not state:
            state['clean']=clean.detach().clone()
            saved=json.loads((Path(audit_file).parent/'inference_config.json').read_text())['training_settings']
            state['training_shift']=saved['shift']
            state['grid_config']=saved.get('context_noise_grid',dict(unshifted_max=1.,unshifted_min=.001,size=1000))
            audit['training_shift']=state['training_shift']
            state['noise']=torch.randn(clean.shape,device=x.device,dtype=torch.float32,
                generator=torch.Generator(device=x.device).manual_seed(seed))
            audit.update(clean_context_sha256=digest(clean),context_noise_sha256=digest(state['noise']),
                future_initial_sha256=digest(noise[:,:,2:]),native_conditional_frame_timestep=float(self.config.conditional_frame_timestep))
        assert torch.equal(clean,state['clean'])
        tf=timesteps.flatten()[0].float()
        ix=(self.sample_scheduler.timesteps.to(tf)-tf).abs().argmin()
        assert abs(float(self.sample_scheduler.timesteps[ix]-tf))<.01
        sf=self.sample_scheduler.sigmas[ix].to(tf)
        # Native UniPC casts sigma*1000 to int64 (fm_solvers_unipc.py:204).
        gap=float(sf*self.rectified_flow.num_train_timesteps-tf)
        assert -.001 <= gap < 1.001, (float(sf),float(tf),gap)
        if variant=='clean':
            result=original(self,noise,x,timesteps,condition)
            sc=torch.zeros_like(sf);tc=float(self.config.conditional_frame_timestep)
            ctx=clean
        else:
            g=state['grid_config'];shift=state['training_shift']
            grid=torch.linspace(g['unshifted_max'],g['unshifted_min'],g['size'],device=x.device)
            grid=shift*grid/(1+(shift-1)*grid)
            sc=choose_sigma(variant,mode,sf,grid);tc=sc*self.rectified_flow.num_train_timesteps
            ctx=((1-sc)*clean.float()+sc*state['noise']).to(x)
            y=x.clone();y[:,:,:2]=ctx
            assert torch.equal(y[:,:,2:],x[:,:,2:])
            frame_t=torch.full((x.shape[0],x.shape[2]),float(tf),device=x.device);frame_t[:,:2]=tc
            cond=replace(condition,condition_video_input_mask_B_C_T_H_W=torch.zeros_like(condition.condition_video_input_mask_B_C_T_H_W))
            result=self.net(x_B_C_T_H_W=y.to(**self.tensor_kwargs),timesteps_B_T=frame_t,**cond.to_dict()).float()
            # Keep the original sampler/decoder context trajectory exactly.
            # This is known analytic velocity, NOT the model's context prediction.
            # Network input is still freshly overwritten above at every call.
            result[:,:,:2]=noise[:,:,:2]-condition.gt_frames[:,:,:2].to(result)
        item=dict(step=int(ix),sigma_future=float(sf),sigma_context=float(sc),t_context=float(tc),
            context_sha256=digest(ctx),mask=1 if variant=='clean' else 0)
        call=len(audit['trace'])
        if call%2: assert item==audit['trace'][-1], 'CFG context mismatch'
        audit['trace'].append(item)
        return result
    def decode(self,sample,*args,**kwargs):
        audit['decoder_context_sha256']=digest(sample[:,:,:2])
        result=decode_orig(self,sample,*args,**kwargs)
        assert len(audit['trace'])==2*len(self.sample_scheduler.timesteps)
        audit.update(status='PASS',calls=len(audit['trace']))
        path=Path(audit_file);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(audit,indent=2))
        return result
    BenchmarkModel.denoise=denoise;BenchmarkModel.decode=decode

class CleanContextAudit:
    """Observe native clean inference without changing its tensors or token roles."""
    def __init__(self,pipe,mode):
        self.pipe=pipe;self.calls=0;self.audit=dict(variant='clean',mode=mode,context_role='native_condition')
    def prepare(self,*args,**kwargs):
        result=self.prepare_orig(*args,**kwargs)
        self.audit.update(clean_context_sha256=digest(result[0][:,:,:2]),future_initial_sha256=digest(result[0][:,:,2:]))
        return result
    def before(self,module,args,kwargs):
        self.calls+=1
    def __enter__(self):
        self.prepare_orig=self.pipe.prepare_latents;self.pipe.prepare_latents=self.prepare
        self.handle=self.pipe.transformer.register_forward_pre_hook(self.before,with_kwargs=True)
        return self
    def __exit__(self,typ,value,tb):
        self.handle.remove();self.pipe.prepare_latents=self.prepare_orig
        if typ is None:
            steps=len(self.pipe.scheduler.timesteps)
            assert self.calls==steps*2
            self.audit.update(status='PASS',steps=steps,calls=self.calls)
