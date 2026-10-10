"""Sampling hooks extracted from the pinned historical implementations. No campaign entrypoints."""
import hashlib, math, json
import torch

def digest(x):
    return hashlib.sha256(x.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()

from diffusers import FlowMatchEulerDiscreteScheduler
def _context_noise_pair(
    scheduler: FlowMatchEulerDiscreteScheduler,
    future_timestep: torch.Tensor,
    future_sigma: torch.Tensor,
    context_noise_scale: float,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve (context timestep, actual sigma, target sigma) for interpolation.

    Endpoint branches deliberately bypass the scheduler lookup.  The terminal
    zero sigma in ``scheduler.sigmas`` has no paired training timestep, so it
    must never be used to implement lambda=0.
    """
    scale = float(context_noise_scale)
    if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
        raise ValueError(f"context_noise_scale must be in [0, 1], got {context_noise_scale!r}")
    if scale == 0.0:
        zero = torch.zeros_like(future_sigma, device=device)
        return torch.zeros_like(future_timestep, device=device), zero, zero
    if scale == 1.0:
        sigma = future_sigma.to(device=device)
        return future_timestep.to(device=device), sigma, sigma

    paired_count = len(scheduler.timesteps)
    if len(scheduler.sigmas) < paired_count:
        raise RuntimeError("scheduler has fewer sigmas than training timesteps")
    paired_sigmas = scheduler.sigmas[:paired_count].to(device=device, dtype=future_sigma.dtype)
    target = future_sigma.to(device=device) * scale
    nearest = torch.argmin((paired_sigmas - target.reshape(-1)[0]).abs())
    context_sigma = paired_sigmas[nearest].reshape_as(future_sigma)
    context_timestep = scheduler.timesteps[nearest].to(device=device, dtype=future_timestep.dtype).reshape_as(future_timestep)
    return context_timestep, context_sigma, target

def training_pair_function():
    return _context_noise_pair, None

class ContextNoise:
    def __init__(self, pipe, scheduler, scale, patch_size, digest=None):
        if not math.isfinite(scale) or not 0 <= scale <= 1:
            raise ValueError('inference context noise scale must be finite and in [0,1]')
        self.pipe, self.scheduler, self.scale = pipe, scheduler, scale
        self.patch = tuple(patch_size)
        self.digest = digest
        self.pair, _ = training_pair_function()
        self.index = 0
        self.trace = []
        self.step_count = 0

    def prepare(self, *args, **kwargs):
        result = self.original_prepare(*args, **kwargs)
        latent, condition, mask, noise = result
        self.condition = condition.detach().clone()
        self.c = int((mask[0, 0, :, 0, 0] == 0).sum())
        assert self.c == 2 and self.patch[0] == 1
        assert torch.all(mask[:, :, :self.c] == 0) and torch.all(mask[:, :, self.c:] == 1)
        self.noise = noise[:, :, :self.c].detach().clone()
        self.latent_dtype = latent.dtype
        return result

    def before(self, module, args, kwargs):
        assert not args
        latent, timestep = kwargs['hidden_states'], kwargs['timestep']
        assert self.index == self.step_count, 'Exactly one conditional forward per sampling step required'
        sigma = self.pipe.scheduler.sigmas[self.index].to(device=latent.device, dtype=self.latent_dtype)
        # Training's lookup returns in its input timestep dtype; do not truncate
        # its FP32 table through UniPC's integer scheduler timestep storage.
        future_t = self.pipe.scheduler.timesteps[self.index].to(device=latent.device, dtype=self.scheduler.timesteps.dtype)
        ct, cs, target = self.pair(self.scheduler, future_t, sigma, self.scale, device=latent.device)
        grid = tuple(n // p for n, p in zip(latent.shape[2:], self.patch))
        space = grid[1] * grid[2]
        assert grid[0] == 13 and timestep.shape == (latent.shape[0], grid[0] * space)
        assert torch.all(timestep[:, :self.c * space] == 0)
        assert torch.all(timestep[:, self.c * space:] == future_t)
        assert torch.equal(latent[:, :, :self.c], self.condition[:, :, :self.c].to(latent))
        x, ts = latent.clone(), timestep.clone()
        context = (1 - cs) * self.condition[:, :, :self.c] + cs * self.noise
        x[:, :, :self.c] = context.to(x)
        ts[:, :self.c * space] = ct.to(ts)
        assert torch.equal(x[:, :, self.c:], latent[:, :, self.c:])
        assert torch.equal(ts[:, self.c * space:], timestep[:, self.c * space:])
        assert torch.isfinite(x).all()
        row = dict(step=self.index, future_sigma=float(sigma), future_timestep=float(future_t),
                   target_context_sigma=float(target), actual_context_sigma=float(cs), context_timestep=float(ct),
                   grid=list(grid), future_unchanged=True)
        if self.digest:
            row['context_sha256'] = self.digest(x[:, :, :self.c])['sha256']
            row['timestep_sha256'] = self.digest(ts)['sha256']
        self.trace.append(row)
        self.index += 1
        return args, dict(kwargs, hidden_states=x, timestep=ts)

    def step(self, model_output, timestep, sample, *args, **kwargs):
        assert not args and kwargs.get('return_dict') is False
        assert self.index == self.step_count + 1
        future = self.original_step(model_output[:, :, self.c:], timestep, sample[:, :, self.c:], **kwargs)[0]
        self.step_count += 1
        return (torch.cat([self.condition[:, :, :self.c], future], dim=2),)

    def __enter__(self):
        self.original_prepare = self.pipe.prepare_latents
        self.original_step = self.pipe.scheduler.step
        self.pipe.prepare_latents = self.prepare
        self.pipe.scheduler.step = self.step
        self.hook = self.pipe.transformer.register_forward_pre_hook(self.before, with_kwargs=True)
        return self

    def __exit__(self, *exc):
        self.hook.remove()
        self.pipe.prepare_latents = self.original_prepare
        self.pipe.scheduler.step = self.original_step

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

def choose_sigma(variant, mode, future, grid):
    if variant == "equal_sigma": return future.clone()
    return grid[(grid - .75 * future).abs().argmin()].reshape_as(future)

class NanoContext:
    def __init__(self, pipe, variant, mode, cfg, seed):
        from diffusers import FlowMatchEulerDiscreteScheduler
        self.pipe,self.variant,self.mode,self.seed=pipe,variant,mode,seed
        self.table=FlowMatchEulerDiscreteScheduler(num_train_timesteps=cfg['num_train_timesteps'],shift=cfg['shift'])
        self.n=cfg['num_train_timesteps'];self.calls=0;self.index=0
        self.audit=dict(variant=variant,mode=mode,context_seed=seed,trace=[])
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
        sc=choose_sigma(self.variant,self.mode,sf,grid)
        tc=sc*self.n
        if self.variant=='equal_sigma':tc=tf
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

def install_predict(variant,mode,seed,audit_file,cache_path,training):
    import json
    from pathlib import Path
    from dataclasses import replace
    from projects.generic_acn75.inference_runtime import BenchmarkModel
    original=BenchmarkModel.denoise;decode_orig=BenchmarkModel.decode
    encode_orig=BenchmarkModel.encode
    # Freeze the actual video latent once across all arms/checkpoints of this case.
    cache=Path(cache_path) if cache_path else None
    def encode(self,rgb):
        latent=encode_orig(self,rgb)
        input_sha=digest(rgb[:,:,:8])
        if cache is not None:
            saved=torch.load(cache,map_location=latent.device,weights_only=True)
            assert saved['input_sha']==input_sha
            assert saved['vae_sha']==self.assets['vae_checkpoint_sha256']
            canonical=saved['latent'].to(latent)
            assert canonical.shape==latent.shape
            self.last_context_audit['fresh_vs_shared_max_abs']=float((latent-canonical).abs().max())
        else:
            canonical=latent.detach().clone()
            # Online encode for a new case; no global cache is written.
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
            saved=training
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
            sc=choose_sigma(variant,mode,sf,grid);tc=tf if variant == "equal_sigma" else sc*self.rectified_flow.num_train_timesteps
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
