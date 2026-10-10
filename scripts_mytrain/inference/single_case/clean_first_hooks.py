"""First forward uses clean conditioning; subsequent forwards rebuild or evolve context."""
import json
from pathlib import Path
import torch
from sampling import ContextNoise,NanoContext,digest,choose_sigma
from initial_only_hooks import WanInitialOnly,NanoInitialOnly

class WanCleanRebuild(ContextNoise):
    output=None
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        pair=self.pair
        self.pair=lambda scheduler,t,sigma,scale,**kw:pair(scheduler,t,sigma,0. if self.index==0 else 1.,**kw)
    def __exit__(self,*exc):
        super().__exit__(*exc)
        if exc[0] is None:
            Path(self.output,'clean_first_audit.json').write_text(json.dumps(dict(status='PASS',protocol='clean_first_rebuild',trace=self.trace),indent=2))

class WanCleanFree(WanInitialOnly):
    def __init__(self,pipe,scheduler,scale,patch_size,digest=None):
        super().__init__(pipe,scheduler,0.,patch_size,digest)
    def prepare(self,*args,**kwargs):
        result=super().prepare(*args,**kwargs)
        self.audit['protocol']='clean_first_free'
        return result

class NanoCleanRebuild(NanoContext):
    output=None
    def before(self,module,args,kw):
        if self.index>0:return super().before(module,args,kw)
        x=kw['vision_tokens'][0]
        assert torch.equal(x[:,:,:2],self.clean.to(x))
        row=dict(step=0,context_sha256=digest(x[:,:,:2]),t_context=None,context_role='native_clean',sigma_context=0.,sigma_future=float(self.pipe.scheduler.sigmas[0]))
        if self.calls%2==0:self.audit['trace'].append(row)
        else:assert row==self.audit['trace'][-1]
        self.calls+=1
        return args,kw
    def __exit__(self,*exc):
        super().__exit__(*exc)
        if exc[0] is None:
            self.audit['protocol']='clean_first_rebuild'
            Path(self.output,'clean_first_audit.json').write_text(json.dumps(self.audit,indent=2))

class NanoCleanFree(NanoInitialOnly):
    def before(self,module,args,kw):
        if self.index>0:return super().before(module,args,kw)
        x=kw['vision_tokens'][0]
        assert torch.equal(x[:,:,:2],self.clean.to(x))
        self.initial=self.clean.to(x).clone()
        self.audit.update(protocol='clean_first_free',initial_sigma=0.,initial_future_sigma=float(self.pipe.scheduler.sigmas[0]))
        row=dict(step=0,context_sha256=digest(x[:,:,:2]),context_timestep=None,future_timestep=float(self.pipe.scheduler.timesteps[0]),carried_state_matches=True,context_role='native_clean')
        if self.calls%2==0:self.audit['trace'].append(row)
        else:assert row==self.audit['trace'][-1]
        self.calls+=1
        # Keep native clean input/role/time. Read all output positions to evolve context.
        return args,dict(kw,vision_mse_loss_indexes=kw['vision_sequence_indexes'])


def install_predict_rebuild(variant,mode,seed,audit_file,cache_path,training):
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
    audit=dict(variant=variant,mode=mode,context_seed=seed,protocol='clean_first_rebuild',trace=[])
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
        if int(ix)==0:
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
            context_sha256=digest(ctx),mask=1 if int(ix)==0 else 0)
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
        path.with_name('clean_first_audit.json').write_text(json.dumps(audit,indent=2))
        return result
    BenchmarkModel.denoise=denoise;BenchmarkModel.decode=decode


def install_predict_free(variant, mode, seed, audit_file, cache_path, training):
    from dataclasses import replace
    from initial_only_hooks import archive
    from projects.generic_acn75.inference_runtime import BenchmarkModel
    assert cache_path is None, 'This experiment uses online encoding'
    state = {}; audit = dict(protocol='clean_first_free', backbone='predict25', trace=[])
    decode_orig = BenchmarkModel.decode
    denoise_orig = BenchmarkModel.denoise

    def denoise(self, noise, x, timesteps, condition):
        tf = timesteps.flatten()[0].float()
        ix = int((self.sample_scheduler.timesteps.to(tf)-tf).abs().argmin())
        sf = self.sample_scheduler.sigmas[ix].to(tf)
        if not state:
            assert ix == 0
            clean = condition.gt_frames[:, :, :2].to(x)
            eps = torch.randn(clean.shape, device=x.device, dtype=torch.float32,
                              generator=torch.Generator(device=x.device).manual_seed(seed))
            g = training.get('context_noise_grid', dict(unshifted_max=1.,unshifted_min=.001,size=1000))
            grid = torch.linspace(g['unshifted_max'],g['unshifted_min'],g['size'],device=x.device)
            shift=training['shift'];grid=shift*grid/(1+(shift-1)*grid)
            sc = torch.zeros_like(sf)
            # Native velocity_fn receives the actual latents tensor (no copy). Initialize it once.
            x[:, :, :2] = ((1-sc)*clean.float()+sc*eps).to(x)
            state.update(initial_t=tf if self.config.conditional_frame_timestep<0 else torch.as_tensor(self.config.conditional_frame_timestep,device=tf.device),
                         expected=x.detach().clone(), steps=0)
            audit.update(initial_sigma=float(sc),initial_future_sigma=float(sf),clean_context_sha256=digest(clean))
            original_step = self.sample_scheduler.step
            def step(model_output,timestep,sample,*args,**kwargs):
                assert torch.equal(sample.squeeze(0),state['expected']), 'Predict sampler state mismatch'
                result=original_step(model_output,timestep,sample,*args,**kwargs)
                state['expected']=result[0].squeeze(0).detach().clone();state['steps']+=1
                return result
            self.sample_scheduler.step=step
        assert torch.equal(x,state['expected']), 'Predict input differs from carried scheduler state'
        tc=state['initial_t'] if ix==0 else tf
        frame_t=torch.full((x.shape[0],x.shape[2]),float(tf),device=x.device);frame_t[:,:2]=tc
        cond=replace(condition,condition_video_input_mask_B_C_T_H_W=torch.zeros_like(condition.condition_video_input_mask_B_C_T_H_W))
        if ix==0:
            old_replace=self.config.denoise_replace_gt_frames
            self.config.denoise_replace_gt_frames=False
            try:result=denoise_orig(self,noise,x,timesteps,condition)
            finally:self.config.denoise_replace_gt_frames=old_replace
        else:
            result=self.net(x_B_C_T_H_W=x.to(**self.tensor_kwargs),timesteps_B_T=frame_t,**cond.to_dict()).float()
        item=dict(step=ix,context_sha256=digest(x[:,:,:2]),context_timestep=float(tc),future_timestep=float(tf),carried_state_matches=True)
        if len(audit['trace'])%2: assert item==audit['trace'][-1]
        audit['trace'].append(item)
        return result  # Keep network predictions on ALL frames, including context.

    def decode(self,sample,*args,**kwargs):
        assert state['steps']==len(self.sample_scheduler.timesteps)
        assert torch.equal(sample,state['expected']), 'Final context was replaced before decoding'
        archive(Path(audit_file).parent,audit,sample)
        Path(audit_file).write_text(json.dumps(audit,indent=2)+'\n')
        return decode_orig(self,sample,*args,**kwargs)
    BenchmarkModel.denoise=denoise;BenchmarkModel.decode=decode
