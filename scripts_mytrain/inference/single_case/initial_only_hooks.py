"""Initialize context once, then evolve all frames with the native scheduler."""
import json
from pathlib import Path
import torch
from sampling import ContextNoise, NanoContext, choose_sigma, digest


def archive(out, audit, final):
    out = Path(out)
    assert torch.isfinite(final).all()
    torch.save(final.detach().cpu(), out/'free_final_latent.pt')
    audit.update(status='PASS', final_context_sha256=digest(final[..., :2, :, :]),
                 final_latent_sha256=digest(final), context_replacements_after_initialization=0)
    (out/'initial_only_audit.json').write_text(json.dumps(audit, indent=2)+'\n')


class WanInitialOnly(ContextNoise):
    output = None

    def prepare(self, *args, **kwargs):
        latent, condition, mask, noise = super().prepare(*args, **kwargs)
        sf = self.pipe.scheduler.sigmas[0].to(latent)
        tf = self.pipe.scheduler.timesteps[0].to(device=latent.device, dtype=self.scheduler.timesteps.dtype)
        tc, sc, target = self.pair(self.scheduler, tf, sf, self.scale, device=latent.device)
        self.initial_t = tc
        latent = latent.clone()
        latent[:, :, :2] = ((1-sc)*condition[:, :, :2]+sc*noise[:, :, :2]).to(latent)
        self.expected = latent.detach().clone()
        self.audit = dict(protocol='initial_only', backbone='wan', initial_sigma=float(sc),
                          initial_target_sigma=float(target), initial_future_sigma=float(sf),
                          clean_context_sha256=digest(condition[:, :, :2]), trace=[])
        # Both forward blending and final blending become identity; token timesteps are noisy.
        return latent, condition, torch.ones_like(mask), noise

    def before(self, module, args, kw):
        x = kw['hidden_states']; ts = kw['timestep'].clone()
        assert torch.equal(x, self.expected.to(x)), 'Wan input differs from carried scheduler state'
        if self.step_count == 0:
            space = (x.shape[3]//self.patch[1])*(x.shape[4]//self.patch[2])
            ts[:, :2*space] = self.initial_t.to(ts)
        self.trace.append(dict(step=self.step_count, context_sha256=digest(x[:, :, :2]),
                               context_timestep=float(ts[0,0]), future_timestep=float(ts[0,-1]),
                               carried_state_matches=True))
        return args, dict(kw, timestep=ts)

    def step(self, model_output, timestep, sample, *args, **kwargs):
        assert torch.equal(sample, self.expected)
        result = self.original_step(model_output, timestep, sample, *args, **kwargs)
        self.expected = result[0].detach().clone()
        self.step_count += 1
        return result

    def __exit__(self, *exc):
        super().__exit__(*exc)
        if exc[0] is None:
            assert self.step_count == len(self.pipe.scheduler.timesteps)
            self.audit['trace'] = self.trace
            archive(self.output, self.audit, self.expected)


class NanoInitialOnly(NanoContext):
    output = None

    def before(self, module, args, kw):
        assert not args and self.calls//2 == self.index
        x = kw['vision_tokens'][0]
        sf = self.pipe.scheduler.sigmas[self.index].to(x.device).float()
        tf = self.pipe.scheduler.timesteps[self.index].to(x.device).float()
        if self.index == 0:
            sc = choose_sigma(self.variant, self.mode, sf, self.table.sigmas[:self.n].to(sf))
            if self.calls == 0:
                self.initial = ((1-sc)*self.clean.float()+sc*self.noise).to(x)
                self.audit.update(protocol='initial_only', initial_sigma=float(sc), initial_future_sigma=float(sf))
            y = x.clone(); y[:, :, :2] = self.initial
            tc = tf if self.variant == 'equal_sigma' else sc*self.n
        else:
            assert torch.equal(x, self.expected.to(x)), 'Nano input differs from carried scheduler state'
            y = x; tc = tf
        t,h,w = kw['vision_token_shapes'][0]
        levels = torch.full((t*h*w,), float(tf), device=x.device); levels[:2*h*w] = tc
        item = dict(step=self.index, context_sha256=digest(y[:, :, :2]), context_timestep=float(tc),
                    future_timestep=float(tf), carried_state_matches=self.index>0)
        if self.calls%2 == 0: self.audit['trace'].append(item)
        else: assert item == self.audit['trace'][-1]
        self.calls += 1
        return args, dict(kw, vision_tokens=[y], vision_timesteps=levels,
                         vision_noisy_frame_indexes=[torch.arange(t,device=x.device)],
                         vision_mse_loss_indexes=kw['vision_sequence_indexes'])

    def step(self, model_output, timestep, sample, *args, **kwargs):
        assert self.calls == 2*(self.index+1)
        if self.index == 0:
            sample = sample.clone(); sample[..., :2, :, :] = self.initial.to(sample)
        else: assert torch.equal(sample.squeeze(0), self.expected)
        result = self.step_orig(model_output, timestep, sample, *args, **kwargs)
        self.expected = result[0].squeeze(0).detach().clone()
        self.index += 1
        return result

    def __enter__(self):
        super().__enter__()
        self.mask_orig = self.pipe._mask_velocity_predictions
        def full_velocity(preds_vision, preds_sound, vision_condition_mask, *args, **kwargs):
            result = self.mask_orig(preds_vision, preds_sound, vision_condition_mask, *args, **kwargs)
            return (preds_vision[0], *result[1:])
        self.pipe._mask_velocity_predictions = full_velocity
        return self

    def __exit__(self, *exc):
        self.pipe._mask_velocity_predictions = self.mask_orig
        super().__exit__(*exc)
        if exc[0] is None: archive(self.output, self.audit, self.expected)


def install_predict_initial(variant, mode, seed, audit_file, cache_path, training):
    from dataclasses import replace
    from projects.generic_acn75.inference_runtime import BenchmarkModel
    assert cache_path is None, 'This experiment uses online encoding'
    state = {}; audit = dict(protocol='initial_only', backbone='predict25', trace=[])
    decode_orig = BenchmarkModel.decode

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
            sc = choose_sigma(variant,mode,sf,grid)
            # Native velocity_fn receives the actual latents tensor (no copy). Initialize it once.
            x[:, :, :2] = ((1-sc)*clean.float()+sc*eps).to(x)
            state.update(initial_t=tf if variant=='equal_sigma' else sc*self.rectified_flow.num_train_timesteps,
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
