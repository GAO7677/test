"""Scoped inferv2 extension; no shared source edits or attention replacement."""
import math
import torch
from common import training_pair_function


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
