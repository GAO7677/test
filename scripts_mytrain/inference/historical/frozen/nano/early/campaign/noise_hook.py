"""Fixed Gaussian context sweep for Cosmos, including both CFG forwards."""
import hashlib
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor


def digest(value):
    return hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


class ContextNoise:
    def __init__(self, pipe, scale, train_shift=5., train_timesteps=1000):
        if not 0 <= scale <= 1:
            raise ValueError(scale)
        self.pipe, self.scale, self.n = pipe, scale, train_timesteps
        table = FlowMatchEulerDiscreteScheduler(num_train_timesteps=train_timesteps, shift=train_shift)
        self.sigmas = table.sigmas[:train_timesteps]
        self.times = table.timesteps
        self.step_index = self.calls = 0
        self.trace = []
        self.audit = dict(scale=scale, trace=self.trace)

    def prepare(self, *args, **kwargs):
        # Clone before native prepare: replay the SAME draw without advancing its RNG.
        generator = kwargs['generator']
        clone = torch.Generator(device=generator.device).set_state(generator.get_state())
        result = self.original_prepare(*args, **kwargs)
        latent = result[0]
        assert latent.ndim == 5 and latent.shape[2] == 13
        assert result[5][:2].eq(1).all() and result[5][2:].eq(0).all()
        noise = randn_tensor(latent.shape, generator=clone, device=latent.device, dtype=kwargs['dtype'])
        assert torch.equal(latent[:, :, 2:], noise[:, :, 2:])
        self.clean = latent[:, :, :2].detach().clone()
        self.noise = noise[:, :, :2].detach().clone()
        self.audit.update(initial_noise_sha256=digest(noise), context_noise_sha256=digest(self.noise),
                          clean_context_sha256=digest(self.clean))
        return result

    def before(self, module, args, kwargs):
        assert not args
        forwards = 2 if self.pipe.do_classifier_free_guidance else 1
        assert self.calls // forwards == self.step_index
        self.calls += 1
        if self.scale == 0:
            return args, kwargs  # Bitwise-native endpoint: no changes at all.
        x = kwargs['vision_tokens'][0]
        sigma = self.pipe.scheduler.sigmas[self.step_index].to(device=x.device, dtype=torch.float32)
        future_t = self.pipe.scheduler.timesteps[self.step_index].to(device=x.device, dtype=torch.float32)
        if self.scale == 1:
            cs, ct = sigma, future_t
        else:
            grid = self.sigmas.to(sigma)
            nearest = (grid - sigma * self.scale).abs().argmin()
            cs, ct = grid[nearest], self.times.to(sigma)[nearest]
        context = ((1 - cs) * self.clean.float() + cs * self.noise.float()).to(x)
        noisy = x.clone()
        noisy[:, :, :2] = context
        assert torch.equal(noisy[:, :, 2:], x[:, :, 2:])
        t, h, w = kwargs['vision_token_shapes'][0]
        stride = h * w
        levels = torch.full((t * stride,), float(future_t), device=x.device, dtype=torch.float32)
        levels[:2 * stride] = ct
        if (self.calls - 1) % forwards == 0:
            self.trace.append(dict(step=self.step_index, future_sigma=float(sigma), context_sigma=float(cs),
                context_timestep=float(ct), target_sigma=float(sigma * self.scale), context_sha256=digest(context)))
        else:
            assert digest(context) == self.trace[-1]['context_sha256'], 'CFG context mismatch'
        return args, dict(kwargs, vision_tokens=[noisy], vision_timesteps=levels,
                          vision_noisy_frame_indexes=[torch.arange(t, device=x.device)],
                          vision_mse_loss_indexes=kwargs['vision_sequence_indexes'])

    def step(self, model_output, timestep, sample, *args, **kwargs):
        forwards = 2 if self.pipe.do_classifier_free_guidance else 1
        assert self.calls == forwards * (self.step_index + 1)
        result = self.original_step(model_output, timestep, sample, *args, **kwargs)
        if self.scale:
            # Native mask zeroes context velocity; explicitly restore its clean values.
            result[0][..., :2, :, :] = self.clean
        self.step_index += 1
        return result

    def __enter__(self):
        self.original_prepare = self.pipe.prepare_latents
        self.original_step = self.pipe.scheduler.step
        self.pipe.prepare_latents = self.prepare
        self.pipe.scheduler.step = self.step
        self.handle = self.pipe.transformer.register_forward_pre_hook(self.before, with_kwargs=True)
        return self

    def __exit__(self, kind, value, tb):
        self.handle.remove()
        self.pipe.prepare_latents = self.original_prepare
        self.pipe.scheduler.step = self.original_step
        if kind is None:
            assert self.step_index == len(self.pipe.scheduler.timesteps)
            self.audit.update(steps=self.step_index, forwards=self.calls, status='PASS')
