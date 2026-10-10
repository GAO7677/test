"""A/C/N75 extension of NVIDIA's native RF model; no custom training loop.

Model, tokenizer, conditioning, RF time sampler, interpolation, PEFT injection,
optimizer and trainer are provided by the official repository.
"""
from dataclasses import replace
from projects.generic_acn75.mechanism import pack_context
from pathlib import Path
import json,hashlib
import attrs
import torch
from safetensors.torch import load_file
from cosmos_predict2._src.imaginaire.lazy_config import instantiate
from cosmos_predict2._src.predict2.models.video2world_model_rectified_flow import (
    Video2WorldModelRectifiedFlow, Video2WorldModelRectifiedFlowConfig)
from cosmos_predict2._src.predict2.text_encoders.text_encoder import TextEncoder

@attrs.define(slots=False)
class MechanismConfig(Video2WorldModelRectifiedFlowConfig):
    mechanism: str = 'A'
    context_frames: int = 2
    context_noise_scales: dict = attrs.field(factory=lambda: {'A':1., 'C':0., 'N75':.75})
    context_noise_grid: dict = attrs.field(factory=lambda: {'unshifted_max':1.,'unshifted_min':.001,'size':1000})
    checkpoint_model_storage: str = 'lora_and_training_counters'
    native_checkpoint: str = ''
    native_checkpoint_sha256: str = ''
    text_weights: str = ''
    audit_directory: str = ''
    preserve_fp32_lora: bool = True

class CudaTextEncoder(TextEncoder):
    """Use the official encoder and embedding function, loading tensors on CUDA."""
    def __init__(self, config, weights):
        self.config = config
        self.device = 'cuda'
        with torch.device('meta'):
            self.model = instantiate(config.model_config)
        self.model.to_empty(device='cuda')
        assert not torch.distributed.is_initialized(), 'Prepare frozen text embeddings outside DDP'
        with torch.no_grad():self.model.init_weights()
        expected={k for k,v in self.model.state_dict().items() if isinstance(v,torch.Tensor)}
        loaded=set()
        for path in sorted(Path(weights).glob('model-*.safetensors')):
            state=load_file(str(path),device=f'cuda:{torch.cuda.current_device()}')
            result=self.model.load_state_dict(state,strict=False)
            assert not result.unexpected_keys,result
            loaded.update(state)
            del state
        assert loaded==expected,dict(missing=sorted(expected-loaded),extra=sorted(loaded-expected))
        self.model.eval().requires_grad_(False)
        torch.cuda.empty_cache()

class MechanismModel(Video2WorldModelRectifiedFlow):
    def __init__(self, config):
        assert config.mechanism in config.context_noise_scales
        assert config.fsdp_shard_size == 1, 'This configuration uses official DDP, without context parallelism.'
        assert not config.text_encoder_config.compute_online
        assert config.input_data_key == 'latent'
        super().__init__(config)
        self._audit_written = False

    def build_net(self, keep_on_cpu=False):
        assert not keep_on_cpu and not self.config.ema.enabled
        # A local consolidated checkpoint is loaded directly on CUDA. Official
        # generic checkpoint loader otherwise stages this file on CPU.
        with torch.device('meta'):
            net = instantiate(self.config.net)
        net.to_empty(device='cuda')
        net.init_weights()
        net.to(dtype=self.precision)
        digest=hashlib.sha256()
        with open(self.config.native_checkpoint,'rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
        assert digest.hexdigest()==self.config.native_checkpoint_sha256
        state = torch.load(self.config.native_checkpoint,
                           map_location=f'cuda:{torch.cuda.current_device()}', weights_only=True)
        state = {k.removeprefix('net.'):v for k,v in state.items()}
        result = net.load_state_dict(state, strict=False)
        assert not result.missing_keys and not result.unexpected_keys, result
        del state
        for name,buffer in net.named_buffers():
            if name.startswith('accum_'):buffer.zero_()
        net = self.add_lora(net, lora_rank=self.config.lora_rank,
                            lora_alpha=self.config.lora_alpha,
                            lora_target_modules=self.config.lora_target_modules,
                            init_lora_weights=self.config.init_lora_weights,
                            use_dora=self.config.use_dora)
        return net

    def on_train_start(self, memory_format=torch.preserve_format):
        super().on_train_start(memory_format)
        if self.config.preserve_fp32_lora:
            for name,p in self.net.named_parameters():
                if p.requires_grad:
                    assert 'lora_' in name, name
                    p.data = p.data.float()

    def _normalize_video_databatch_inplace(self, data_batch, input_key=None):
        # Latents have already passed the official RGB normalization and VAE.
        assert (input_key or self.input_data_key) == 'latent'
        assert data_batch['latent'].dtype == torch.bfloat16
        assert 'video' not in data_batch

    def forward(self, data_batch):
        # Fresh native raw text embeddings are supplied through the official
        # dataset interface; native DiT still executes its frozen projection.
        assert data_batch['t5_text_embeddings'].shape[-1] == 100352
        _, clean, condition = self.get_data_and_condition(data_batch)
        b, _, frames, _, _ = clean.shape
        n = self.config.context_frames
        noise = torch.randn(clean.shape, **self.tensor_kwargs_fp32)
        u = self.rectified_flow.sample_train_time(b)
        timestamps = self.rectified_flow.get_discrete_timestamp(u, self.tensor_kwargs_fp32)
        sigma = self.rectified_flow.get_sigmas(timestamps, self.tensor_kwargs_fp32)
        mixed, target, frame_sigma, mask = pack_context(
            self.rectified_flow,clean,noise,sigma,n,self.config.mechanism,self.config.context_noise_scales,self.config.context_noise_grid)
        context_sigma=frame_sigma[:,0]
        condition = replace(condition, condition_video_input_mask_B_C_T_H_W=mask)
        # Use native network directly: official denoise() replaces context with
        # clean GT, which would erase A/N75's intentionally noisy context.
        with torch.autocast('cuda', dtype=self.precision):
            prediction = self.net(x_B_C_T_H_W=mixed.to(**self.tensor_kwargs),
                                  timesteps_B_T=frame_sigma * self.rectified_flow.num_train_timesteps,
                                  **condition.to_dict()).float()
        loss = (prediction[:,:,n:] - target[:,:,n:]).square().mean()
        assert torch.isfinite(loss), 'Non-finite training loss'
        if not self._audit_written:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            out = Path(self.config.audit_directory);out.mkdir(parents=True,exist_ok=True)
            trainable = [(k,p.numel(),str(p.dtype)) for k,p in self.net.named_parameters() if p.requires_grad]
            audit = dict(mechanism=self.config.mechanism, latent_shape=list(clean.shape),
                         text_shape=list(data_batch['t5_text_embeddings'].shape),
                         trainable=trainable, trainable_count=sum(x[1] for x in trainable),
                         frozen_count=sum(p.numel() for p in self.net.parameters() if not p.requires_grad),
                         native_checkpoint=self.config.native_checkpoint, video_cache_used=True,text_cache="native_raw_embeddings",
                         sigma=sigma.tolist(),context_sigma=context_sigma.tolist(), loss=float(loss.detach()))
            (out/f'first_forward_rank{rank}.json').write_text(json.dumps(audit,indent=2))
            self._audit_written=True
        return dict(x0=clean, xt=mixed, sigma=sigma[:,None], condition=condition,
                    model_pred=prediction, edm_loss=loss, timesteps=timestamps,
                    per_instance_loss=(prediction[:,:,n:]-target[:,:,n:]).square().mean(dim=(1,2,3,4)),
                    n_cond_frames=condition.num_conditional_frames_B), loss
