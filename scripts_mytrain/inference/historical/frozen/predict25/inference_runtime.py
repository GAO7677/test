"""Native NVIDIA inference, with local CUDA loading and benchmark input adapters.

The sampler, CFG, condition masks, denoising and decoder are inherited unchanged.
"""
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file

from cosmos_predict2.config import DEFAULT_CHECKPOINT
from cosmos_predict2._src.imaginaire.lazy_config import instantiate, LazyCall as L
from cosmos_predict2._src.imaginaire.utils.config_helper import override
from cosmos_predict2._src.imaginaire.utils import misc
from cosmos_predict2._src.predict2.configs.video2world.config import make_config
from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
from cosmos_predict2._src.predict2.models.video2world_model_rectified_flow import Video2WorldModelRectifiedFlow
from projects.generic_acn75.model import CudaTextEncoder
from projects.generic_acn75.text import LocalProcessor
from projects.generic_acn75.inference_io import normalize_adapter_keys, sha


class BenchmarkModel(Video2WorldModelRectifiedFlow):
    def __init__(self, config, assets, io):
        self.assets = assets
        self.benchmark_io = io
        super().__init__(config)

    def build_net(self, keep_on_cpu=False):
        assert not keep_on_cpu and not self.config.ema.enabled
        with torch.device('meta'):
            net = instantiate(self.config.net)
        net.to_empty(device='cuda')
        net.init_weights()
        net.to(dtype=self.precision)
        state = torch.load(self.assets['native_checkpoint'], map_location='cuda', weights_only=True)
        state = {k.removeprefix('net.'): v for k, v in state.items()}
        net.load_state_dict(state, strict=True)
        del state
        return net

    @torch.no_grad()
    def encode(self, state):
        # The input adapter supplies a padded RGB tensor solely to specify output
        # shape. Encode exactly the observed prefix; never encode padded future.
        n = self.benchmark_io['context_rgb_frames']
        context = super().encode(state[:, :, :n].contiguous())
        keep = self.benchmark_io['context_latents']
        assert context.shape[2] >= keep
        total = self.tokenizer.get_latent_num_frames(state.shape[2])
        packed = context.new_zeros(context.shape[0], context.shape[1], total, *context.shape[3:])
        packed[:, :, :keep] = context[:, :, :keep]
        self.last_context_audit = dict(encoded_rgb_frames=n, encoded_latent_shape=list(context.shape),
                                       kept_latent_frames=keep, packed_shape=list(packed.shape))
        return packed


def native_config(settings):
    assets = settings['assets']
    # Match Video2WorldInference's public-release config composition: NVIDIA's
    # internal training dataset is not shipped and is irrelevant to inference.
    config = override(make_config(), ['--', f'experiment={DEFAULT_CHECKPOINT.experiment}', '~data_train'])
    config.defaults = []
    mc = config.model.config
    OmegaConf.set_struct(mc, False)
    mc.ema.enabled = False
    mc.fsdp_shard_size = 1
    mc.use_lora = False  # Attach verified adapters after loading the base.
    mc.state_t = (settings['output']['frames'] - 1) // 4 + 1
    mc.tokenizer.chunk_duration = settings['output']['frames']
    mc.tokenizer.vae_pth = assets['vae_checkpoint']
    mc.text_encoder_config.compute_online = False  # CUDA loader is attached below.
    mc.text_encoder_config.model_config.tokenizer = L(LocalProcessor)(path=assets['text_weights'])
    mc.text_encoder_config.model_config.model_config.name_or_path = assets['text_weights']
    return config


class BenchmarkInference(Video2WorldInference):
    def __init__(self, settings, training, adapter, fps, seed):
        assets = settings['assets']
        for key in ['native_checkpoint', 'vae_checkpoint']:
            assert sha(assets[key]) == assets[key + '_sha256'], f'Incorrect {key}'
        config = native_config(settings)
        mc = config.model.config
        misc.set_random_seed(seed=seed, by_rank=True)
        torch.backends.cudnn.deterministic = config.trainer.cudnn.deterministic
        torch.backends.cudnn.benchmark = config.trainer.cudnn.benchmark
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True
        self.model = BenchmarkModel(mc, assets, settings['input'])
        self.model.on_train_start()  # Official inference dtype handling.
        self.weight_audit = dict(base=assets['native_checkpoint'], base_sha256=assets['native_checkpoint_sha256'],
                                 adapter=None, lora_tensors=0, load_device='cuda')
        if adapter is not None:
            state = normalize_adapter_keys(load_file(str(adapter / 'adapter.safetensors'), device='cuda:0'))
            self.model.net = self.model.add_lora(
                self.model.net, lora_rank=training['lora_rank'], lora_alpha=training['lora_alpha'],
                lora_target_modules=training['lora_target_modules'], init_lora_weights=True, use_dora=False)
            # Match official inference precision, including LoRA tensors.
            self.model.net.to(dtype=self.model.precision)
            expected = get_peft_model_state_dict(self.model.net)
            assert state.keys() == expected.keys(), dict(missing=sorted(expected.keys()-state.keys()),
                                                        unexpected=sorted(state.keys()-expected.keys()))
            assert all(state[k].shape == expected[k].shape for k in state)
            result = set_peft_model_state_dict(self.model.net, state)
            assert not result.unexpected_keys and not any('lora_' in k for k in result.missing_keys)
            loaded = get_peft_model_state_dict(self.model.net)
            assert all(torch.equal(loaded[k], state[k].to(loaded[k])) for k in state)
            assert any(torch.count_nonzero(v).item() for k,v in state.items() if 'lora_B' in k)
            mc.use_lora = True
            self.weight_audit.update(adapter=str(adapter), adapter_sha256=sha(adapter/'adapter.safetensors'),
                                     lora_tensors=len(state), all_loaded_equal_after_official_dtype_cast=True,
                                     rank=training['lora_rank'], alpha=training['lora_alpha'],
                                     target_modules=training['lora_target_modules'])
            del state, expected, loaded
        else:
            assert not any('lora_' in n for n, _ in self.model.net.named_parameters())
        self.model.text_encoder = CudaTextEncoder(mc.text_encoder_config, assets['text_weights'])
        self.model.eval().requires_grad_(False)
        self.config = config
        self.batch_size = 1
        self.neg_t5_embeddings = None
        self.offload_diffusion_model = self.offload_text_encoder = self.offload_tokenizer = False
        self.context_parallel_size = 1
        self.process_group = None
        self.fps = fps
        self.weight_audit.update(status='PASS', dtype=str(self.model.precision), online_encoders=True)

    def _get_data_batch_input(self, *args, **kwargs):
        batch = super()._get_data_batch_input(*args, **kwargs)
        # FPS is benchmark input metadata. All sampler parameters stay native.
        batch['fps'].fill_(self.fps)
        return batch
