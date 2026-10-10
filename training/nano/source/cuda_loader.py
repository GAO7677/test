"""Materialize only local FSDP shards; deserialize each base tensor directly on CUDA."""
import json
import torch
from safetensors import safe_open
from diffusers import Cosmos3OmniTransformer
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import distribute_tensor
from common import inject_lora, CFG


def load_sharded_cuda(folder, mesh, local_rank, dtype):
    device = torch.device('cuda', local_rank)
    with torch.device('meta'):
        model = Cosmos3OmniTransformer.from_config(Cosmos3OmniTransformer.load_config(folder))
        model.requires_grad_(False)
        inject_lora(model)
        model.to(dtype=dtype)
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=dtype, cast_forward_inputs=False)
    for layer in model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
    model.time_embedder.to(dtype=torch.float32)
    fully_shard(model.time_embedder, mesh=mesh,
                mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32))
    fully_shard(model, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
    model.to_empty(device=device)
    # to_empty also discards nonpersistent buffers. Reconstruct the rotary frequencies.
    buffers = dict(model.named_buffers())
    assert set(buffers) == {'rotary_emb.inv_freq'}, set(buffers)
    cfg = model.config
    model.rotary_emb.inv_freq = 1 / (cfg.rope_theta ** (
        torch.arange(0, cfg.head_dim, 2, device=device, dtype=torch.float32) / cfg.head_dim))
    parameters = dict(model.named_parameters())
    loaded = set()
    index = json.loads((folder / 'diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
    with torch.no_grad():
        for filename in sorted(set(index.values())):
            with safe_open(folder / filename, framework='pt', device=str(device)) as reader:
                for original in reader.keys():
                    name = original
                    if name not in parameters:
                        prefix, attr = name.rsplit('.', 1)
                        name = prefix + '.base_layer.' + attr
                    assert name in parameters, original
                    param = parameters[name]
                    value = reader.get_tensor(original).to(dtype=param.dtype)
                    assert value.device.type == 'cuda' and value.shape == param.shape
                    param.copy_(distribute_tensor(value, mesh, param.placements))
                    loaded.add(name)
                    del value
            print(f'rank={local_rank} loaded CUDA shard file {filename}', flush=True)
        generator = torch.Generator(device=device).manual_seed(CFG['training']['seed'])
        for name, param in parameters.items():
            if name in loaded:
                continue
            assert 'lora_' in name, f'Missing pretrained parameter: {name}'
            value = torch.empty(param.shape, device=device, dtype=param.dtype)
            if '.lora_A.' in name:
                value.normal_(std=1/CFG['lora']['rank'], generator=generator)
            else:
                value.zero_()
            param.copy_(distribute_tensor(value, mesh, param.placements))
            del value
    assert all(p.device.type == 'cuda' for p in parameters.values())
    print(f'rank={local_rank} CUDA load complete: {len(loaded)} base tensors; '
          f'allocated={torch.cuda.memory_allocated()/2**30:.2f} GiB', flush=True)
    return model
