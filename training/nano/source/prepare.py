"""Build immutable Cosmos video latents and token IDs for the exact 2175 selection."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer
from diffusers import AutoencoderKLWan
from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import _SYSTEM_PROMPT_VIDEO

from common import CFG, CONFIG_PATH, MODEL, DATA, CACHE, OUTPUT, SELECTION, sha, write_json


def manifest():
    tokenizer = AutoTokenizer.from_pretrained(MODEL / 'text_tokenizer', local_files_only=True)
    config = dict(
        schema=1, selection_sha256=sha(SELECTION), count=CFG['data']['count'], frames=CFG['data']['frames'],
        height=CFG['data']['height'], width=CFG['data']['width'], fps=CFG['data']['fps'], resize='stretch_bilinear_uint8',
        window='prefix', encode_dtype='bfloat16', latent_dtype='bfloat16', posterior='mode',
        vae_sha256=sha(MODEL / 'vae/diffusion_pytorch_model.safetensors'),
        vae_config=json.loads((MODEL / 'vae/config.json').read_text()),
        text_tokenizer_sha256=sha(MODEL / 'text_tokenizer/tokenizer.json'),
        prompt_field=CFG['data']['caption_field'], prompt_system=_SYSTEM_PROMPT_VIDEO,
        add_resolution_template=False, add_duration_template=False,
    )
    prior = {d['logical_key']:d['caption_sha256'] for line in
             (DATA/'my_process/train_cache/prompt_embeddings_initial0907/index.jsonl').read_text().splitlines()
             if (d := json.loads(line))}
    records = []
    for line in SELECTION.read_text().splitlines():
        case = json.loads(open(line).read())
        assert case['family_key'] == 'generic'
        root = DATA / case['logical_key']
        caption = json.loads((root / 'metadata.json').read_text())[CFG['data']['caption_field']]
        assert isinstance(caption, str) and caption.strip()
        if CFG['data']['caption_field'] == 'caption_initial0907':
            assert hashlib.sha256(caption.encode()).hexdigest() == prior[case['logical_key']]
        tokens = tokenizer.apply_chat_template(
            [{'role': 'system', 'content': _SYSTEM_PROMPT_VIDEO}, {'role': 'user', 'content': caption}],
            tokenize=True, add_generation_prompt=True, add_vision_id=False, return_dict=True,
        ).input_ids + [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|vision_start|>')]
        assert len(tokens) <= 2048
        records.append(dict(key=case['logical_key'], video=str(root / 'video.mp4'),
                            caption=caption, input_ids=tokens, source_sha256=sha(root / 'video.mp4'),
                            latent_file=f'latents/{len(records):05d}.safetensors'))
    assert len(records) == CFG['data']['count'] and len({r['key'] for r in records}) == CFG['data']['count']
    CACHE.mkdir(parents=True, exist_ok=True)
    for name, obj in [('config.json', config), ('manifest.json', records)]:
        target = CACHE / name
        if target.exists() and json.loads(target.read_text()) != obj:
            raise RuntimeError(f'Cache provenance changed: {target}')
        write_json(target, obj)
    return config, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=CONFIG_PATH)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(CFG['runtime']['prepare_gpu'])
    torch.set_num_threads(CFG['runtime']['cpu_threads'])
    config, records = manifest()
    fingerprint = sha(CACHE / 'config.json')
    selected = records[:args.limit] if args.limit else records
    (CACHE / 'latents').mkdir(exist_ok=True)
    vae = None
    start = time.time()
    for i, record in enumerate(selected):
        path = CACHE / record['latent_file']
        if path.exists():
            with safe_open(path, framework='pt') as f:
                assert f.metadata()['config_sha256'] == fingerprint
                assert f.metadata()['source_sha256'] == record['source_sha256']
                assert f.get_slice('latent').get_shape() == [48, (CFG['data']['frames']-1)//4+1, CFG['data']['height']//16, CFG['data']['width']//16]
            continue
        if vae is None:
            vae = AutoencoderKLWan.from_pretrained(MODEL / 'vae', torch_dtype=torch.bfloat16,
                                                  local_files_only=True, device_map={'': 'cuda:0'}).eval().requires_grad_(False)
        with imageio.get_reader(record['video']) as reader:
            assert round(reader.get_meta_data()['fps']) == CFG['data']['fps']
            frames = np.stack([reader.get_data(j) for j in range(CFG['data']['frames'])])
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        x = F.interpolate(x, size=(CFG['data']['height'], CFG['data']['width']), mode='bilinear', align_corners=False).clamp(0, 255).to(torch.uint8)
        x = (x.float() / 255 - 0.5) / 0.5
        x = x.permute(1, 0, 2, 3)[None].to(device='cuda', dtype=torch.bfloat16)
        with torch.inference_mode():
            mu = vae.encode(x).latent_dist.mode()
            mean = torch.tensor(vae.config.latents_mean, device='cuda', dtype=mu.dtype).view(1, -1, 1, 1, 1)
            inv = 1 / torch.tensor(vae.config.latents_std, device='cuda', dtype=mu.dtype).view(1, -1, 1, 1, 1)
            latent = ((mu - mean) * inv)[0].cpu().contiguous()
        assert list(latent.shape) == [48, (CFG['data']['frames']-1)//4+1, CFG['data']['height']//16, CFG['data']['width']//16] and torch.isfinite(latent).all()
        tmp = path.with_suffix('.tmp')
        save_file({'latent': latent}, str(tmp), metadata=dict(config_sha256=fingerprint,
                  source_sha256=record['source_sha256'], key=record['key']))
        tmp.replace(path)
        if i % 25 == 0 or i + 1 == len(selected):
            status = dict(done=i+1, requested=len(selected), total=CFG['data']['count'], elapsed_seconds=time.time()-start)
            write_json(OUTPUT / 'cache_progress.json', status)
            print(status, flush=True)
    if not args.limit:
        write_json(CACHE / 'complete.json', dict(count=len(records), config_sha256=fingerprint,
                   manifest_sha256=sha(CACHE / 'manifest.json')))


if __name__ == '__main__':
    main()
