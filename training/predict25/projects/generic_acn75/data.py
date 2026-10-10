"""Official VideoDataset with deterministic prefix selection and fail-fast loading."""
from pathlib import Path
import hashlib,json
from safetensors.torch import load_file
from safetensors import safe_open
import numpy as np
import torch
from decord import VideoReader, cpu
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import VideoDataset

class PrefixVideoDataset(VideoDataset):
    def __init__(self,*args,limit=None,**kwargs):
        super().__init__(*args,**kwargs)
        if limit is not None:self.video_paths=self.video_paths[:limit]

    def _load_video(self, video_path):
        reader = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        if len(reader) < self.sequence_length:
            raise ValueError(f'{video_path}: {len(reader)} < {self.sequence_length}')
        return reader.get_batch(list(range(self.sequence_length))).asnumpy(), reader.get_avg_fps()

    def __getitem__(self, index):
        # Do not silently substitute another sample when the selected one is invalid.
        frames, fps = self._get_frames(self.video_paths[index])
        video = frames.permute(1, 0, 2, 3)
        caption = (Path(self.caption_dir) / (Path(self.video_paths[index]).stem + '.txt')).read_text().strip()
        assert caption and round(fps) == 24
        _, _, h, w = video.shape
        return dict(video=video, ai_caption=caption, fps=fps,
                    image_size=torch.tensor([h,w,h,w]), num_frames=self.sequence_length,
                    padding_mask=torch.zeros(1,h,w))


class CachedVideoDataset(torch.utils.data.Dataset):
    """Read the two official encoder outputs; never decode video during training."""
    def __init__(self, cache_root, limit=None):
        self.root = Path(cache_root)
        self.contract = json.loads((self.root/'contract.json').read_text())
        self.rows = json.loads((self.root/'manifest.json').read_text())
        if limit is not None:
            self.rows = self.rows[:limit]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        values = {}
        for kind, field in [('latent', 'latent_file'), ('embedding', 'text_file')]:
            path = self.root/row[field]
            with safe_open(str(path), framework='pt') as handle:
                assert handle.metadata()['fingerprint'] == self.contract['fingerprint']
                assert handle.metadata()['source_sha256'] == row['video_sha256' if kind=='latent' else 'caption_sha256']
                values[kind] = handle.get_tensor(kind)
        h,w = self.contract['contract']['video_size']
        assert list(values['latent'].shape) == [16,13,h//8,w//8]
        assert list(values['embedding'].shape) == [512,100352]
        return dict(latent=values['latent'], ai_caption=row['caption'], fps=row['fps'],
                    t5_text_embeddings=values['embedding'], t5_text_mask=torch.ones(512),
                    image_size=torch.tensor([h,w,h,w]), num_frames=49,
                    padding_mask=torch.zeros(1,h,w))


class CachedLatentInput:
    """Weightless cache boundary for the official model's tokenizer interface."""
    latent_ch = 16
    spatial_compression_factor = 8

    def encode(self, state):
        assert state.ndim == 5 and state.shape[1] == self.latent_ch
        assert state.dtype == torch.bfloat16
        return state

    def decode(self, state):
        raise RuntimeError('Cache-only training does not load a video decoder')
