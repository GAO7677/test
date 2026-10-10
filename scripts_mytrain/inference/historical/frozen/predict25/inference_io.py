"""Benchmark input/output helpers. No model or sampling implementation."""
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def read_prefix(path, count=8):
    import av
    import numpy as np
    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format='rgb24'))
            if len(frames) == count:
                break
    if len(frames) != count:
        raise ValueError(f'{path}: expected {count} context frames, got {len(frames)}')
    return np.stack(frames)


def video_info(path):
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return dict(width=stream.width, height=stream.height, frames=stream.frames,
                    fps=float(stream.average_rate), codec=stream.codec_context.name,
                    pix_fmt=stream.pix_fmt)


def normalize_adapter_keys(state):
    """Remove only PyTorch activation-checkpoint wrapper names from PEFT export."""
    out = {}
    for key, value in state.items():
        normalized = key.replace('._checkpoint_wrapped_module.', '.')
        if 'lora_' not in normalized or normalized in out:
            raise ValueError(f'Invalid or duplicate adapter tensor: {key}')
        out[normalized] = value
    return out


def check_adapter_metadata(folder, mode, step, base_checkpoint):
    folder = Path(folder)
    metadata = json.loads((folder / 'metadata.json').read_text())
    if metadata.get('framework') != 'nvidia-native':
        raise ValueError('Only native-framework adapter exports are accepted')
    if metadata.get('mode') != mode or metadata.get('iteration') != step:
        raise ValueError(f'Wrong mode/step: {metadata}')
    if Path(metadata['base_checkpoint']).resolve() != Path(base_checkpoint).resolve():
        raise ValueError('Adapter base checkpoint does not match inference base')
    if not metadata.get('adapter_roundtrip_exact'):
        raise ValueError('Training export did not pass adapter serialization audit')
    if not (folder / 'adapter.safetensors').is_file():
        raise FileNotFoundError(folder / 'adapter.safetensors')
    return metadata
