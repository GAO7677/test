import ast
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = Path('/home/gaoya/code_V2V_baselines/PhysRVG-main')
OLD = Path('/data/gaoya/agent-data/outputs/context_attention_replay_20260927')
TRAIN = REPO / 'scripts_mytrain/train/train_full_sa_pybullet_aux.py'
PIPE = REPO / 'fastvideo/models/wan_v2v/pipeline_wan_v2v.py'
PYTHON = '/data/gaoya/agent-data/envs/physrvg-full-sa/bin/python'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def training_pair_function():
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler
    source = TRAIN.read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == '_context_noise_pair')
    function_source = ast.get_source_segment(source, node)
    namespace = dict(torch=torch, math=math, FlowMatchEulerDiscreteScheduler=FlowMatchEulerDiscreteScheduler)
    exec(compile(function_source, str(TRAIN), 'exec'), namespace)
    return namespace['_context_noise_pair'], function_source


def training_scheduler(resolved, device='cpu'):
    from diffusers import FlowMatchEulerDiscreteScheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(resolved['base']['model_id'], subfolder='scheduler', local_files_only=True)
    scheduler.set_timesteps(resolved['num_train_timesteps'], device=device)
    return scheduler
