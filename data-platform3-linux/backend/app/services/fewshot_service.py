"""FewShotLab adapter: immutable inputs, durable queue, image-only imports."""
from __future__ import annotations
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from fastapi import HTTPException
from PIL import Image, ImageOps
from sqlalchemy import select
from ..config import settings
from ..models import FewshotJob, ImageData, DataFile, Dataset, DatasetVersion, AutoLabelTask
from . import file_service as fs, version_service as vs

ACTIVE = ('training', 'generating', 'importing', 'labeling')
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}
ENGINES = {
    'flux2_klein_base_4b': dict(method='beyondobjects', model='flux2_klein_base_4b', name='BeyondObjects + FLUX.2 Klein Base 4B',
                              note='多组素材共同训练一个 LoRA。训练文本最多 512 tokens。'),
    'beyondobjects_sd21': dict(method='beyondobjects', model='sd21', name='BeyondObjects + SD2.1',
                             note='多组素材共同训练一个 LoRA，按完整训练轮次执行。训练文本最多 77 tokens。'),
    'chimera_sd21': dict(method='chimeralora', model='sd21', name='ChimeraLoRA + SD2.1',
                        note='保留原生共享 A、多 B 头结构；每张素材一个 B 头，生成时混合全部头。训练文本最多 77 tokens。'),
}


class GpuMemoryWait(ValueError):
    pass


def engine_id(options):
    method, model = options.get('method', 'beyondobjects'), options.get('model', 'flux2_klein_base_4b')
    for key, spec in ENGINES.items():
        if (spec['method'], spec['model']) == (method, model):
            return key
    raise ValueError('该方法与模型组合不受仓库支持')


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def platform_settings():
    c = dict(settings.data.get('fewshot', {}))
    for name in ('repo', 'output_root', 'gpu'):
        if not c.get(name):
            raise ValueError(f'未配置 fewshot.{name}')
    if not c.get('enabled', False):
        raise ValueError('少样本生成服务尚未启用')
    if str(c['gpu']) in ('4', 'cuda:4', c.get('forbidden_gpu')):
        raise ValueError('禁止使用物理 GPU 4')
    if not str(c['gpu']).startswith('GPU-'):
        raise ValueError('训练 GPU 必须配置物理 GPU UUID，避免编号映射错误')
    if not (Path(c['repo']) / 'run.py').is_file():
        raise ValueError('缺少 FewShotLab run.py')
    if not shutil.which('tmux'):
        raise ValueError('未安装 tmux')
    return c


def engine_settings(backend='flux2_klein_base_4b'):
    if backend not in ENGINES:
        raise ValueError('未知训练方案')
    c = platform_settings()
    override = c.get('backends', {}).get(backend)
    if backend != 'flux2_klein_base_4b' and not override:
        raise ValueError('管理员尚未配置该方案的环境与权重')
    if override:
        for key in ('python', 'base_model', 'gpu_min_free_mb', 'gpu_reserve_mb'):
            if key in override:
                c[key] = override[key]
    for key in ('python', 'base_model'):
        if not c.get(key):
            raise ValueError('未配置运行文件：' + key)
    for p in (Path(c['python']), Path(c['base_model']) / 'model_index.json'):
        if not p.is_file():
            raise ValueError(f'缺少运行文件：{p}')
    c['backend'] = backend
    return c


def job_engine(job):
    return engine_settings(engine_id(json.loads(job.options)))


def training_steps(options, groups):
    import math
    if engine_id(options) == 'beyondobjects_sd21':
        per_epoch = max(1, math.ceil(sum(g['images'] for g in groups) / 4))
        return math.ceil(options['steps'] / per_epoch) * per_epoch
    return options['steps']


def failure_message(job, exc=None):
    failure = read_json(Path(job.run_dir) / 'failure.json', {}) if job.run_dir else {}
    message = str(failure.get('error') or exc or job.error)
    log = Path(job.job_dir) / 'worker.log'
    if log.is_file():
        with log.open('rb') as f:
            f.seek(max(0, log.stat().st_size - 64000))
            recent = f.read().decode('utf-8', errors='replace').rsplit('\nSTAGE ', 1)[-1]
    else:
        recent = ''
    if 'out of memory' in (message + recent).lower() or 'OutOfMemoryError' in message + recent:
        return 'GPU 7 显存不足（CUDA OOM）。共享任务的显存占用发生变化；素材和已有进度已保留，显存充足后可手动重试。'
    if not failure.get('error'):
        lines = [line.strip() for line in recent.splitlines()
                 if any(word in line for word in ('Error:', 'Exception:')) and 'CalledProcessError' not in line]
        if lines:
            message = lines[-1]
    return message[-1500:]


def gpu_memory(c):
    result = subprocess.check_output(['nvidia-smi', '-i', c['gpu'],
                                     '--query-gpu=memory.used,memory.free,memory.total',
                                     '--format=csv,noheader,nounits'], text=True)
    return tuple(int(value.strip()) for value in result.strip().split(','))


def gpu_admission(c, memory):
    used, free, total = memory
    if c.get('allow_shared_gpu', False):
        reserve = int(c.get('gpu_reserve_mb', 2048))
        minimum = int(c.get('gpu_min_free_mb', 24576))
        if not 0 <= reserve < minimum <= total:
            raise ValueError('共享 GPU 的显存门槛配置无效')
        return free >= minimum, min(0.9, max(0.0, (free - reserve) / total))
    return used < 1024, 0.9


def public_job(job):
    progress = read_json(Path(job.run_dir) / 'status.json', {}) if job.run_dir else {}
    options = json.loads(job.options)
    backend = engine_id(options)
    groups = json.loads(job.groups)
    waiting = read_json(Path(job.job_dir) / 'waiting.json', {})
    return dict(id=job.id, name=job.name, project=job.project, scene=job.scene,
                secret_level=job.secret_level, status=job.status, groups=groups,
                options=options, engine=backend, engine_name=ENGINES[backend]['name'],
                effective_steps=training_steps(options, groups), waiting=waiting,
                progress=progress, error=failure_message(job) if job.status == 'failed' else job.error,
                dataset_id=job.dataset_id, version_id=job.version_id, label_task_id=job.label_task_id,
                source_data_id=job.source_data_id, result_data_id=job.result_data_id,
                reuse_job_id=job.reuse_job_id, label_errors=job.label_errors, empty_labels=job.empty_labels,
                created_by=job.created_by, download_request_id=job.download_request_id,
                created_at=job.created_at.isoformat(), smoke=job.smoke)


def get_job(db, jid, cur, write=False):
    job = db.get(FewshotJob, jid)
    if not job:
        raise HTTPException(404, '任务不存在')
    cur.require_view(job.secret_level)
    if job.created_by != cur.id and not cur.unrestricted:
        raise HTTPException(403, '只能访问自己创建的任务')
    if write:
        cur.require_operate(job.secret_level)
        if job.created_by != cur.id and not cur.unrestricted:
            raise HTTPException(403, '只能操作自己创建的任务')
    return job


def new_job_dir(key):
    c = platform_settings()
    day = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d')
    p = Path(c['output_root']) / f'{day}_fewshot_{key}'
    p.mkdir(parents=True, exist_ok=False)
    return p


def image_info(path):
    with Image.open(path) as im:
        if im.width * im.height > 40_000_000:
            raise ValueError('单图不得超过 4000 万像素')
        im = ImageOps.exif_transpose(im).convert('RGB')
        return im.size, hashlib.sha256(im.tobytes() + str(im.size).encode()).hexdigest()


def add_file(db, raw, source, relative):
    dest = fs.dataset_dir('image', raw.id) / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source != dest:
        shutil.copyfile(source, dest)
    db.add(DataFile(dataset_id=raw.id, data_type='image', filename=relative,
                    stored_name=relative, size_bytes=dest.stat().st_size, file_hash=sha(dest),
                    chunk_received=1, status='complete', stage='raw'))
    raw.file_count += 1
    raw.size_bytes += dest.stat().st_size
    return dest


def new_raw(db, job, name, source_type):
    raw = ImageData(name=name, project=job.project, scene=job.scene, secret_level=job.secret_level,
                    source_type=source_type, created_by=job.created_by, file_count=0, size_bytes=0,
                    format='image', status='未标注')
    db.add(raw)
    db.flush()
    raw.file_path = str(fs.dataset_dir('image', raw.id))
    return raw


def ensure_worker():
    """Only the queue worker is detached; HTTP/UI continues in the foreground."""
    platform_settings()
    backend = Path(__file__).resolve().parents[2]
    identity = hashlib.sha256(str(settings.path.resolve()).encode()).hexdigest()[:10]
    session = 'dp-fewshot-' + identity
    if subprocess.run(['tmux', 'has-session', '-t', session], capture_output=True).returncode == 0:
        return
    env = ['env', 'DP_CONFIG_PATH=' + str(settings.path.resolve()), 'PYTHONDONTWRITEBYTECODE=1']
    if os.environ.get('DATABASE_URL'):
        env.append('DATABASE_URL=' + os.environ['DATABASE_URL'])
    cmd = env + [sys.executable, '-m', 'app.fewshot_worker']
    result = subprocess.run(['tmux', 'new-session', '-d', '-s', session, '-c', str(backend), shlex.join(cmd)], capture_output=True, text=True)
    if result.returncode and subprocess.run(['tmux', 'has-session', '-t', session], capture_output=True).returncode:
        raise RuntimeError('无法启动任务队列：' + result.stderr.strip())


def build_config(job, run_dir, parent=None):
    c = job_engine(job)
    memory_fraction = 0.9
    if c.get('allow_shared_gpu', False):
        allowed, memory_fraction = gpu_admission(c, gpu_memory(c))
        if not allowed:
            raise GpuMemoryWait('GPU 剩余显存不足，继续排队')
    opt = json.loads(job.options)
    groups = json.loads(job.groups)
    plan = []
    for group in groups:
        for _ in range(group['count']):
            plan.append(dict(prompt=group['caption'], seed=opt['seed'] + len(plan)))
    plan_path = Path(job.job_dir) / 'generation_plan.json'
    dump(plan_path, plan)
    chimera = c['backend'] == 'chimera_sd21'
    return dict(name=f'platform_{job.id}', model=c['backend'],
                paths=dict(base_model=c['base_model'], output_root=str(run_dir),
                           adapter=str(Path(parent.run_dir) / 'adapter') if parent else None),
                data=dict(root=str(Path(job.job_dir) / 'support')),
                train=dict(rank=opt['rank'], alpha=opt['rank'], steps=training_steps(opt, groups), batch_size=1,
                           gradient_accumulation=1 if chimera else 4, learning_rate=0.0001, seed=opt['seed'],
                           resolution=opt['resolution'], warmup=min(100, max(0, opt['steps'] // 10)),
                           checkpoint_every=opt['steps']+1 if chimera else 100, keep_checkpoints=2,
                           gradient_checkpointing=True),
                generate=dict(num_images=len(plan), plan=str(plan_path), compare_base=False,
                              steps=opt['sampling_steps'], seed=opt['seed']),
                runtime=dict(gpu=c['gpu'], threads=4, memory_fraction=memory_fraction),
                tracking=dict(enabled=not job.smoke, project=c.get('tracking_project', 'data-platform-fewshot')))


def completed_batch(run_dir):
    for p in sorted((Path(run_dir) / 'generations').glob('*'), reverse=True):
        marker = read_json(p / 'complete.json', {})
        if marker.get('passed'):
            return p
    return None


def import_batch(db, job):
    """Validate the complete batch before publishing; a committed import is idempotent."""
    if job.dataset_id:
        return
    batch = Path(job.batch_dir)
    complete = read_json(batch / 'complete.json', {})
    records = read_json(batch / 'index.json', [])
    expected = sum(g['count'] for g in json.loads(job.groups))
    if not complete.get('passed') or complete.get('images') != expected or len(records) != expected:
        raise ValueError('生成批次未完整完成或数量不符，暂不入库')
    images, seen = [], set()
    for record in records:
        path = (batch / record['image']).resolve()
        if record.get('kind') != 'lora' or path.parent != (batch / 'lora').resolve() or path.suffix != '.png':
            raise ValueError('批次包含非 LoRA 图片或非法路径')
        if path.name in seen or sha(path) != record.get('image_sha256'):
            raise ValueError('生成图重复或校验和不匹配')
        image_info(path)
        seen.add(path.name)
        images.append((path, record))
    raw = new_raw(db, job, job.name + ' · 生成图像', 'AIGC')
    ds = Dataset(name=job.name + ' · 待标注', data_type='image', secret_level=job.secret_level,
                 source_data_ids=json.dumps([raw.id]), created_by=job.created_by,
                 description=f'FewShot 任务 #{job.id}；来源素材 #{job.source_data_id}；全部图像待人工核验')
    db.add(ds)
    db.flush()
    out = fs.processed_dir(ds.id, 1)
    v = DatasetVersion(dataset_id=ds.id, version_no=1, status='draft', stage='原始', file_path=str(out),
                       source_task_type='fewshot', source_task_id=job.id, created_by=job.created_by)
    db.add(v)
    db.flush()
    for path, record in images:
        dest = add_file(db, raw, path, path.name)
        shutil.copyfile(dest, out / path.name)
    lineage = dict(fewshot_job_id=job.id, source_data_id=job.source_data_id,
                   reuse_job_id=job.reuse_job_id, source_type='AIGC', generation_records=records)
    dump(out / '_manifest.json', lineage)
    dump(Path(job.job_dir) / 'generation_metadata.json', lineage)
    ds.current_version_id = v.id
    ds.data_count = v.data_count = len(images)
    v.size_bytes = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    job.result_data_id, job.dataset_id, job.version_id = raw.id, ds.id, v.id
    db.commit()


def label_batch(db, job):
    from .preprocess import _exec_auto_label
    # Always label V1 on retry; preserve earlier versions and human edits.
    initial = db.scalars(select(DatasetVersion).where(DatasetVersion.dataset_id == job.dataset_id,
                                                     DatasetVersion.version_no == 1)).one()
    task = AutoLabelTask(name=job.name + ' · COCO 自动标注', data_type='image',
                         source_ids=json.dumps([job.dataset_id]), algorithm='coco_detect',
                         rules_config=json.dumps({'confThreshold': 0.4}), input_version_id=initial.id,
                         created_by=job.created_by)
    db.add(task)
    db.flush()
    job.label_task_id = task.id
    db.commit()
    _exec_auto_label(task.id)
    db.expire_all()
    task = db.get(AutoLabelTask, task.id)
    if task.status != 'completed':
        raise RuntimeError(task.error_msg or '自动标注失败')
    v = db.get(DatasetVersion, task.output_version_id)
    errors = empty = 0
    for path in Path(v.file_path).glob('*_label.json'):
        label = read_json(path, {})
        errors += bool(label.get('error'))
        empty += not label.get('error') and not label.get('boxes')
    job.label_errors, job.empty_labels = errors, empty
    job.version_id = v.id
    if errors:
        raise RuntimeError(f'{errors} 张图像自动标注失败，可重试；无目标图像 {empty} 张')
    job.status = 'completed'
    db.commit()
