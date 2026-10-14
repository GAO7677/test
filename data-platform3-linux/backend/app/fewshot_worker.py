"""Single durable tmux queue worker. The web process never owns training."""
import fcntl
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from sqlalchemy import select
from .config import settings
from .database import SessionLocal
from .models import FewshotJob
from .services import fewshot_service as svc


def command(job, *args):
    c = svc.job_engine(job)
    env = dict(os.environ, PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               CUDA_VISIBLE_DEVICES=c['gpu'], USE_TF='0', TOKENIZERS_PARALLELISM='false')
    if c.get('allow_shared_gpu', False):
        env.update(FEWSHOT_GPU_MIN_FREE_MB=str(c.get('gpu_min_free_mb', 24576)),
                   FEWSHOT_GPU_RESERVE_MB=str(c.get('gpu_reserve_mb', 4096)))
        if args[0] in ('train', 'smoke', 'resume', 'generate') and not svc.gpu_admission(c, svc.gpu_memory(c))[0]:
            raise svc.GpuMemoryWait('GPU 剩余显存不足，继续排队')
    # Real training has online scalar tracking; no model artifact upload.
    with (Path(job.job_dir) / 'worker.log').open('a') as log:
        log.write('\nSTAGE ' + ' '.join(args[:1]) + '\n')
        log.flush()
        subprocess.run([c['python'], '-u', str(Path(c['repo']) / 'run.py'), *args],
                       cwd=c['repo'], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def process(db, job):
    c = svc.job_engine(job)
    parent = db.get(FewshotJob, job.reuse_job_id) if job.reuse_job_id else None
    if parent and not (Path(parent.run_dir) / 'adapter/complete.json').is_file():
        raise ValueError('来源 LoRA 已不存在')
    if not job.dataset_id:
        existing = Path(job.run_dir) if job.run_dir else None
        reusable = existing and ((existing / 'adapter/complete.json').is_file()
                                 or (c['backend'] != 'chimera_sd21' and any((existing / 'adapter').glob('checkpoint-*')))
                                 or (parent and (existing / 'resolved_config.yaml').is_file()))
        if not reusable:
            job.attempts += 1
            job.run_dir = str(Path(job.job_dir) / f'run_{job.attempts:03d}')
            config = svc.build_config(job, job.run_dir, parent)
            # JSON is also valid YAML; avoid adding YAML to the web environment.
            config_path = Path(job.job_dir) / f'config_{job.attempts:03d}.json'
            svc.dump(config_path, config)
            job.status = 'generating' if parent else 'training'
            db.commit()
            command(job, 'check', '--config', str(config_path))
            if parent:
                command(job, 'prepare', '--config', str(config_path))
            else:
                command(job, 'smoke' if job.smoke else 'train', '--config', str(config_path), '--foreground')
        elif not parent and not (Path(job.run_dir) / 'adapter/complete.json').exists():
            job.status = 'training'
            db.commit()
            command(job, 'resume', '--run-dir', job.run_dir, '--foreground')
        job.status = 'generating'
        db.commit()
        batch = svc.completed_batch(job.run_dir)
        if batch is None:
            partials = sorted((Path(job.run_dir) / 'generations').glob('*'))
            args = ['generate', '--run-dir', job.run_dir]
            if partials:
                args += ['--resume-batch', partials[-1].name]
            command(job, *args)
            batch = svc.completed_batch(job.run_dir)
        if batch is None:
            raise RuntimeError('生成命令结束但没有完整批次')
        job.batch_dir = str(batch)
        job.status = 'importing'
        db.commit()
        svc.import_batch(db, job)
    job.status = 'labeling'
    db.commit()
    svc.label_batch(db, job)


def main():
    c = svc.platform_settings()
    root = Path(c['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / '.fewshot-worker.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    # A previous worker vanished. Never silently relaunch costly training.
    with SessionLocal() as db:
        for job in db.scalars(select(FewshotJob).where(FewshotJob.status.in_(svc.ACTIVE))):
            job.status, job.error = 'failed', '执行进程中断；已保留进度和产物，可重试续跑'
        db.commit()
    stable_job, stable_samples = None, 0
    while True:
        try:
            with SessionLocal() as db:
                job = db.scalar(select(FewshotJob).where(FewshotJob.status == 'queued').order_by(FewshotJob.id).limit(1).with_for_update(skip_locked=True))
                if job:
                    c = svc.job_engine(job)
                    gpu_lock = (root / ('.' + c['gpu'] + '.lock')).open('a')
                    try:
                        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        # Shared GPUs require sufficient free memory; exclusive mode waits for idle.
                        memory = svc.gpu_memory(c)
                        allowed, _ = svc.gpu_admission(c, memory)
                        if stable_job != job.id:
                            stable_job, stable_samples = job.id, 0
                        stable_samples = stable_samples + 1 if allowed else 0
                        required_samples = 3 if c.get('allow_shared_gpu', False) else 1
                        svc.dump(Path(job.job_dir) / 'waiting.json', dict(free_mb=memory[1],
                                 required_mb=c.get('gpu_min_free_mb', 24576),
                                 samples=stable_samples, required_samples=required_samples))
                        if allowed and stable_samples >= required_samples:
                            stable_samples = 0
                            job.status = 'generating' if job.reuse_job_id else 'training'
                            db.commit()
                            try:
                                process(db, job)
                            except Exception as exc:
                                jid = job.id
                                db.rollback()
                                job = db.get(FewshotJob, jid)
                                message = svc.failure_message(job, exc)
                                waiting = isinstance(exc, svc.GpuMemoryWait) or 'GPU_MEMORY_WAIT' in message
                                job.status = 'queued' if waiting else 'failed'
                                job.error = '' if waiting else message
                                with (Path(job.job_dir) / 'worker.log').open('a') as out:
                                    out.write(traceback.format_exc())
                                db.commit()
                    except BlockingIOError:
                        pass
                    finally:
                        gpu_lock.close()
        except Exception:
            traceback.print_exc()
        time.sleep(3)


if __name__ == '__main__':
    main()
