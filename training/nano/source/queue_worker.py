"""One tmux worker per GPU pair; wait without importing CUDA, then train."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--mode',required=True);a=p.parse_args()
    cfg=json.loads(a.config.read_text());root=Path(cfg['paths']['output']);run=root/a.mode
    gpus=cfg['runtime']['train_gpus'];assert 4 not in gpus
    lock_dir=Path('/data/gaoya/agent-data/locks/cosmos_gpu');lock_dir.mkdir(parents=True,exist_ok=True)
    locks=[]
    for gpu in sorted(gpus):
        handle=(lock_dir/f'gpu{gpu}.lock').open('w');fcntl.flock(handle,fcntl.LOCK_EX);locks.append(handle)
    status=root/f'queue_{a.mode}.json'
    def report(state, **extra):
        row=dict(status=state,mode=a.mode,gpus=gpus,worker_pid=os.getpid(),time=time.time(),**extra)
        write(status,row);print(json.dumps(row),flush=True)
    if a.mode!='C':
        while not (root/'C_preflight_pass.json').exists():
            report('waiting_for_C_preflight');time.sleep(30)
    report('checking_gpu_availability')
    idle_since=None
    while True:
        try:
            result=subprocess.check_output(['nvidia-smi','-i',','.join(map(str,gpus)),
                '--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=180)
            rows=[[int(x.strip()) for x in line.split(',')] for line in result.strip().splitlines()]
            active=subprocess.check_output(['nvidia-smi','-i',','.join(map(str,gpus)), '--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=180).strip()
            idle=not active and len(rows)==len(gpus) and all(mem<1024 and util<5 for _,mem,util in rows)
        except (subprocess.SubprocessError, ValueError) as exc:
            idle=False;rows=[];report('gpu_query_retry',error=type(exc).__name__)
        idle_since=(idle_since or time.monotonic()) if idle else None
        if idle and (a.mode=='C' or time.monotonic()-idle_since>=180):break
        report('waiting_for_idle_gpus',observed=rows,required_idle_seconds=180)
        time.sleep(30)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),OMP_NUM_THREADS=str(cfg['runtime']['cpu_threads']),
             OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTORCH_ALLOC_CONF=cfg['runtime']['cuda_allocator'],
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',
             WANDB_CACHE_DIR=str(root/'wandb_cache'),WANDB_CONFIG_DIR=str(root/'wandb_config'),
             WANDB_DATA_DIR=str(root/'wandb_data'),WANDB_MODE='online')
    env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    for key in ['http_proxy','https_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','all_proxy']:
        env.pop(key,None)
    cmd=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc-per-node={len(gpus)}',
         str(Path(__file__).with_name('train.py')),'--config',str(a.config),'--mode',a.mode]
    phases=[]
    if a.mode=='C' and not (root/'C_preflight_pass.json').exists():
        phases.append(('preflight',cmd+['--stop-after','2','--verify-roundtrip']+(['--resume'] if (run/'latest.json').exists() else [])))
    phases.append(('training',cmd))
    for phase,command in phases:
        if phase=='training' and (run/'latest.json').exists():command=command+['--resume']
        log=root/f'{a.mode}_{phase}.log'
        with log.open('a',buffering=1) as f:
            child=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            def stop(signum,frame):
                os.killpg(child.pid,signal.SIGTERM)
                report('interrupted',phase=phase,child_pid=child.pid)
                raise SystemExit(128+signum)
            signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
            report('running',phase=phase,child_pid=child.pid,log=str(log),command=command)
            code=child.wait()
        if code:
            report('failed',phase=phase,exit_code=code,log=str(log));return code
        if phase=='preflight':
            passed=json.loads((run/'roundtrip_pass.json').read_text());assert passed['exact']
            write(root/'C_preflight_pass.json',passed)
    report('complete',completed_steps=cfg['training']['max_steps'])
    return 0

if __name__=='__main__':
    raise SystemExit(main())
