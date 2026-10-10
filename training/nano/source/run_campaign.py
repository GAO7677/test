"""Foreground campaign controller: cache -> independent C/A/N75 training -> fixed evaluation."""
import fcntl
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from common import CFG, CONFIG_PATH, OUTPUT, CACHE, write_json, sha


def wait_for_gpus(gpus):
    while True:
        output = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used',
                                          '--format=csv,noheader,nounits'],text=True)
        used = {int(line.split(',')[0]):int(line.split(',')[1]) for line in output.splitlines()}
        if all(used[i]<2048 for i in gpus):
            return
        write_json(OUTPUT/'campaign_status.json',dict(status='waiting_for_gpus',gpus=gpus,memory_MiB=used))
        time.sleep(30)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=CONFIG_PATH)
    parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    lock=(OUTPUT/'campaign.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock.write(str(os.getpid()));lock.flush()
    preflight=json.loads((OUTPUT/'preflight_pass.json').read_text())
    assert preflight['status']=='PASS'
    assert preflight['config_sha256']==sha(CONFIG_PATH), 'Configuration changed since validation'
    source=Path(__file__).parent
    expected=json.loads((OUTPUT/'source_hashes.json').read_text())
    for name,fingerprint in expected.items():
        assert sha(source/name)==fingerprint, f'Source changed after validation: {name}'
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS=str(CFG['runtime']['cpu_threads']),
             OPENBLAS_NUM_THREADS=str(CFG['runtime']['cpu_threads']),MKL_NUM_THREADS=str(CFG['runtime']['cpu_threads']),PYTORCH_ALLOC_CONF='expandable_segments:True',
             HF_HOME='/data/gaoya/agent-data/cache/huggingface-cosmos3',PYTHONUNBUFFERED='1')
    env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    options=['--config',str(CONFIG_PATH)]
    commands=[('cache',[CFG['runtime']['prepare_gpu']],[sys.executable,str(source/'prepare.py')]+options)]
    for mode in list(CFG['noise']['context_scales']):
        commands.append((f'train_{mode}',CFG['runtime']['train_gpus'],[sys.executable,'-m','torch.distributed.run','--standalone',
                         f"--nproc-per-node={len(CFG['runtime']['train_gpus'])}",str(source/'train.py'),'--mode',mode]+options))
    for name in ['baseline']+list(CFG['noise']['context_scales']):
        for benchmark in CFG['evaluation']['benchmarks']:
            cmd=[sys.executable,str(source/'evaluate.py'),'--benchmark',benchmark,'--name',name]+options
            if name!='baseline':cmd += ['--adapter',str(OUTPUT/name/f"step-{CFG['training']['max_steps']:06d}")]
            commands.append((f'evaluate_{name}_{benchmark}',[CFG['runtime']['evaluation_gpu']],cmd))
    write_json(OUTPUT/'campaign_plan.json',dict(stages=[dict(stage=n,gpus=g,command=c) for n,g,c in commands],
               preflight=preflight,source_hashes=expected))
    for stage,gpus,command in commands:
        if stage=='cache' and (CACHE/'complete.json').exists():
            continue
        if stage.startswith('train_'):
            run=OUTPUT/stage.removeprefix('train_')
            if (run/'run_status.json').exists():
                status=json.loads((run/'run_status.json').read_text())
                if status['status']=='complete' and status['completed_steps']==CFG['training']['max_steps']:continue
            if (run/'latest.json').exists():command += ['--resume']
        if stage.startswith('evaluate_'):
            _,name,benchmark=stage.split('_')
            if (OUTPUT/'evaluation'/benchmark/name/'complete.json').exists():continue
        if shutil.disk_usage('/data/gaoya').free < 12*1024**3:
            raise RuntimeError('Less than 12 GiB free: refusing a new stage')
        wait_for_gpus(gpus)
        env['CUDA_VISIBLE_DEVICES']=','.join(map(str,gpus))
        started=time.time()
        with (OUTPUT/f'{stage}.log').open('a',buffering=1) as log:
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            write_json(OUTPUT/'campaign_status.json',dict(status='running',stage=stage,pid=child.pid,
                       controller_pid=os.getpid(),gpus=gpus,started=started,command=command))
            def stop(signum,frame):
                child.terminate()
                raise SystemExit(128+signum)
            signal.signal(signal.SIGTERM,stop)
            signal.signal(signal.SIGINT,stop)
            code=child.wait()
        if code:
            write_json(OUTPUT/'campaign_status.json',dict(status='failed',stage=stage,exit_code=code,
                       log=str(OUTPUT/f'{stage}.log')))
            raise SystemExit(code)
        print(f'COMPLETE {stage} seconds={time.time()-started:.1f}',flush=True)
    write_json(OUTPUT/'campaign_status.json',dict(status='complete',
               note='Training and evaluation videos complete; scoring has not been run by this controller.'))


if __name__=='__main__':
    main()
