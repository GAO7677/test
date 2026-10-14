import argparse, copy, hashlib, json, os, shlex, subprocess, sys, traceback
from pathlib import Path
from .core import REPO,config,check,prepare,load_run,dump,read,now

def main():
 p=argparse.ArgumentParser(description='FewShotLab: caption folders → model → LoRA → PNG + JSON')
 sub=p.add_subparsers(dest='command',required=True)
 for name in ['check','prepare','train','run','smoke']:
  a=sub.add_parser(name);a.add_argument('--config',required=True);a.add_argument('--foreground',action='store_true');a.add_argument('--no-token-check',action='store_true',help='check only: skip tokenizer load')
 for name in ['generate','resume','status','report','_train','_pipeline']:
  a=sub.add_parser(name);a.add_argument('--run-dir',required=True)
  if name=='generate':a.add_argument('--prompt');a.add_argument('--num-images',type=int);a.add_argument('--resume-batch')
  if name in ['resume','_train','_pipeline']:a.add_argument('--checkpoint',default=None)
  if name=='resume':a.add_argument('--foreground',action='store_true')
 args=p.parse_args();run=None;started=now()
 try:
  if args.command in ['check','prepare','train','run','smoke']:
   c=config(args.config)
   if args.command=='check':print(json.dumps(check(c,tokens=not args.no_token_check)[0],ensure_ascii=False,indent=2));return
   if args.command in ('train','run','smoke') and c['paths']['adapter']:raise ValueError('For an existing adapter use prepare + generate; clear paths.adapter before training')
   if args.command=='smoke':
    c['name']+='-smoke';c['train'].update(steps=2,warmup=0,checkpoint_every=100000000)
    c['generate'].update(num_images=1) # Keep official sampling steps for save/load validation.
   run=prepare(c,smoke=args.command=='smoke');print('RUN_DIR='+str(run),flush=True)
   if args.command=='prepare':return
   stage='_train' if args.command=='train' else '_pipeline'
   launch(run,stage,foreground=args.foreground or args.command=='smoke');return
  run,c=load_run(args.run_dir)
  if args.command=='status':print(json.dumps(read(run/'status.json'),ensure_ascii=False,indent=2));return
  if args.command=='report':
   from .report import build
   print(build(run));return
  if args.command=='resume':
   if not args.checkpoint:args.checkpoint='latest'
   launch(run,'_pipeline',args.foreground,args.checkpoint);return
  if args.command=='generate':
   from .generation import generate
   print(generate(run,c,args.prompt,args.num_images,args.resume_batch));return
  # Prevent duplicate training/recovery workers for the same run.
  import fcntl
  lock=(run/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if args.command=='_pipeline':
   env=dict(os.environ,PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
   if not (run/'adapter/complete.json').exists():
    cmd=[sys.executable,str(REPO/'run.py'),'_train','--run-dir',str(run)]
    if args.checkpoint:cmd+=['--checkpoint',args.checkpoint]
    # _train owns its own lock; release only while waiting for that child.
    fcntl.flock(lock,fcntl.LOCK_UN)
    subprocess.run(cmd,check=True,env=env)
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   subprocess.run([sys.executable,str(REPO/'run.py'),'generate','--run-dir',str(run)],check=True,env=env)
  else:
   from .training import train
   train(run,c,args.checkpoint)
 except BaseException as e:
  if run and Path(run).exists():
   failure=dict(error=repr(e),traceback=traceback.format_exc(),time=now())
   saved=read(Path(run)/'failure.json') if (Path(run)/'failure.json').exists() else None
   if isinstance(e,subprocess.CalledProcessError) and saved and saved.get('time','')>=started:
    failure=dict(saved,launcher_error=repr(e))
   dump(Path(run)/'failure.json',failure)
   status=read(Path(run)/'status.json') if (Path(run)/'status.json').exists() else {}
   status.update(phase='failed',error=failure['error']);dump(Path(run)/'status.json',status)
  raise

def launch(run,stage,foreground=False,checkpoint=None):
 cmd=[sys.executable,'-u',str(REPO/'run.py'),stage,'--run-dir',str(run)]
 if checkpoint:cmd+=['--checkpoint',checkpoint]
 if foreground:
  if not os.environ.get('TMUX') and not read(run/'run.json')['smoke']:raise RuntimeError('For formal foreground training, first enter tmux; omit --foreground for automatic tmux launch')
  subprocess.run(cmd,check=True);return
 session='fewshot-'+hashlib.sha256(str(run.resolve()).encode()).hexdigest()[:12]
 env=['env','PYTHONNOUSERSITE=1','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1']
 shell=shlex.join(env+cmd)+' > '+shlex.quote(str(run/'logs/console.log'))+' 2>&1'
 subprocess.run(['tmux','new-session','-d','-s',session,shell],check=True)
 dump(run/'session.json',dict(tmux=session,command=shlex.join(cmd)))
 print('tmux attach -t '+session);print('Logs: '+str(run/'logs/console.log'))
