"""Generate all selected combinations and require byte-exact reference videos."""
import argparse,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ENVS={'wan':'physrvg-full-sa','nano':'cosmos_nano_acn75','predict25':'cosmos_predict25_official'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backbone',choices=['all',*ENVS],default='all')
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference-dir',type=Path,default=HERE.parents[2]/'demos/gap038')
    a=p.parse_args();names=list(ENVS) if a.backbone=='all' else [a.backbone]
    root=a.output.resolve()
    if not root.is_relative_to('/data/gaoya') or (root.exists() and any(root.iterdir())):p.error('Use a new/empty /data/gaoya output directory')
    for b in names:
        for mode in ['clean','noisy','equal_sigma']:
            if not (a.reference_dir/(b+'_'+mode+'.mp4')).is_file():p.error('Missing reference video: '+b+'_'+mode)
    root.mkdir(parents=True,exist_ok=True);rows=[]
    for b in names:
        for mode in ['clean','noisy','equal_sigma']:
            dest=root/(b+'_'+mode)
            cmd=[f'/data/gaoya/agent-data/envs/{ENVS[b]}/bin/python',str(HERE/(b+'.py')),'--mode',mode,'--gpu',str(a.gpu),'--output',str(dest),'--verify-against',str((a.reference_dir/(b+'_'+mode+'.mp4')).resolve())]
            execution=cmd
            if b=='predict25':
                execution=['bash','-c','source "$1"; shift; exec "$@"','single-case',str(HERE.parent/'historical/frozen/predict25/env.sh'),*cmd]
            with (root/(b+'_'+mode+'.log')).open('w') as log:code=subprocess.call(execution,stdout=log,stderr=subprocess.STDOUT)
            comparison=json.loads((dest/'comparison.json').read_text()) if (dest/'comparison.json').exists() else None
            rows.append(dict(backbone=b,mode=mode,exit_code=code,command=cmd,comparison=comparison))
            (root/'matrix.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(b,mode,comparison['status'] if comparison else 'FAILED',flush=True)
    if any(r['exit_code']!=0 for r in rows):raise SystemExit(2)

if __name__=='__main__':main()
