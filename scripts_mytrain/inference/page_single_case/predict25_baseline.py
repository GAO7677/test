"""One explicit video/prompt, preserving the archived Test70_v2 inference implementation."""
import argparse, hashlib, json, os, runpy, subprocess, sys
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--prompt',required=True)
    p.add_argument('--case',default='single')
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args();c=json.loads(a.config.read_text())
    assert a.gpu in [0,1,2,3,5,6], 'Physical GPU4 forbidden'
    assert a.case and Path(a.case).name==a.case and a.case not in ('.','..')
    video=a.video.resolve();out=a.output.resolve()
    assert video.is_file() and out.is_relative_to('/data/gaoya')
    assert not out.exists() or not any(out.iterdir()), 'Use a new output directory'
    for name,digest in c['source_sha256'].items():
        assert sha(name)==digest, 'Source changed: '+name
    if a.check_only:
        print('SOURCE_AND_INPUT_CHECK_PASS');return
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',TOKENIZERS_PARALLELISM='false')
    out.mkdir(parents=True,exist_ok=True)
    row=dict(prompt=a.prompt,context_video=str(video),gt_video=str(video))
    save(out/'single_case_config.json',dict(config=c,video=str(video),prompt=a.prompt,
        case=a.case,gpu=a.gpu,input_sha256=sha(video),entry_sha256=sha(__file__)))
    settings=c['settings'];settings['benchmarks']['test70_v2']=dict(manifest=str(out/'manifest.json'),count=1,fps=30)
    save(out/'settings.json',settings);save(out/'manifest.json',dict(fps=30,cases={a.case:row}))
    env=dict(os.environ);repo=Path(c['model_repo']);env['PYTHONPATH']=str(repo)+os.pathsep+str(repo/'packages/cosmos-oss')
    env.pop('SMOKE',None)
    subprocess.run([sys.executable,c['entry'],'--settings',str(out/'settings.json'),
        '--benchmark','test70_v2','--method','baseline','--output',str(out/'native'),
        '--case',a.case],env=env,cwd=repo,check=True)
    generated=out/'native/results'/a.case/'video.mp4'

    assert generated.is_file(), str(generated)
    if generated != out/'video.mp4':(out/'video.mp4').symlink_to(generated.relative_to(out))
    save(out/'single_case_result.json',dict(status='GENERATED',video_sha256=sha(generated),case=a.case))
    print('GENERATED',out/'video.mp4',flush=True)

if __name__=='__main__':main()
