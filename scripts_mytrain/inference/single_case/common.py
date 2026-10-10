"""Single-video CLI and output verification; no historical result/manifest loading."""
import argparse, hashlib, importlib.util, json, os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
HIST = HERE.parent / 'historical'

def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')

def arguments(backbone):
    p=argparse.ArgumentParser(description=f'{backbone}: one video + prompt -> one output video')
    p.add_argument('--config',type=Path,default=HERE/'presets'/f'{backbone}_gap038.json')
    for key in ['video','prompt','model-path','adapter','negative-prompt','context-cache','prompt-cache','physrvg-dit-checkpoint']:
        p.add_argument('--'+key)
    p.add_argument('--mode',choices=['clean','noisy','equal_sigma'])
    for key in ['seed','steps','context-seed']:p.add_argument('--'+key,type=int)
    for key in ['guidance','fps']:p.add_argument('--'+key,type=float)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--gpu',type=int,default=3)
    p.add_argument('--repo',type=Path,default=Path('/home/gaoya/code_V2V_baselines/PhysRVG-main'))
    p.add_argument('--no-cache',action='store_true')
    p.add_argument('--verify-against',type=Path,help='Optional reference MP4; never used as sampling input')
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args();c=json.loads(a.config.read_text());assert c['backbone']==backbone
    original_video,original_prompt=c['video'],c['prompt']
    for key in ['context_cache','prompt_cache','physrvg_dit_checkpoint','negative_prompt']:
        if getattr(a,key) is not None and key not in c:p.error('--'+key.replace('_','-')+' is not supported by this backbone')
    for k,v in vars(a).items():
        if k in c and v is not None:c[k]=v
    changed=c['video']!=original_video or c['prompt']!=original_prompt
    if a.no_cache or changed:
        for k in ['context_cache','prompt_cache']:c[k]=getattr(a,k,None) if not a.no_cache else None
    for key in ['video','adapter','model_path','context_cache','prompt_cache','physrvg_dit_checkpoint']:
        if c.get(key):c[key]=str(Path(c[key]).expanduser().resolve())
    if a.verify_against:a.verify_against=a.verify_against.resolve()
    if not Path(c['video']).is_file() or not Path(c['adapter']).is_dir():p.error('Video and adapter must exist')
    if c.get('cache_video_sha256') and (c.get('context_cache') or c.get('prompt_cache')):
        assert sha(c['video'])==c['cache_video_sha256'] and c['prompt']==c['cache_prompt'],'Cache input mismatch; use --no-cache'
    for key in ['context_cache','prompt_cache']:
        if c.get(key) and not Path(c[key]).is_file():p.error('Missing '+key+': '+c[key])
    if c['steps']<=0 or c['fps']<=0:p.error('Steps and fps must be positive')
    if backbone=='nano' and c['guidance']<=1:p.error('This Nano protocol requires CFG guidance > 1')
    if (c['height'],c['width'],c['frames'])!=(512,896,49):p.error('Pinned implementation supports 512x896,49 frames,8 RGB/2 latent context')
    out=a.output.resolve()
    if not out.is_relative_to('/data/gaoya'):p.error('Output must be under /data/gaoya')
    if out.exists() and any(out.iterdir()):p.error('Output must be new or empty')
    if a.check_only:print(json.dumps(c,indent=2));raise SystemExit(0)
    for row in json.loads((HERE/'dependency_manifest.json').read_text()):
        if sha(HERE/row['file'])!=row['sha256']:p.error('Pinned helper changed: '+row['file'])
    out.mkdir(parents=True,exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS']:os.environ[key]='1' if backbone=='wan' else '4'
    sys.path[:0]=[str(a.repo),str(a.repo/'scripts_mytrain/inference')]
    c['input_video_sha256']=sha(c['video']);save(out/'config.json',c)
    save(out/'sources.json',dict(entrypoints={f.name:sha(f) for f in HERE.glob('*.py')},dependencies=json.loads((HERE/'dependency_manifest.json').read_text())))
    return a,c,out

def finish(a,c,out,audit):
    video=out/'video.mp4';report=dict(status='GENERATED',video_sha256=sha(video),input_video_sha256=c['input_video_sha256'],audit=audit)
    save(out/'result.json',report)
    if a.verify_against:
        import imageio.v2 as imageio,numpy as np,itertools,math
        ref=a.verify_against.resolve();exact=sha(ref)==report['video_sha256'];ss=0.;count=0;maximum=0;frames=0
        left=imageio.get_reader(str(ref));right=imageio.get_reader(str(video))
        try:
            for x,y in itertools.zip_longest(left,right):
                if x is None or y is None or x.shape!=y.shape:raise ValueError('Video shape/frame count mismatch')
                d=x.astype(np.float64)-y.astype(np.float64);ss+=float((d*d).sum());count+=d.size;maximum=max(maximum,int(abs(d).max()));frames+=1
        finally:left.close();right.close()
        mse=ss/count
        compare=dict(reference=str(ref),reference_sha256=sha(ref),video_bytes_equal=exact,decoded_pixels_equal=maximum==0,frames=frames,mse=mse,max_abs=maximum,psnr_db=None if mse==0 else 10*math.log10(255**2/mse),status='BYTE_EXACT' if exact else 'PIXEL_EXACT' if maximum==0 else 'DIFFERENT_REQUIRES_REVIEW')
        save(out/'comparison.json',compare);print(json.dumps(compare),flush=True)
        if not exact:raise SystemExit(2)
    print('COMPLETE',str(video),flush=True)
