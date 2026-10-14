"""Configuration, caption-folder snapshots and immutable experiment metadata."""
import copy, datetime, hashlib, importlib.metadata, json, os, re, shutil, sys
from pathlib import Path
import yaml

REPO=Path(__file__).resolve().parents[1]
MODELS=('beyondobjects_sd21','flux2_klein_base_4b','chimera_sd21')
DEFAULTS={
 'name':'experiment','model':'beyondobjects_sd21',
 'paths':{'base_model':'','output_root':'','adapter':None},
 'data':{'root':'','caption_file':'caption.txt','groups':None,'labels':None,'allow_duplicates':False,'allow_caption_truncation':False,'semantic_boosting':False},
 'train':{'rank':4,'alpha':4,'steps':400,'batch_size':4,'gradient_accumulation':1,'learning_rate':0.0001,'learning_rate_b':0.001,'warmup':100,'scheduler':'cosine','seed':0,'resolution':512,'precision':'no','gradient_checkpointing':False,'checkpoint_every':100,'keep_checkpoints':4,'workers':2,'center_crop':False},
 'generate':{'prompt':None,'num_images':10,'steps':50,'guidance_scale':2.0,'seed':0,'compare_base':True,'negative_prompt':None,'class_name':None,'plan':None,'dirichlet_alpha':1.0},
 'runtime':{'gpu':0,'threads':4,'memory_fraction':0.9},
 'tracking':{'enabled':False,'project':'fewshot-generation','entity':None},
}

def dump(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def stamp():return datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')

def resolve_path(value,config_dir):
 raw=str(value)
 expanded=os.path.expandvars(os.path.expanduser(raw))
 if '$' in expanded:raise ValueError(f'Unresolved environment variable: {expanded}')
 p=Path(expanded)
 # Shell-provided relative paths use the launch directory; literal YAML paths
 # retain their existing config-directory semantics.
 base=Path.cwd() if raw.startswith('$') else Path(config_dir)
 return str((base/p).resolve() if not p.is_absolute() else p.resolve())

def config(path):
 path=Path(path).resolve();raw=yaml.safe_load(path.read_text())
 if not isinstance(raw,dict):raise ValueError('Config must be a YAML mapping')
 if raw.get('model',DEFAULTS['model']) not in MODELS:raise ValueError(f'model must be one of {MODELS}')
 c=copy.deepcopy(DEFAULTS)
 if raw.get('model')=='flux2_klein_base_4b':
  c['train'].update(batch_size=1,gradient_accumulation=4,precision='bf16',gradient_checkpointing=True)
  c['generate']['guidance_scale']=4.0
 for k,v in raw.items():
  if k not in c:raise ValueError(f'Unknown config key: {k}')
  if isinstance(c[k],dict):
   if not isinstance(v,dict):raise ValueError(f'{k} must be a mapping')
   unknown=set(v)-set(c[k])
   if unknown:raise ValueError(f'Unknown {k} settings: {unknown}')
   c[k].update(v)
  else:c[k]=v
 if not re.fullmatch(r'[A-Za-z0-9_-]+',c['name']):raise ValueError('name: use letters, numbers, underscore, hyphen')
 for section,key in [('paths','base_model'),('paths','output_root'),('data','root'),('paths','adapter'),('data','labels'),('generate','plan')]:
  value=c[section][key]
  if value is None:continue
  if not value:raise ValueError(f'Set {section}.{key} to your own path')
  c[section][key]=resolve_path(value,path.parent)
 if not isinstance(c['tracking']['enabled'],bool):raise ValueError('tracking.enabled must be true or false')
 for k in ['rank','steps','batch_size','gradient_accumulation','resolution','checkpoint_every','keep_checkpoints']:
  if not isinstance(c['train'][k],int) or isinstance(c['train'][k],bool) or c['train'][k]<1:raise ValueError(f'train.{k} must be a positive integer')
 if c['train']['resolution']%16:raise ValueError('resolution must be divisible by 16')
 if c['train']['precision'] not in ('no','fp16','bf16'):raise ValueError('precision: no/fp16/bf16')
 for k in ['num_images','steps']:
  if not isinstance(c['generate'][k],int) or c['generate'][k]<1:raise ValueError(f'generate.{k} must be positive')
 if not 0<c['runtime']['memory_fraction']<=1:raise ValueError('memory_fraction must be in (0,1]')
 if c['model']!='flux2_klein_base_4b' and c['train']['alpha']!=c['train']['rank']:raise ValueError('SD official trainers use alpha=rank')
 if c['model']=='chimera_sd21' and c['train']['gradient_accumulation']!=1:raise ValueError('Chimera wrapper currently supports accumulation=1 only')
 if c['model']=='flux2_klein_base_4b' and c['generate']['negative_prompt']:raise ValueError('Klein pipeline does not support negative_prompt')
 if c['model']=='chimera_sd21' and c['train']['checkpoint_every']<=c['train']['steps']:raise ValueError('Historical Chimera multi-head intermediate save/resume is not validated; set checkpoint_every greater than steps (final adapter is always saved)')
 return c

def inventory(c):
 from PIL import Image,ImageOps
 if Path(c['data']['caption_file']).name!=c['data']['caption_file']:raise ValueError('caption_file must be a filename')
 if c['data']['groups'] is not None and (not isinstance(c['data']['groups'],list) or not all(isinstance(x,str) for x in c['data']['groups'])):raise ValueError('data.groups must be a list of folder names')
 root=Path(c['data']['root'])
 if not root.is_dir():raise ValueError(f'Missing caption-folder root: {root}')
 groups=c['data']['groups'] or sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith('.'))
 labels=read(c['data']['labels']) if c['data']['labels'] else {}
 rows=[];seen={};warnings=[]
 for group in groups:
  if Path(group).name!=group or group in ('.','..'):raise ValueError(f'Invalid group: {group}')
  folder=root/group;caption=(folder/c['data']['caption_file']).read_text().strip()
  if not caption:raise ValueError(f'Empty caption: {folder}')
  files=sorted(p for p in folder.iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff'))
  if not files:raise ValueError(f'No images: {folder}')
  for p in files:
   with Image.open(p) as im:
    im=ImageOps.exif_transpose(im).convert('RGB');pixels=hashlib.sha256(im.tobytes()+str(im.size).encode()).hexdigest();size=list(im.size)
   if pixels in seen:
    msg=f'Duplicate pixels: {p} and {seen[pixels]}'
    if not c['data']['allow_duplicates']:raise ValueError(msg)
    warnings.append(msg)
   seen[pixels]=str(p)
   mask=None
   if c['data']['semantic_boosting']:
    if c['model']!='chimera_sd21':raise ValueError('semantic_boosting is only for Chimera')
    mask=p.with_suffix('.json');box=read(mask)['enclosing_box']
    if len(box)!=4 or not all(isinstance(x,(int,float)) for x in box):raise ValueError(f'Invalid SB box: {mask}')
   rows.append(dict(mask_source=str(mask) if mask else None,index=len(rows),group=group,source=str(p),filename=p.name,prompt=caption,sha256=sha(p),pixel_sha256=pixels,size=size,label=labels.get(group)))
 if not rows:raise ValueError('No caption groups found')
 if c['model']=='beyondobjects_sd21':
  import math
  per_epoch=math.ceil(math.ceil(len(rows)/c['train']['batch_size'])/c['train']['gradient_accumulation'])
  if c['train']['steps']%per_epoch:raise ValueError(f'Official BeyondObjects trains whole epochs: steps must be a multiple of {per_epoch}')
 return rows,warnings

def check(c,tokens=True):
 rows,warnings=inventory(c);model=Path(c['paths']['base_model'])
 if not (model/'model_index.json').is_file():raise ValueError(f'Expected full Diffusers model directory: {model}')
 for k in ['vae','scheduler','tokenizer','text_encoder','transformer' if c['model']=='flux2_klein_base_4b' else 'unet']:
  if not (model/k).is_dir():raise ValueError(f'Missing model component: {model/k}')
 lengths={}
 if tokens:
  from transformers import AutoTokenizer
  tokenizer=AutoTokenizer.from_pretrained(str(model/'tokenizer'),local_files_only=True,use_fast=False)
  limit=512 if c['model']=='flux2_klein_base_4b' else tokenizer.model_max_length
  for caption in dict.fromkeys(r['prompt'] for r in rows):
   text=caption
   if c['model']=='flux2_klein_base_4b':text=tokenizer.apply_chat_template([{'role':'user','content':caption}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
   n=len(tokenizer(text,truncation=False).input_ids);lengths[caption]=n
   if n>limit:
    msg=f'Caption exceeds {limit} tokens ({n}): {caption}'
    if not c['data']['allow_caption_truncation']:raise ValueError(msg+'; shorten it or explicitly set allow_caption_truncation: true')
    warnings.append(msg)
 return dict(passed=True,images=len(rows),groups=len({r['group'] for r in rows}),warnings=warnings,caption_tokens=lengths),rows

def versions():
 out={'python':sys.version}
 for n in ['torch','torchvision','diffusers','transformers','peft','accelerate','numpy','Pillow','wandb','datasets']:
  try:out[n]=importlib.metadata.version(n)
  except importlib.metadata.PackageNotFoundError:pass
 return out

def prepare(c,smoke=False):
 run=Path(c['paths']['output_root']).resolve()
 if run.exists() and (not run.is_dir() or any(run.iterdir())):
  raise ValueError('Output directory is not empty; choose a new FEWSHOT_OUTPUT, or use generate/resume for this run')
 audit,rows=check(c);run.mkdir(parents=True,exist_ok=True)
 # Exclusive creation also prevents concurrent launches from overwriting config.
 with (run/'resolved_config.yaml').open('x') as f:
  f.write(yaml.safe_dump(c,allow_unicode=True,sort_keys=False))
 for row in rows:
  target=run/'inputs'/row['group']/row['filename'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(row['source'],target)
  (target.parent/'caption.txt').write_text(row['prompt']+'\n');row['image']=str(target.relative_to(run))
  if row.get('mask_source'):
   mask=target.with_suffix('.json');shutil.copyfile(row['mask_source'],mask);row['mask']=str(mask.relative_to(run));row['mask_sha256']=sha(mask)
 dump(run/'manifests/inputs.json',rows);dump(run/'manifests/data_audit.json',audit)
 src={str(p.relative_to(REPO)):sha(p) for base in ['fewshot','vendor'] for p in (REPO/base).rglob('*.py')}
 import zipfile
 with zipfile.ZipFile(run/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
  for name in src:z.write(REPO/name,name)
  z.write(REPO/'run.py','run.py')
 dump(run/'run.json',dict(id=run.name,created=now(),smoke=smoke,model=c['model'],environment=versions(),source_sha256=src,base_model=str(Path(c['paths']['base_model'])),base_config_sha256=sha(Path(c['paths']['base_model'])/'model_index.json')))
 for name in ['logs','adapter','generations','report']:(run/name).mkdir()
 dump(run/'status.json',dict(phase='prepared',step=0,total_steps=c['train']['steps']))
 return run

def load_run(run):
 run=Path(run).resolve();c=yaml.safe_load((run/'resolved_config.yaml').read_text());return run,c

def device(c):
 cache=Path(c['paths']['output_root'])/'.cache';cache.mkdir(parents=True,exist_ok=True)
 for key,subdir in [('HF_HOME','huggingface'),('HF_DATASETS_CACHE','datasets'),('TORCH_HOME','torch'),('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('WANDB_CACHE_DIR','wandb')]:os.environ[key]=str(cache/subdir)
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
 # Some environments preload Hub at Python startup, before these variables exist.
 hub_constants=sys.modules.get('huggingface_hub.constants')
 if hub_constants is not None:hub_constants.HF_HUB_OFFLINE=True
 os.environ['CUDA_VISIBLE_DEVICES']=str(c['runtime']['gpu']);os.environ['TOKENIZERS_PARALLELISM']='false'
 fraction=c['runtime']['memory_fraction']
 if os.environ.get('FEWSHOT_GPU_MIN_FREE_MB'):
  import subprocess
  raw=subprocess.check_output(['nvidia-smi','-i',str(c['runtime']['gpu']),'--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'],text=True)
  free,total=[int(v.strip()) for v in raw.strip().split(',')]
  reserve=int(os.environ.get('FEWSHOT_GPU_RESERVE_MB','4096'))
  if free<int(os.environ['FEWSHOT_GPU_MIN_FREE_MB']) or free<=reserve:raise RuntimeError('GPU_MEMORY_WAIT: available memory changed before model loading')
  fraction=min(fraction,(free-reserve)/total)
 import torch
 if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required')
 torch.set_num_threads(c['runtime']['threads']);torch.cuda.set_per_process_memory_fraction(fraction,0)
 return torch


def verify_inputs(run):
 rows=read(run/'manifests/inputs.json')
 for r in rows:
  if sha(run/r['image'])!=r['sha256']:raise ValueError(f"Input snapshot changed: {r['image']}")
  if r.get('mask') and sha(run/r['mask'])!=r['mask_sha256']:raise ValueError('SB annotation changed')
 return rows
