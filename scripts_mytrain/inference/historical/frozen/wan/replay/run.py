"""One GPU, serial reference sampling and same-input forward replay."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS']:os.environ[key]='1'
os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import contextlib,hashlib,json,sys,time,traceback,subprocess
from argparse import Namespace
from pathlib import Path
import numpy as np
import torch
from audit import ROOT,REPO,save,sha
from observer import Observer,query_indices
torch.set_num_threads(1);torch.set_num_interop_threads(1)
sys.path[:0]=[str(REPO),str(REPO/'scripts_mytrain/inference')]

def cpu(x):
 if torch.is_tensor(x):return x.detach().cpu().clone()
 if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return type(x)(cpu(v) for v in x)
 return x
def cuda(x):
 if torch.is_tensor(x):return x.to('cuda:0')
 if isinstance(x,dict):return {k:cuda(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return type(x)(cuda(v) for v in x)
 return x
def digest(x):
 if torch.is_tensor(x):return dict(shape=list(x.shape),dtype=str(x.dtype),sha256=hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
 if isinstance(x,dict):return {k:digest(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [digest(v) for v in x]
 return x
def diff(a,b):
 d=(a.float()-b.float());return dict(max_abs=float(d.abs().max()),relative_l2=float(d.norm()/a.float().norm().clamp_min(1e-12)))

class Experiment:
 def __init__(self):
  from infer_full_sa_lora_json_list import load_pipeline
  self.cfg=json.loads((ROOT/'resolved_config.json').read_text());self.validation=dict(unit_tests='PASS',status='RUNNING',forwards=[],adapter_checks=[],sampling_checks=[])
  args=Namespace(lora_checkpoint=None,context_frames=8,context_mask_mode='official_fixed2',model_id=Path(self.cfg['base']['model_id']),physrvg_dit_checkpoint=Path(self.cfg['base']['path']),device='cuda:0',object_xssc_trainable=None)
  print('LOAD PIPELINE',flush=True)
  self.pipe,_,self.base_info,_,_=load_pipeline(args)
  self.raw=self.pipe.transformer;self.raw.eval();self.pipe.vae.eval();self.pipe.text_encoder.eval()
  assert not getattr(self.raw,'context_only_latent_frames',0)
  assert not getattr(self.raw,'_cache_config',None)
  assert self.raw.config.patch_size[0]==1
  self.peft=None;self.prompts={};self.inputs={}
  self.cfg['runtime']=dict(blocks=len(self.raw.blocks),heads=[b.attn1.heads for b in self.raw.blocks],patch_size=list(self.raw.config.patch_size),scheduler=dict(self.pipe.scheduler.config),dtype=str(next(self.raw.parameters()).dtype),base_load=self.base_info,cache='not enabled',sequence_parallel=False,backend=[str(b.attn1.processor._attention_backend) for b in self.raw.blocks],cpu_affinity=sorted(os.sched_getaffinity(0)))
  save(ROOT/'resolved_config.json',self.cfg)
 def save_validation(self):save(ROOT/'validation.json',self.validation)
 @contextlib.contextmanager
 def method(self,name):
  if self.peft is None:
   assert name=='Base';yield
  elif name=='Base':
   with self.peft.disable_adapter():yield
  else:
   self.peft.set_adapter('analysis_'+name);self.peft.eval();yield
 def prepare_case(self,case):
  from infer_full_sa_lora_json_list import load_context_video
  if case not in self.prompts:
   with torch.inference_mode():emb,_=self.pipe.encode_prompt(prompt=self.cfg['cases'][case]['prompt'],negative_prompt='',do_classifier_free_guidance=False,device=torch.device('cuda:0'))
   self.prompts[case]=emb.detach()
   self.inputs[case]=load_context_video(Path(self.cfg['cases'][case]['context_video']),512,896,max_frames=8)
   save(ROOT/'replay_states'/f'{case}_prompt.json',dict(prompt=self.cfg['cases'][case]['prompt'],embedding=digest(emb),context_video_sha256=sha(self.cfg['cases'][case]['context_video'])))
 def reference(self,case,seed,owner):
  existing=[f'{owner}__{case}__seed{seed}__sigma{target:g}' for target in self.cfg['targets']]
  if all((ROOT/'replay_states'/(ident+'.pt')).exists() for ident in existing):
   for ident in existing:self.state(ident)
   print('REUSE VERIFIED REFERENCE STATES',existing,flush=True)
   return existing
  self.prepare_case(case);saved=[];counter=0;condition={};orig=self.pipe.prepare_latents
  def prepare(*a,**kw):
   out=orig(*a,**kw);condition.update(condition=cpu(out[1]),condition_mask=cpu(out[2]));return out
  def capture(module,args,kw):
   nonlocal counter
   sig=self.pipe.scheduler.sigmas[:len(self.pipe.scheduler.timesteps)].detach().cpu()
   chosen=[int((sig-t).abs().argmin()) for t in self.cfg['targets']]
   assert len(set(chosen))==3
   if counter in chosen:
    target=self.cfg['targets'][chosen.index(counter)];actual=float(sig[counter]);inputs=cpu(kw)
    shape=inputs['hidden_states'].shape;grid=tuple(s//p for s,p in zip(shape[2:],self.raw.config.patch_size));space=grid[1]*grid[2]
    assert grid[0]==13 and inputs['timestep'].shape==(1,13*space)
    assert (inputs['timestep'][:,:2*space]==0).all()
    assert torch.equal(inputs['hidden_states'][:,:,:2],condition['condition'][:,:,:2].to(inputs['hidden_states'].dtype))
    ident=f'{owner}__{case}__seed{seed}__sigma{target:g}'
    metadata=dict(id=ident,case=case,seed=seed,reference_owner=owner,target_sigma=target,actual_sigma=actual,timestep=float(self.pipe.scheduler.timesteps[counter]),sampling_step_index=counter,grid=grid,seq_len=13*space,branch='conditional_no_cfg',input_hashes=digest(inputs),condition_hashes=digest(condition))
    torch.save(dict(inputs=inputs,condition=condition,metadata=metadata),ROOT/'replay_states'/(ident+'.pt'))
    save(ROOT/'replay_states'/(ident+'.json'),metadata);saved.append(ident)
    print('STATE',ident,'step',counter,'sigma',actual,flush=True)
   counter+=1
  self.pipe.prepare_latents=prepare;hook=self.pipe.transformer.register_forward_pre_hook(capture,with_kwargs=True)
  try:
   with self.method(owner),torch.inference_mode():
    self.pipe(video=self.inputs[case],device=torch.device('cuda:0'),prompt_embeds=self.prompts[case],negative_prompt=None,height=512,width=896,num_frames=49,num_inference_steps=40,guidance_scale=5.,do_cfg=False,generator=torch.Generator(device='cuda').manual_seed(seed),output_type='latent',dynamic_condition_mask=False,context_noise_mode='clean')
  finally:hook.remove();self.pipe.prepare_latents=orig
  assert counter==40 and len(saved)==3
  return saved
 def load_adapters(self):
  from peft import PeftModel,LoraConfig,get_peft_model_state_dict
  from safetensors.torch import load_file
  matches={}
  for method in ['C','A','B','N75']:
   adapter='analysis_'+method
   path=self.cfg['models'][method]['source_checkpoint']
   expected=load_file(str(Path(path)/'adapter_model.safetensors'))
   assert all('.lora_A.' in k or '.lora_B.' in k for k in expected),'Extra non-LoRA tensors'
   if self.peft is None:self.peft=PeftModel.from_pretrained(self.raw,path,adapter_name=adapter,is_trainable=False)
   else:
    # Match from_pretrained's FP32 adapter allocation before copying saved weights.
    self.peft.add_adapter(adapter,LoraConfig.from_pretrained(path))
    for module in self.peft.modules():
     for attr in ['lora_A','lora_B']:
      bank=getattr(module,attr,None)
      if bank is not None and adapter in bank:bank[adapter].float()
    self.peft.load_adapter(path,adapter_name=adapter,is_trainable=False)
   actual=get_peft_model_state_dict(self.peft,adapter_name=adapter)
   assert set(actual)==set(expected),(set(actual)-set(expected),set(expected)-set(actual))
   assert all(torch.equal(actual[k].detach().cpu().float(),v.float()) for k,v in expected.items()),'Adapter tensors not exact'
   targets={n:dict(r=m.r[adapter],alpha=m.lora_alpha[adapter],scaling=m.scaling[adapter],merged=bool(m.merged)) for n,m in self.peft.named_modules() if hasattr(m,'lora_A') and adapter in m.lora_A}
   assert targets and not any(t['merged'] for t in targets.values())
   matches[method]=dict(tensors=len(actual),keys_exact=True,tensors_exact=True,targets=targets,extra_non_lora=False,inference_scale=1.)
  self.pipe.transformer=self.peft;self.peft.eval();self.raw=self.peft.get_base_model();save(ROOT/'adapter_loading.json',matches)
 def state(self,ident):
  state=torch.load(ROOT/'replay_states'/(ident+'.pt'),map_location='cpu',weights_only=False)
  assert digest(state['inputs'])==state['metadata']['input_hashes'];return state
 def forward(self,ident,method,observe=True,nquery=32,blocks=None,save_arrays=True):
  state=self.state(ident);inputs=cuda(state['inputs']);meta=state['metadata'];before=digest(inputs)
  context=Observer(self.raw,meta['grid'],nquery=nquery,blocks=blocks) if observe else contextlib.nullcontext()
  start=time.time()
  with self.method(method),torch.inference_mode(),context as obs,torch.autocast('cuda',dtype=torch.bfloat16):out=self.pipe.transformer(**inputs)[0].detach().cpu()
  assert before==digest(inputs)==meta['input_hashes']
  arrays=None
  if observe:
   assert set(obs.arrays)==set(range(len(self.raw.blocks)) if blocks is None else blocks)
   indexes=sorted(obs.arrays);arrays=dict(token_attention_frame_mass=np.stack([obs.arrays[i][0] for i in indexes]),pooled_qk_frame_map=np.stack([obs.arrays[i][1] for i in indexes]),blocks=np.array(indexes),query_indices=obs.indices)
   metadata=dict(**meta,method=method,checkpoint_step=0 if method=='Base' else 1000,query_count=nquery,estimator='query_sample_estimate_all_legal_keys',input_hashes_verified=True,observer=obs.metadata,elapsed_seconds=time.time()-start)
   if save_arrays:
    path=ROOT/'attention_arrays'/f'{ident}__{method}__q{nquery}'
    np.savez_compressed(str(path)+'.npz',**arrays);save(str(path)+'.json',metadata)
  print('FORWARD',ident,method,'observer',observe,'queries',nquery,round(time.time()-start,2),flush=True)
  return out,arrays
 def pilot(self,states):
  first=states[0];plain,_=self.forward(first,'Base',False)
  self.load_adapters()
  off,_=self.forward(first,'Base',False);check=diff(plain,off);assert check['max_abs']<=1e-5,check
  self.validation['plain_base_vs_disabled_adapter']=check;del plain,off
  # Validate normal repeat, observer on/off, and on->off->on for every adapter.
  base,_=self.forward(first,'Base',False)
  for method in ['Base','C','A','B','N75']:
   normal,_=self.forward(first,method,False);repeat,_=self.forward(first,method,False)
   observed,a=self.forward(first,method,True)
   observed_repeat,ar=self.forward(first,method,True,save_arrays=False)
   array_repeat={key:float(np.abs(a[key]-ar[key]).max()) for key in ['token_attention_frame_mass','pooled_qk_frame_map']}
   noise=diff(normal,repeat);effect=diff(normal,observed)
   assert effect['max_abs']<=max(1e-5,noise['max_abs']*2),effect
   assert diff(observed,observed_repeat)['max_abs']<=max(1e-5,noise['max_abs']*2)
   self.validation['forwards'].append(dict(method=method,normal_repeat=noise,observer_effect=effect,attention_repeat_max_abs=array_repeat))
   if method!='Base':
    restored,_=self.forward(first,'Base',False);on_again,_=self.forward(first,method,False)
    restore=diff(base,restored);on=diff(normal,on_again)
    assert restore['max_abs']<=1e-5 and on['max_abs']<=max(1e-5,noise['max_abs']*2)
    self.validation['adapter_checks'].append(dict(method=method,off_restores_base=restore,on_restores_adapter=on))
   self.save_validation()
  count=len(self.raw.blocks);blocks=sorted(set([0,count//2,count-1]));heads=sorted(set([0,self.raw.blocks[0].attn1.heads//2,self.raw.blocks[0].attn1.heads-1]))
  for ident in states:
   for method in ['Base','C','A','B','N75']:
    file=ROOT/'attention_arrays'/f'{ident}__{method}__q32.npz'
    if file.exists():a=dict(np.load(file))
    else:_,a=self.forward(ident,method)
    _,b=self.forward(ident,method,nquery=128,blocks=blocks)
    delta=b['token_attention_frame_mass'][:,heads]-a['token_attention_frame_mass'][blocks][:,heads]
    context_delta=delta[:,:,2:,:2].sum(-1).mean(-1)
    # Threshold fixed before measurement: 1 percentage point absolute context mass.
    self.validation['sampling_checks'].append(dict(state=ident,method=method,blocks=blocks,heads=heads,max_abs_context_mass=float(np.abs(context_delta).max()),mean_abs_context_mass=float(np.abs(context_delta).mean()),unstable=bool(np.abs(context_delta).max()>.01),threshold=.01))
    self.save_validation()
  self.validation['status']='PILOT_CRITICAL_CHECKS_PASS';self.save_validation()
 def main(self):
  case='v2v_pendulum_l110';pilot_states=self.reference(case,42,'Base');self.pilot(pilot_states)
  for case in self.cfg['cases']:
   for seed in self.cfg['seeds']:
    if case=='v2v_pendulum_l110' and seed==42:continue
    for state in self.reference(case,seed,'Base'):
     for method in ['Base','C','A','B','N75']:self.forward(state,method)
  for state in self.reference('v2v_pendulum_l110',42,'B'):
   for method in ['Base','C','A','B','N75']:self.forward(state,method)
  self.validation['status']='COMPLETE';self.save_validation()

def main():
 ROOT.joinpath('logs').mkdir(exist_ok=True)
 for name in ['failure.json','validation.json']:
  previous=ROOT/name
  if previous.exists():previous.rename(ROOT/'logs'/f'{int(time.time())}_{name}')
 class Tee:
  def __init__(self,stream,file):self.stream=stream;self.file=file
  def write(self,text):self.stream.write(text);self.file.write(text);self.file.flush()
  def flush(self):self.stream.flush();self.file.flush()
 log=open(ROOT/'logs'/f'run_{int(time.time())}.log','a',buffering=1)
 sys.stdout=Tee(sys.stdout,log);sys.stderr=Tee(sys.stderr,log)
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
 mem=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
 if mem>1024:raise RuntimeError('GPU0 occupied; no preemption')
 lease=Path('/data/gaoya/agent-data/outputs/a3_training_redesign_20260908/p4_v2_test70_20260912/configs/training_gpu_reservation.json')
 if lease.exists():
  v=json.loads(lease.read_text())
  if v.get('expires_unix',0)>time.time() and 0 in v.get('reserved_gpus',[]):raise RuntimeError('GPU0 reserved')
 exp=None
 try:exp=Experiment();exp.main()
 except Exception as error:
  save(ROOT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),time=time.time()))
  if exp is not None:exp.validation['status']='STOPPED_CRITICAL_FAILURE';exp.save_validation()
  raise
if __name__=='__main__':main()
