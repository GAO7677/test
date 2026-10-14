"""Thin data/config adapters around the exact historical training sources."""
import importlib.util, math, os, sys, time
from pathlib import Path
import yaml
from .core import verify_inputs,REPO,device,dump,read,sha,now


def module_at(path,name):
 sys.path.insert(0,str(path.parent));spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def train(run,c,resume=None):
 torch=device(c);t=c['train'];rows=verify_inputs(run);out=run/'adapter';smoke=read(run/'run.json')['smoke']
 if (out/'complete.json').exists():raise RuntimeError('Training already complete; use generate')
 if c['model']=='chimera_sd21' and resume:raise ValueError('Historical Chimera multi-head checkpoint resume is not validated; start a new run')
 if not smoke and not os.environ.get('TMUX'):raise RuntimeError('Formal training must run in tmux (CLI launches it automatically)')
 if resume:
  target=out/resume if resume!='latest' else max(out.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]),default=None)
  if target is None or not target.is_dir():raise ValueError('Requested checkpoint not found; refusing fresh training')
  resume=target.name
 from .tracking import start_run
 run_wb=start_run(c['tracking'],name=run.name,directory=run,config=c,smoke=smoke)
 if run_wb:
  dump(run/'wandb.json',dict(url=run_wb.url,id=run_wb.id,weights_uploaded=False))
 import accelerate
 start=time.monotonic();last={};original_log=accelerate.Accelerator.log
 def record(values,step):
  if step is None or step<=last.get('step',-1):return
  loss=float(values['loss'])
  if not math.isfinite(loss):raise RuntimeError('Nonfinite loss')
  last.update(phase='training',step=step,total_steps=t['steps'],loss=loss,elapsed_seconds=time.monotonic()-start)
  dump(run/'status.json',last)
  with (run/'logs/metrics.jsonl').open('a') as f:
   import json
   f.write(json.dumps(last)+'\n')
  if run_wb:run_wb.log(values,step=step)
  if step==1 or step%20==0:print('PROGRESS',last,flush=True)
 def log(self,values,step=None,**kwargs):
  if self.sync_gradients and 'loss' in values:record(values,step)
  return original_log(self,values,step=step,**kwargs)
 accelerate.Accelerator.log=log
 try:
  if c['model']=='beyondobjects_sd21':
   m=module_at(REPO/'vendor/beyondobjects/finetune_t2i.py','official_beyondobjects')
   if resume:
    original_state=m.LoraLoaderMixin.lora_state_dict
    def local_state(*a,**kw):
     kw.setdefault('weight_name','pytorch_lora_weights.safetensors');kw.setdefault('local_files_only',True);return original_state(*a,**kw)
    m.LoraLoaderMixin.lora_state_dict=staticmethod(local_state)
   # Preserve original transforms, tokenization, loss, optimizer and sample order.
   def get_list(self):
    images=[str(run/r['image']) for r in rows];captions=[r['prompt'] for r in rows]
    return images,['unused']*len(rows),captions,['unused']*len(rows),len(rows)
   m.Dataset.get_instance_data_list=get_list
   native=run/'native_sd.yaml';native.write_text(yaml.safe_dump(dict(model=c['paths']['base_model'],lora_path=str(out),data_path=str(run/'inputs'),caption_path=str(run/'inputs'),template='{1}',classnames=['support'])))
   per_epoch=math.ceil(math.ceil(len(rows)/t['batch_size'])/t['gradient_accumulation'])
   argv=[f'--yaml_file={native}',f'--rank={t["rank"]}',f'--num_train_epochs={t["steps"]//per_epoch}',f'--train_batch_size={t["batch_size"]}',f'--gradient_accumulation_steps={t["gradient_accumulation"]}',f'--learning_rate={t["learning_rate"]}',f'--lr_scheduler={t["scheduler"]}',f'--lr_warmup_steps={t["warmup"]}',f'--mixed_precision={t["precision"]}',f'--seed={t["seed"]}',f'--resolution={t["resolution"]}',f'--checkpointing_steps={t["checkpoint_every"]}',f'--checkpoints_total_limit={t["keep_checkpoints"]}',f'--dataloader_num_workers={t["workers"]}','--is_tqdm=False','--report_to=none']
   if resume:argv.append(f'--resume_from_checkpoint={resume}')
   if t['center_crop']:argv.append('--center_crop')
   if t['gradient_checkpointing']:argv.append('--gradient_checkpointing')
   sys.argv=['train']+argv;args=m.parse_args();args.train_text_encoder=True;args.report_to=None
  elif c['model']=='flux2_klein_base_4b':
   data=run/'native_flux';data.mkdir(exist_ok=True)
   import json
   with (data/'metadata.jsonl').open('w') as f:
    for r in rows:
     name=f'{r["index"]:06d}'+Path(r['image']).suffix;p=data/name
     if not p.exists():p.symlink_to(run/r['image'])
     f.write(json.dumps({'file_name':name,'text':r['prompt']})+'\n')
   m=module_at(REPO/'vendor/flux2/official_train.py','official_flux')
   argv=[f'--pretrained_model_name_or_path={c["paths"]["base_model"]}',f'--dataset_name={data}','--caption_column=text',f'--output_dir={out}','--instance_prompt=an image',f'--resolution={t["resolution"]}',f'--rank={t["rank"]}',f'--lora_alpha={t["alpha"]}',f'--train_batch_size={t["batch_size"]}',f'--gradient_accumulation_steps={t["gradient_accumulation"]}',f'--max_train_steps={t["steps"]}',f'--learning_rate={t["learning_rate"]}',f'--lr_scheduler={t["scheduler"]}',f'--lr_warmup_steps={t["warmup"]}',f'--mixed_precision={t["precision"]}','--cache_latents','--max_sequence_length=512','--text_encoder_out_layers','9','18','27',f'--dataloader_num_workers={t["workers"]}',f'--seed={t["seed"]}',f'--checkpointing_steps={t["checkpoint_every"]}',f'--checkpoints_total_limit={t["keep_checkpoints"]}','--skip_final_inference','--report_to=none']
   if t['gradient_checkpointing']:argv.append('--gradient_checkpointing')
   if t['center_crop']:argv.append('--center_crop')
   if resume:argv.append(f'--resume_from_checkpoint={resume}')
   args=m.parse_args(argv);args.report_to=None;m.args=args
  else:
   m=module_at(REPO/'vendor/chimera/main_chimera.py','historical_chimera')
   # Single support collection -> one shared A and one B per image.
   # Optional class labels partition heads for generation, not caption folders.
   m.CLASSNAMES['fgvc_aircraft']=['support']
   def get_list(self):
    import random
    for _ in rows:random.choice(m.TEMPLATES_SMALL[:1]) # Preserve native Python RNG consumption.
    return [str(run/r['image']) for r in rows],[r['prompt'] for r in rows],len(rows)
   def get_masked_list(self):
    images,prompts,n=get_list(self);return [str(run/r['mask']) for r in rows],images,prompts,n
   m.DreamBoothDataset.get_instance_data_list=get_masked_list
   m.DreamBoothDataset.get_instance_data_list_wo_mask=get_list
   native=run/'native_chimera.yaml';native.write_text(yaml.safe_dump(dict(model_dir=c['paths']['base_model'],output_dir=str(out),fewshot_dir=str(run/'inputs'),mask_dir=str(run/'absent_masks'))));os.environ['CHIMERA_CONFIG']=str(native)
   argv=['--dataset=fgvc_aircraft',f'--n_shot={len(rows)}','--n_template=1','--method=JOINT',f'--rank={t["rank"]}',f'--max_train_steps={t["steps"]}',f'--train_batch_size={t["batch_size"]}','--gradient_accumulation_steps=1',f'--learning_rate_A={t["learning_rate"]}',f'--learning_rate_B={t["learning_rate_b"]}',f'--lr_warmup_steps={t["warmup"]}',f'--lr_scheduler={t["scheduler"]}',f'--seed={t["seed"]}',f'--mixed_precision={t["precision"]}',f'--resolution={t["resolution"]}',f'--center_crop={t["center_crop"]}',f'--checkpointing_steps={t["checkpoint_every"]}',f'--dataloader_num_workers={t["workers"]}','--is_tqdm=False','--report_to=none']
   args=m.parse_args(argv);args.output_dir=str(out);args.fewshot_dir=str(run/'inputs');args.mask_dir=str(run/'inputs' if c['data']['semantic_boosting'] else run/'absent_masks');args.report_to=None
   if t['gradient_checkpointing']:args.gradient_checkpointing=True
   old_backward=accelerate.Accelerator.backward;counter=[0]
   def backward(self,loss,**kw):
    result=old_backward(self,loss,**kw);counter[0]+=1;record({'loss':loss.detach().item()},counter[0]);return result
   accelerate.Accelerator.backward=backward
  dump(run/'native_args.json',vars(args));m.main(args)
  if last.get('step')!=t['steps']:raise RuntimeError(f'Incomplete step count: {last}')
  from safetensors.torch import load_file
  weights=list(out.rglob('pytorch_lora_weights.safetensors')) if c['model']=='chimera_sd21' else [out/'pytorch_lora_weights.safetensors']
  weights=[p for p in weights if not any(s.startswith('checkpoint-') for s in p.relative_to(out).parts)]
  if len(weights)!=(len(rows) if c['model']=='chimera_sd21' else 1):raise RuntimeError('Unexpected final adapter count')
  hashes={}
  for p in weights:
   tensors=load_file(str(p))
   if not tensors or not all(torch.isfinite(v).all() for v in tensors.values()):raise RuntimeError('Invalid adapter tensors')
   down=[v for k,v in tensors.items() if '.down.weight' in k or 'lora_A' in k];up=[v for k,v in tensors.items() if '.up.weight' in k or 'lora_B' in k]
   if not down or not all(v.shape[0]==t['rank'] for v in down) or not any(v.abs().max()>0 for v in up):raise RuntimeError('Rank/nonzero adapter validation failed')
   hashes[str(p.relative_to(out))]=sha(p)
  dump(out/'complete.json',dict(passed=True,step=t['steps'],rank=t['rank'],weights=hashes,finished=now(),elapsed_seconds=time.monotonic()-start))
  dump(run/'status.json',dict(**last,training_complete=True))
  if run_wb:run_wb.finish()
 except BaseException:
  if run_wb:run_wb.finish(exit_code=1)
  raise
