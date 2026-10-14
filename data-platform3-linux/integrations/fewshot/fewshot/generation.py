"""Paired generation with concise image sidecars and a full batch index."""
import hashlib, os, time
from pathlib import Path
from .core import verify_inputs,device,dump,read,sha,stamp,now


def image_metadata(meta,json_path):
 """Public image metadata; every file path is relative to its JSON file."""
 fields=('image','prompt','seed','model','base_model','adapter_path','training_step',
         'width','height','sampling_steps','guidance_scale','scheduler',
         'negative_prompt','label','label_source','adapter_heads','adapter_fusion_weights')
 result={key:meta[key] for key in fields if meta.get(key) is not None}
 result['image']=Path(meta['image']).name
 for key in ('base_model','adapter_path'):
  if key in result and Path(result[key]).is_absolute():
   result[key]=os.path.relpath(result[key],Path(json_path).parent)
 return result


def existing_record(sidecar,audit,prompt,seed,adapter):
 saved=read(sidecar);record=audit if audit is not None else saved
 if 'image_sha256' not in record:
  raise ValueError('Missing batch index.json; cannot verify existing image')
 if (record['prompt']!=prompt or record['seed']!=seed
     or record['adapter_weights_sha256']!=adapter
     or sha(sidecar.with_suffix('.png'))!=record['image_sha256']
     or (saved!=record and saved!=image_metadata(record,sidecar))):
  raise ValueError('Existing image/metadata mismatch; refusing to overwrite')
 return record


def generate(run,c,prompt=None,num=None,resume_batch=None):
 torch=device(c);g=dict(c['generate']);g['prompt']=prompt if prompt is not None else g['prompt'];g['num_images']=num if num is not None else g['num_images']
 if g['num_images']<1:raise ValueError('num_images must be positive')
 adapter=Path(c['paths']['adapter']) if c['paths']['adapter'] else run/'adapter'
 trained=read(adapter/'complete.json') if (adapter/'complete.json').exists() else None
 if not trained and not c['paths']['adapter']:raise RuntimeError('Train or import an adapter first')
 rows=verify_inputs(run);flux=c['model']=='flux2_klein_base_4b';chimera=c['model']=='chimera_sd21'
 if chimera:
  if (adapter/'support').is_dir():weights=[adapter/'support'/f'B{i}_each/pytorch_lora_weights.safetensors' for i in range(len(rows))]
  else:
   counts={};weights=[]
   for r in rows:
    label=r.get('label')
    if label is None:raise ValueError('Imported class-wise Chimera adapters require explicit input labels')
    i=counts.get(label,0);weights.append(adapter/label/f'B{i}_each/pytorch_lora_weights.safetensors');counts[label]=i+1
 else:weights=[adapter/'pytorch_lora_weights.safetensors']
 provenance={str(p.relative_to(adapter)):sha(p) for p in weights}
 if trained and trained.get('weights') and provenance!=trained['weights']:raise ValueError('Adapter hash changed since training')
 from diffusers import StableDiffusionPipeline
 if flux:
  from diffusers import Flux2KleinPipeline
  pipe=Flux2KleinPipeline.from_pretrained(c['paths']['base_model'],torch_dtype=torch.bfloat16,local_files_only=True).to('cuda')
 else:pipe=StableDiffusionPipeline.from_pretrained(c['paths']['base_model'],torch_dtype=torch.float16,local_files_only=True).to('cuda')
 pipe.set_progress_bar_config(disable=True)
 batch=run/'generations'/(resume_batch or stamp())
 if batch.parent.resolve()!=(run/'generations').resolve():raise ValueError('resume-batch must be a batch ID within this run')
 if g.get('plan'):g['plan_sha256']=sha(g['plan'])
 if resume_batch:
  if read(batch/'config.json')!=g:raise ValueError('Generation config/plan changed; create a new batch')
 else:
  batch.mkdir(parents=True);dump(batch/'config.json',g)
 import fcntl
 lock=(batch/'generation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 dump(batch/'scheduler.json',dict(pipe.scheduler.config))
 audit={(r['kind'],r['index']):r for r in read(batch/'index.json')} if (batch/'index.json').exists() else {}
 noise_hash={}
 if flux:
  original=pipe.prepare_latents
  def prepare(*args,**kw):
   result=original(*args,**kw);t=result[0].detach().cpu().contiguous();noise_hash['sha256']=hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest();return result
  pipe.prepare_latents=prepare
 plan=read(g['plan']) if g.get('plan') else None
 if plan:
  if prompt is not None or num is not None:raise ValueError('A frozen generation plan cannot be overridden by prompt/count')
  g['num_images']=len(plan)
 def applied_weights(label):
  if not chimera:return provenance
  return {str(weights[j].relative_to(adapter)):provenance[str(weights[j].relative_to(adapter))] for j,r in enumerate(rows) if label is None or r.get('label')==label}
 records=[];start=time.monotonic();total=g['num_images']*(2 if g['compare_base'] else 1)
 for kind in (['base','lora'] if g['compare_base'] else ['lora']):
  if kind=='lora':
   if chimera:
    for i,p in enumerate(weights):pipe.load_lora_weights(str(p.parent),adapter_name=f'head_{i}',weight_name=p.name)
   else:pipe.load_lora_weights(str(adapter),weight_name='pytorch_lora_weights.safetensors')
  dest=batch/kind;dest.mkdir(exist_ok=True)
  for i in range(g['num_images']):
   row=rows[i%len(rows)];seed=g['seed']+i;text=g['prompt'] if g['prompt'] is not None else row['prompt'];label=g['class_name'] if g['prompt'] is not None else row.get('label')
   if plan:
    text=plan[i]['prompt'];seed=plan[i]['seed'];label=plan[i].get('label')
   existing=dest/f'{i:06d}.json'
   if existing.exists():
    expected_adapter=applied_weights(label) if kind=='lora' else None
    saved=existing_record(existing,audit.get((kind,i)),text,seed,expected_adapter)
    audit[kind,i]=saved
    if kind=='lora' and g['compare_base'] and audit['base',i]['initial_noise_sha256']!=saved['initial_noise_sha256']:raise RuntimeError('Paired noise mismatch')
    records.append(saved);continue
   token_text=text
   if flux:token_text=pipe.tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
   token_count=len(pipe.tokenizer(token_text,truncation=False).input_ids);token_limit=512 if flux else pipe.tokenizer.model_max_length
   if token_count>token_limit and not c['data']['allow_caption_truncation']:raise ValueError(f'Generation prompt exceeds token limit {token_limit}: {token_count}')
   adapter_weights=None;head_names=None
   if kind=='lora' and chimera:
    ids=[j for j,r in enumerate(rows) if label is None or r.get('label')==label]
    if not ids:raise ValueError(f'No Chimera heads for explicit class_name: {label}')
    if len(weights)!=len(rows):raise ValueError('Chimera imported heads must match input manifest order')
    # Dedicated fusion RNG: independent of image noise and base comparison.
    with torch.random.fork_rng(devices=[0]):
     torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
     adapter_weights=torch.distributions.Dirichlet(torch.full((len(ids),),g['dirichlet_alpha'],device='cuda',dtype=torch.float32)).sample().tolist()
    if plan and plan[i].get('adapter_weights') is not None:
     adapter_weights=plan[i]['adapter_weights']
     if len(adapter_weights)!=len(ids) or any(w<0 for w in adapter_weights) or abs(sum(adapter_weights)-1)>1e-5:raise ValueError('Invalid frozen fusion weights')
    head_names=[f'head_{j}' for j in ids];pipe.set_adapters(head_names,adapter_weights=adapter_weights)
   kwargs=dict(prompt=text,height=c['train']['resolution'],width=c['train']['resolution'],num_inference_steps=g['steps'],guidance_scale=g['guidance_scale'])
   if flux:kwargs.update(generator=torch.Generator('cpu').manual_seed(seed),max_sequence_length=512,text_encoder_out_layers=(9,18,27))
   else:
    n=c['train']['resolution']//8
    noise=torch.randn((1,4,n,n),generator=torch.Generator('cpu').manual_seed(seed),dtype=torch.float32).half()
    noise_hash['sha256']=hashlib.sha256(noise.numpy().tobytes()).hexdigest();kwargs['latents']=noise.to('cuda')
    if g['negative_prompt'] is not None:kwargs['negative_prompt']=g['negative_prompt']
   with torch.inference_mode():image=pipe(**kwargs).images[0]
   filename=f'{i:06d}.png';p=dest/filename;image.save(p)
   meta=dict(schema_version=1,image=str(p.relative_to(batch)),image_sha256=sha(p),created=now(),run_id=run.name,batch_id=batch.name,pair_id=f'{batch.name}:{i}',kind=kind,index=i,prompt=text,negative_prompt=g['negative_prompt'],seed=seed,initial_noise_sha256=noise_hash['sha256'],width=image.width,height=image.height,model=c['model'],base_model=c['paths']['base_model'],base_model_config_sha256=sha(Path(c['paths']['base_model'])/'model_index.json'),adapter_weights_sha256=applied_weights(label) if kind=='lora' else None,training_step=trained.get('step') if trained and kind=='lora' else None,adapter_path=str(adapter) if kind=='lora' else None,sampling_steps=g['steps'],guidance_scale=g['guidance_scale'],scheduler=pipe.scheduler.__class__.__name__,scheduler_config_sha256=sha(batch/'scheduler.json'),adapter_fusion_weights=adapter_weights,adapter_heads=head_names,caption_source_image=row['image'] if g['prompt'] is None and not plan else None,label=label,label_source='user_condition_unverified' if label is not None else None,reference_image_used=False,prompt_token_count=token_count,prompt_token_limit=token_limit,prompt_truncated=token_count>token_limit,inference_dtype='bfloat16' if flux else 'float16')
   if kind=='lora' and g['compare_base']:
    base=audit['base',i]
    if base['initial_noise_sha256']!=meta['initial_noise_sha256']:raise RuntimeError('Paired noise mismatch')
   audit[kind,i]=meta
   # Save the full record first so a compact sidecar always has recovery evidence.
   dump(batch/'index.json',list(audit.values()))
   dump(p.with_suffix('.json'),image_metadata(meta,p.with_suffix('.json')));records.append(meta)
   dump(run/'status.json',dict(phase='generating',training_complete=trained is not None,step=trained.get('step') if trained else None,images=len(records),total_images=total,batch_id=batch.name,elapsed_seconds=time.monotonic()-start))
   print('GENERATED',kind,i,flush=True)
 if provenance!={str(p.relative_to(adapter)):sha(p) for p in weights}:raise RuntimeError('Adapter changed during generation')
 dump(batch/'index.json',records);dump(batch/'complete.json',dict(passed=True,images=len(records),noise_pairs_verified=g['num_images'] if g['compare_base'] else 0,elapsed_seconds=time.monotonic()-start))
 dump(run/'status.json',dict(phase='complete',step=trained.get('step') if trained else None,images=len(records),total_images=total,batch_id=batch.name))
 from .report import build
 build(run);return batch
