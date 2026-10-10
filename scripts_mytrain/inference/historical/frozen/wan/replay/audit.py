import csv,hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=Path('/home/gaoya/code_V2V_baselines/PhysRVG-main')
COMMIT='8a8c4f523a0cadb94ab7c9f1e0620a4847b92991'
CASES=['v2v_pendulum_l110','v2v_seesaw_x058','v2v_obstacle_v160','v2v_bowl_r180']
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(path,obj):Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def readcsv(path):return list(csv.DictReader(Path(path).open(encoding='utf-8-sig')))
def main():
 for name in ['reference_source','replay_states','attention_arrays','figures','logs']: (ROOT/name).mkdir(exist_ok=True)
 files=['LICENSE','PusaV1/diffsynth/models/wan_video_pusa.py','PusaV1/diffsynth/models/wan_video_dit.py','PusaV1/diffsynth/pipelines/wan_video_pusa_multi_frames.py']
 sources=[]
 for file in files:
  url=f'https://raw.githubusercontent.com/Yaofang-Liu/Pusa-VidGen/{COMMIT}/{file}'
  dest=ROOT/'reference_source'/file;dest.parent.mkdir(parents=True,exist_ok=True)
  with urllib.request.urlopen(url,timeout=60) as response:dest.write_bytes(response.read())
  sources.append(dict(url=url,path=str(dest),sha256=sha(dest)))
 mapping=readcsv('/data/gaoya/agent-data/outputs/test70_noise_family_diagnostic_20260925/task_mapping.csv')
 white=readcsv('/data/gaoya/agent-data/outputs/context_noise_lambda_metrics_20260925/abc_n25_n50_n75_all_checkpoint_metrics.csv')
 allowed={(r['method_code'],r['model_key'],r['step']) for r in white};models={};cases={};tasks=[]
 for r in mapping:
  if r['code'] not in ['C','A','B','N75'] or r['step']!='1000':continue
  assert (r['code'],r['model_key'],r['step']) in allowed
  cp=Path(r['source_checkpoint']);cfg=json.loads((cp/'adapter_config.json').read_text())
  models[r['code']]=dict(**r,adapter_config=cfg,files=[dict(path=str(p),sha256=sha(p)) for p in cp.glob('*') if p.is_file()])
  for case in CASES:
   meta_path=Path(r['result_root'])/(case+'.json');meta=json.loads(meta_path.read_text())
   assert Path(meta['model']['lora_checkpoint'])==cp
   assert meta['inference']['context_noise_mode']=='clean' and not meta['inference']['classifier_free_guidance_enabled']
   c=dict(prompt=meta['prompt'],context_video=meta['input']['video_path'])
   if case in cases:assert cases[case]==c
   cases[case]=c;tasks.append(dict(method=r['code'],step=1000,model_key=r['model_key'],task_id=r['task_id'],case=case,metadata_path=str(meta_path),metadata_sha256=sha(meta_path),adapter_path=str(cp)))
   base=meta['model']['dit_checkpoint'];model_id=meta['model']['model_id']
 assert len(models)==4 and len(cases)==4
 cfg=dict(models=models,cases=cases,base=dict(path=base,sha256=sha(base),model_id=model_id),reference_source=dict(commit=COMMIT,files=sources),seeds=[42,202],targets=[.9,.5,.1],steps=40,height=512,width=896,num_frames=49,context_latents=[0,1],future_latents=list(range(2,13)),cfg=False,queries_per_frame=32,query_rng_seed=1729,gpu_allowed=[0,1,2,3],gpu=0,cpu_affinity=[0,1],auxiliary=dict(case=CASES[0],seed=42,reference_owner='B'),ranking='mean_abs_delta_context_mass; case equal weight; stages and reference owners separate')
 save(ROOT/'resolved_config.json',cfg)
 with (ROOT/'task_manifest.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=tasks[0]);w.writeheader();w.writerows(tasks)
 print('Pinned sources and 16 exact task/case links; base SHA',cfg['base']['sha256'])
if __name__=='__main__':main()
