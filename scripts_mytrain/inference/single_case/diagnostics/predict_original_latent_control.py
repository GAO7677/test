from pathlib import Path
import sys,importlib.util,hashlib
hist=Path('/home/gaoya/code_V2V_baselines/PhysRVG-main/scripts_mytrain/inference/historical')
spec=importlib.util.spec_from_file_location('reproduce',hist/'reproduce.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
output=Path('/data/gaoya/agent-data/outputs/single_case_20261010/predict25_original_noisy_latent')
original_load=r.load

def load(name,path):
 core=original_load(name,path)
 if name=='historical_predict_core':
  install=core.install_predict
  def install_with_diagnostic(*args,**kwargs):
   install(*args,**kwargs)
   from projects.generic_acn75.inference_runtime import BenchmarkModel
   import torch
   decode=BenchmarkModel.decode
   def wrapped(self,sample,*a,**kw):
    torch.save(sample.detach().cpu(),output/'final_latent.pt')
    y=decode(self,sample,*a,**kw)
    r.write(output/'decoder_hash.json',dict(sha256=hashlib.sha256(y.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()))
    return y
   BenchmarkModel.decode=wrapped
  core.install_predict=install_with_diagnostic
 return core
r.load=load
sys.argv=[str(hist/'predict25_noisy.py'),'--gpu','2','--output',str(output)]
r.launch('predict25','noisy',hist/'predict25_noisy.py')
