import sys,os,json,types,hashlib
from pathlib import Path
root=Path('/home/gaoya/code_V2V_baselines/PhysRVG-main');sys.path.insert(0,str(root/'scripts_mytrain/inference/single_case'))
from common import HIST,save
native=Path('/home/gaoya/Code_Video/Code_MODEL/cosmos-predict2.5');os.chdir(native);sys.path[:0]=[str(native),str(native/'packages/cosmos-oss')]
import projects
package=types.ModuleType('projects.generic_acn75');package.__path__=[str(HIST/'frozen/predict25')];sys.modules['projects.generic_acn75']=package;projects.generic_acn75=package
import torch
from projects.generic_acn75.inference_runtime import native_config
from cosmos_predict2._src.imaginaire.lazy_config import instantiate
from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video
run=Path(sys.argv[1]);cfg=json.loads((run/'config.json').read_text());native_cfg=native_config(cfg['settings'])
vae=instantiate(native_cfg.model.config.tokenizer)
z=torch.load(run/'final_latent.pt',map_location='cuda',weights_only=True)
report=[]
for label,benchmark,deterministic in [('historical_1',True,False),('historical_2',True,False),('deterministic',False,True)]:
 torch.backends.cudnn.benchmark=benchmark;torch.backends.cudnn.deterministic=deterministic
 torch.backends.cudnn.allow_tf32=torch.backends.cuda.matmul.allow_tf32=True
 with torch.inference_mode():y=vae.decode(z)
 digest=hashlib.sha256(y.float().cpu().contiguous().numpy().tobytes()).hexdigest()
 save_img_or_video((1+y[0])/2,str(run/('decode_'+label)),fps=cfg['fps'])
 if report:delta=(y-first).float();stats=dict(max_abs=float(delta.abs().max()),mse=float(delta.square().mean()))
 else:first=y.detach().clone();stats={}
 report.append(dict(label=label,decoded_sha256=digest,equals_original=digest==json.loads((run/'result.json').read_text())['audit']['decoded_tensor_sha256'],**stats))
 save(run/'decode_probe.json',report)
print(json.dumps(report),flush=True)
