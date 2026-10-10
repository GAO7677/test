"""CPU contract tests: context participates in native updates and is never restored."""
import tempfile,types,unittest
from pathlib import Path
import torch
from diffusers import FlowMatchEulerDiscreteScheduler,UniPCMultistepScheduler
from clean_first_hooks import WanCleanFree as WanInitialOnly,NanoCleanFree as NanoInitialOnly

class FreeContextTests(unittest.TestCase):
 def test_wan_and_nano_carried_state(self):
  for backbone in ['wan','nano']:
   for mode in ['equal_sigma','noisy']:
    with self.subTest(backbone=backbone,mode=mode), tempfile.TemporaryDirectory() as dest:
     torch.manual_seed(42);clean=torch.randn(1,2,13,2,2);noise=torch.randn_like(clean)
     initial=noise.clone();initial[:,:,:2]=clean[:,:,:2]
     scheduler=UniPCMultistepScheduler(prediction_type='flow_prediction',use_flow_sigmas=True,flow_shift=5)
     scheduler.set_timesteps(3)
     pipe=types.SimpleNamespace(scheduler=scheduler,transformer=torch.nn.Identity(),do_classifier_free_guidance=True)
     if backbone=='wan':
      mask=torch.ones(1,1,13,2,2);mask[:,:,:2]=0
      pipe.prepare_latents=lambda: (initial.clone(),clean,mask,noise)
      train=FlowMatchEulerDiscreteScheduler(shift=5);train.set_timesteps(1000)
      hook=WanInitialOnly(pipe,train,1 if mode=='equal_sigma' else .75,(1,1,1));hook.output=Path(dest)
     else:
      pipe.prepare_latents=lambda: (initial.clone(),)
      pipe._mask_velocity_predictions=lambda pv,ps,vm,*a,**k:(pv[0]*0,None,None)
      hook=NanoInitialOnly(pipe,mode,'N75',dict(num_train_timesteps=1000,shift=5),20261007);hook.output=Path(dest)
     with hook:
      sample=pipe.prepare_latents()[0]
      for i,t in enumerate(scheduler.timesteps):
       if backbone=='wan':
        _,kw=hook.before(None,(),dict(hidden_states=sample,timestep=torch.full((1,52),float(t))))
        prediction=kw['hidden_states']*.25+.1
        sample=pipe.scheduler.step(prediction,t,sample,return_dict=False)[0]
       else:
        kw=dict(vision_tokens=[sample],vision_token_shapes=[(13,2,2)],vision_sequence_indexes=[torch.arange(52)])
        _,actual=hook.before(None,(),kw);hook.before(None,(),kw)
        raw=actual['vision_tokens'][0]*.25+.1
        pred=pipe._mask_velocity_predictions([raw],None,[torch.ones_like(raw)])[0]
        assert torch.equal(pred,raw)
        sample=pipe.scheduler.step(pred.unsqueeze(0),t,sample.unsqueeze(0),return_dict=False)[0].squeeze(0)
      assert not torch.equal(sample[:,:,:2],clean[:,:,:2])
     assert (Path(dest)/'initial_only_audit.json').is_file()
     assert torch.equal(torch.load(Path(dest)/'free_final_latent.pt',weights_only=True),sample)

 def test_predict_cfg_initializes_only_once(self):
  import sys
  from dataclasses import dataclass,replace
  from unittest.mock import patch
  from clean_first_hooks import install_predict_free as install_predict_initial
  @dataclass
  class Condition:
   gt_frames: torch.Tensor
   condition_video_input_mask_B_C_T_H_W: torch.Tensor
   def to_dict(self): return {}
  class Model:
   def decode(self,sample):return sample
   def denoise(self,noise,x,t,condition):return self.net(x_B_C_T_H_W=x)
  fake=types.ModuleType('projects.generic_acn75.inference_runtime');fake.BenchmarkModel=Model
  for variant in ['equal_sigma','noisy']:
   with tempfile.TemporaryDirectory() as dest, patch.dict(sys.modules,{'projects.generic_acn75.inference_runtime':fake}):
    Model.decode=lambda self,sample:sample
    Model.denoise=lambda self,noise,x,t,condition:self.net(x_B_C_T_H_W=x)
    install_predict_initial(variant,'N75',42,str(Path(dest)/'audit.json'),None,dict(shift=5))
    model=Model();model.config=types.SimpleNamespace(conditional_frame_timestep=.1,denoise_replace_gt_frames=True);model.sample_scheduler=UniPCMultistepScheduler(prediction_type='flow_prediction',use_flow_sigmas=True)
    model.sample_scheduler.set_timesteps(3);model.tensor_kwargs={'dtype':torch.float32};model.rectified_flow=types.SimpleNamespace(num_train_timesteps=1000)
    model.net=lambda x_B_C_T_H_W,**kw:x_B_C_T_H_W*.2+.1
    noise=torch.randn(1,2,13,2,2);sample=noise
    condition=Condition(torch.randn_like(noise),torch.ones(1,1,13,2,2))
    for t in model.sample_scheduler.timesteps:
     v=model.denoise(noise,sample,t.reshape(1),condition)
     before=sample.clone();v2=model.denoise(noise,sample,t.reshape(1),condition)
     assert torch.equal(sample,before) and torch.equal(v,v2)
     sample=model.sample_scheduler.step(v.unsqueeze(0),t,sample.unsqueeze(0),return_dict=False)[0].squeeze(0)
    assert torch.equal(model.decode(sample),sample)

if __name__=='__main__':unittest.main()
