"""CPU/GPU-small contract tests; no pretrained weights or production data altered."""
import ast
import math
import os
import unittest
from pathlib import Path

import torch
from diffusers import Cosmos3OmniTransformer, FlowMatchEulerDiscreteScheduler

from common import context_pair, noise_input, future_loss, pack, TARGETS, sample_sigma, inject_lora


class Contracts(unittest.TestCase):
    def test_context_mapping_matches_wan_reference(self):
        source = Path('/home/gaoya/code_V2V_baselines/PhysRVG-main/scripts_mytrain/train/train_full_sa_pybullet_aux.py')
        tree = ast.parse(source.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_context_noise_pair')
        scope = dict(torch=torch, math=math, FlowMatchEulerDiscreteScheduler=FlowMatchEulerDiscreteScheduler)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=5)
        for sigma in [torch.tensor(0.0001), torch.tensor(0.23), torch.tensor(0.9)]:
            for scale in [0., .75, 1.]:
                t, expected, _ = scope['_context_noise_pair'](scheduler, sigma*1000, sigma, scale, device=torch.device('cpu'))
                actual = context_pair(sigma, scale, scheduler.sigmas[:1000])
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(actual*1000, t, rtol=1e-6, atol=1e-5)

    def test_native_waver(self):
        p = Path(__file__).parents[2] / 'cosmos_framework/model/generator/diffusion/rectified_flow.py'
        node = next(n for n in ast.parse(p.read_text()).body if isinstance(n, ast.ClassDef) and n.name=='TrainTimeSampler')
        scope={'torch':torch}
        exec(compile(ast.Module(body=[node], type_ignores=[]),str(p),'exec'),scope)
        g1,g2 = torch.Generator().manual_seed(9),torch.Generator().manual_seed(9)
        for _ in range(10):
            ref=scope['TrainTimeSampler']('waver')(1,generator=g1,shifts=torch.tensor([5.]))[0]
            torch.testing.assert_close(ref,sample_sigma(g2),rtol=0,atol=0)

    def test_noise_and_loss(self):
        clean=torch.randn(1,48,13,4,4); noise=torch.randn_like(clean); sigma=torch.tensor(.6)
        x,target,levels=noise_input(clean,noise,sigma,torch.tensor(0.))
        torch.testing.assert_close(x[:,:,:2],clean[:,:,:2],rtol=0,atol=0)
        pred=target.clone().requires_grad_()
        with torch.no_grad():pred[:,:,:2]+=100;pred[:,:,2:]+=1
        loss=future_loss(pred,target);self.assertEqual(loss.item(),1.)
        loss.backward();self.assertEqual(pred.grad[:,:,:2].abs().max().item(),0.)

    def test_real_model_backward_all_modes(self):
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='5'
        device='cuda'
        torch.manual_seed(42)
        model=Cosmos3OmniTransformer(hidden_size=64,intermediate_size=128,num_hidden_layers=2,
            num_attention_heads=4,num_key_value_heads=2,head_dim=16,vocab_size=256,
            rope_scaling={'mrope_interleaved':True,'mrope_section':[2,3,3],'rope_type':'default'},
            sound_gen=False,action_gen=False).to(device=device,dtype=torch.bfloat16)
        model.requires_grad_(False)
        inject_lora(model,rank=4,alpha=8)
        model.enable_gradient_checkpointing()
        model.train()
        for mode,scale in [('C',0.),('A',1.),('N75',.75)]:
            model.zero_grad(set_to_none=True)
            clean=torch.randn(1,48,13,4,4,device=device)
            x,target,levels=noise_input(clean,torch.randn_like(clean),torch.tensor(.6),torch.tensor(.6*scale))
            args=pack(model,[1,2,3],x.bfloat16(),levels,mode)
            pred=model(**args)[0][0]
            self.assertEqual(pred.shape,clean.shape)
            loss=future_loss(pred,target);loss.backward()
            self.assertTrue(torch.isfinite(loss))
            grads=[p.grad for n,p in model.named_parameters() if 'lora_' in n]
            self.assertTrue(any(g is not None and g.abs().max()>0 for g in grads))
            self.assertTrue(all(p.grad is None for n,p in model.named_parameters() if 'lora_' not in n))


if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main()
