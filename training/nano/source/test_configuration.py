import copy
import json
import tempfile
import unittest
from pathlib import Path
import torch
from common import CFG, inject_lora
from settings import load_config

class ConfigurationTests(unittest.TestCase):
    def model(self):
        with torch.device('meta'):
            m=torch.nn.Module()
            m.self_attn=torch.nn.Module()
            m.self_attn.add_q_proj=torch.nn.Linear(8,8,bias=False)
            m.self_attn.to_q=torch.nn.Linear(8,8,bias=False)
            m.mlp_moe_gen=torch.nn.Module()
            m.mlp_moe_gen.up_proj=torch.nn.Linear(8,16,bias=False)
            m.requires_grad_(False)
        return m

    def test_target_selection_and_rank(self):
        m=self.model()
        names=inject_lora(m,rank=3,alpha=9,targets=['mlp_moe_gen.up_proj'],dropout=0.2)
        self.assertEqual(names,['mlp_moe_gen.up_proj'])
        self.assertEqual(sum(p.numel() for p in m.parameters() if p.requires_grad),72)
        self.assertEqual(m.mlp_moe_gen.up_proj.scale,3)
        self.assertEqual(m.mlp_moe_gen.up_proj.dropout.p,0.2)
        self.assertFalse(m.self_attn.to_q.weight.requires_grad)

    def test_bad_target_does_not_partially_mutate(self):
        m=self.model()
        with self.assertRaises(ValueError):inject_lora(m,targets=['self_attn.add_q_proj','typo'])
        self.assertIsInstance(m.self_attn.add_q_proj,torch.nn.Linear)

    def test_invalid_configuration(self):
        for section,key,value in [('lora','rank',0),('runtime','train_gpus',[4,5]),('training','per_gpu_batch',2),('training','lr_scheduel','cosine')]:
            c=copy.deepcopy(CFG);c[section][key]=value
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'test.json';p.write_text(json.dumps(c))
                with self.assertRaises(ValueError):load_config(p)

if __name__=='__main__':unittest.main()
