"""CPU checks for extracted noise contracts and explicit input overrides."""
import ast,json,subprocess,sys,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;HIST=HERE.parent/'historical'

def node(path,name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)

class Contracts(unittest.TestCase):
    def test_wan_hook_unchanged(self):
        self.assertEqual(ast.dump(node(HERE/'sampling.py','ContextNoise')),ast.dump(node(HIST/'wan_noisy.py','ContextNoise')))
    def test_clean_nano_hook_unchanged(self):
        self.assertEqual(ast.dump(node(HERE/'sampling.py','CleanContextAudit')),ast.dump(node(HIST/'nano_noisy.py','CleanContextAudit')))
    def test_nano_only_equal_time_branch_changed(self):
        old=ast.get_source_segment((HIST/'frozen/nano/v2/hooks.py').read_text(),node(HIST/'frozen/nano/v2/hooks.py','NanoContext'))
        old=old.replace("if self.variant=='matched' and self.mode=='A':tc=tf","if self.variant=='equal_sigma':tc=tf")
        self.assertEqual(ast.dump(ast.parse(old).body[0]),ast.dump(node(HERE/'sampling.py','NanoContext')))
    def test_three_modes_all_backbones(self):
        for backbone in ['wan','nano','predict25']:
            for mode in ['clean','noisy','equal_sigma']:
                result=subprocess.run([sys.executable,str(HERE/(backbone+'.py')),'--check-only','--mode',mode,'--output','/data/gaoya/agent-data/outputs/single_case_contract_check'],capture_output=True,text=True,check=True)
                self.assertEqual(json.loads(result.stdout)['mode'],mode)
    def test_new_prompt_drops_old_cache(self):
        result=subprocess.run([sys.executable,str(HERE/'wan.py'),'--check-only','--prompt','A different scene.','--output','/data/gaoya/agent-data/outputs/single_case_contract_check'],capture_output=True,text=True,check=True)
        cfg=json.loads(result.stdout)
        self.assertIsNone(cfg['context_cache']);self.assertIsNone(cfg['prompt_cache'])

if __name__=='__main__':unittest.main()
