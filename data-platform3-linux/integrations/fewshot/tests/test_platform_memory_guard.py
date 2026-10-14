import copy, os, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from fewshot.core import DEFAULTS, device


class MemoryGuard(unittest.TestCase):
 def test_refuse_before_cuda_when_free_memory_dropped(self):
  with tempfile.TemporaryDirectory() as folder:
   c=copy.deepcopy(DEFAULTS);c['paths']['output_root']=folder;c['runtime']['gpu']='GPU-test'
   with patch.dict(os.environ,{'FEWSHOT_GPU_MIN_FREE_MB':'24576','FEWSHOT_GPU_RESERVE_MB':'4096'}),patch('subprocess.check_output',return_value='14000,49140\n'):
    with self.assertRaisesRegex(RuntimeError,'GPU_MEMORY_WAIT'):device(c)

 def test_cap_against_current_free_memory(self):
  with tempfile.TemporaryDirectory() as folder:
   c=copy.deepcopy(DEFAULTS);c['paths']['output_root']=folder;c['runtime']['gpu']='GPU-test'
   captured=[]
   torch=SimpleNamespace(set_num_threads=lambda n:None,cuda=SimpleNamespace(is_available=lambda:True,set_per_process_memory_fraction=lambda f,i:captured.append((f,i))))
   with patch.dict(os.environ,{'FEWSHOT_GPU_MIN_FREE_MB':'24576','FEWSHOT_GPU_RESERVE_MB':'4096'}),patch('subprocess.check_output',return_value='26000,49140\n'),patch.dict('sys.modules',{'torch':torch}):device(c)
   self.assertEqual(captured[0][1],0)
   self.assertAlmostEqual(captured[0][0],(26000-4096)/49140)

if __name__=='__main__':unittest.main()
