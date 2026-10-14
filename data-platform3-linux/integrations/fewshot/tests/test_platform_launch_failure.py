import json, subprocess, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from fewshot import cli
from fewshot.core import dump, now


class LaunchFailure(unittest.TestCase):
 def test_preserve_child_oom_instead_of_called_process_error(self):
  with tempfile.TemporaryDirectory() as folder:
   run=Path(folder)
   def launch(*args,**kwargs):
    dump(run/'failure.json',dict(error='OutOfMemoryError: CUDA out of memory',traceback='child failure',time=now()))
    raise subprocess.CalledProcessError(1,['python','_train'])
   with patch.object(sys,'argv',['run.py','train','--config','unused']),patch.object(cli,'config',return_value={'paths':{'adapter':None}}),patch.object(cli,'prepare',return_value=run),patch.object(cli,'launch',side_effect=launch):
    with self.assertRaises(subprocess.CalledProcessError):cli.main()
   saved=json.loads((run/'failure.json').read_text())
   self.assertIn('OutOfMemoryError',saved['error'])
   self.assertIn('CalledProcessError',saved['launcher_error'])
   self.assertIn('OutOfMemoryError',json.loads((run/'status.json').read_text())['error'])

if __name__=='__main__':unittest.main()
