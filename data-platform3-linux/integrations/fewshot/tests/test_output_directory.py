import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fewshot.core import DEFAULTS, prepare, load_run

class OutputDirectory(unittest.TestCase):
 def test_direct_output_and_reload(self):
  for precreate in (False,True):
   with self.subTest(precreate=precreate), tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);out=root/'experiment';model=root/'model';model.mkdir();(model/'model_index.json').write_text('{}')
    if precreate:out.mkdir()
    c=copy.deepcopy(DEFAULTS);c['paths'].update(output_root=str(out),base_model=str(model))
    with patch('fewshot.core.check',return_value=({'passed':True},[])),patch('fewshot.core.versions',return_value={}):
     run=prepare(c)
    self.assertEqual(run,out)
    self.assertTrue((out/'adapter').is_dir())
    self.assertTrue((out/'resolved_config.yaml').is_file())
    self.assertEqual(load_run(out)[1]['paths']['output_root'],str(out))
    before=(out/'resolved_config.yaml').read_bytes()
    with self.assertRaisesRegex(ValueError,'not empty'):prepare(c)
    self.assertEqual((out/'resolved_config.yaml').read_bytes(),before)
 def test_unrelated_files_are_not_overwritten(self):
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp);(out/'keep.txt').write_text('keep')
   c=copy.deepcopy(DEFAULTS);c['paths']['output_root']=str(out)
   with patch('fewshot.core.check') as check:
    with self.assertRaisesRegex(ValueError,'not empty'):prepare(c)
    check.assert_not_called()
   self.assertEqual((out/'keep.txt').read_text(),'keep')
