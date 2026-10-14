import builtins
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
from fewshot.core import config, resolve_path
from fewshot.tracking import start_run

class TrackingPaths(unittest.TestCase):
 def test_environment_relative_path_uses_launch_directory(self):
  with patch.dict(os.environ, FEWSHOT_DATA='asset/my_data'):
   self.assertEqual(resolve_path('${FEWSHOT_DATA}', '/tmp/configs'), str(Path('asset/my_data').resolve()))
 def test_literal_relative_path_keeps_config_semantics(self):
  self.assertEqual(resolve_path('../data', '/tmp/configs'), '/tmp/data')
 def test_default_tracking_is_off(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'config.yaml';p.write_text('paths:\n  base_model: /tmp/model\n  output_root: /tmp/out\ndata:\n  root: /tmp/input\n')
   self.assertFalse(config(p)['tracking']['enabled'])
 def test_disabled_tracking_never_imports_wandb(self):
  original=builtins.__import__
  def guarded(name,*args,**kwargs):
   if name=='wandb':raise AssertionError('Attempted to import W&B while disabled')
   return original(name,*args,**kwargs)
  with patch('builtins.__import__', side_effect=guarded):
   for settings,smoke in [({},False),({'enabled':False},False),({'enabled':True},True)]:
    self.assertIsNone(start_run(settings,name='test',directory='/tmp',config={},smoke=smoke))
 def test_offline_mode_updates_preloaded_hub(self):
  import copy,sys,types
  from fewshot.core import DEFAULTS,device
  with tempfile.TemporaryDirectory() as tmp:
   c=copy.deepcopy(DEFAULTS);c['paths']['output_root']=tmp
   constants=types.SimpleNamespace(HF_HUB_OFFLINE=False);torch=Mock()
   with patch.dict(sys.modules,{'huggingface_hub.constants':constants,'torch':torch}),patch.dict(os.environ):
    device(c)
    self.assertTrue(constants.HF_HUB_OFFLINE)
    self.assertEqual(os.environ['HF_HUB_OFFLINE'],'1')
 def test_explicit_enable_initializes_metrics_only(self):
  wb=Mock();wb.init.return_value.offline=False
  with patch.dict('sys.modules',wandb=wb):
   run=start_run({'enabled':True,'project':'test'},name='test',directory='/tmp',config={})
  self.assertIs(run,wb.init.return_value)
  self.assertEqual(wb.init.call_args.kwargs['mode'],'online')
  wb.Settings.assert_called_once_with(disable_code=True,disable_git=True)
