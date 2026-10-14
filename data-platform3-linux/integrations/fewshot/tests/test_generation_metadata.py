import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from PIL import Image

from fewshot.core import DEFAULTS, dump, read, sha
from fewshot.generation import generate, image_metadata
from fewshot.report import build


class GenerationMetadata(unittest.TestCase):
 def setUp(self):
  temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
  self.run=Path(temporary.name)/'run';self.run.mkdir()
  model=self.run.parent/'model';model.mkdir();dump(model/'model_index.json',{})
  adapter=self.run/'adapter';adapter.mkdir()
  weights=adapter/'pytorch_lora_weights.safetensors';weights.write_bytes(b'test-only mock adapter')
  dump(adapter/'complete.json',{'step':400,'weights':{weights.name:sha(weights)}})
  self.config=copy.deepcopy(DEFAULTS)
  self.config['paths']['base_model']=str(model)
  self.config['generate']['num_images']=1
  self.config['train']['resolution']=16
  self.pipe=Mock(return_value=SimpleNamespace(images=[Image.new('RGB',(16,16),'red')]))
  self.pipe.to.return_value=self.pipe
  self.pipe.tokenizer=Mock(return_value=SimpleNamespace(input_ids=[1,2]),model_max_length=77)
  self.pipe.scheduler=SimpleNamespace(config={})
  torch=MagicMock()
  torch.randn.return_value.half.return_value.numpy.return_value.tobytes.return_value=b'same noise'
  patches=[
   patch('fewshot.generation.device',return_value=torch),
   patch('fewshot.generation.verify_inputs',return_value=[{'prompt':'<aircraft>','image':'inputs/group/01.png'}]),
   patch('fewshot.generation.stamp',return_value='batch'),
   patch.dict('sys.modules',{'diffusers':SimpleNamespace(StableDiffusionPipeline=SimpleNamespace(from_pretrained=Mock(return_value=self.pipe)))})]
  for item in patches:item.start();self.addCleanup(item.stop)

 def test_generate_and_resume_preserve_pairs_and_relative_paths(self):
  batch=generate(self.run,self.config)
  records=read(batch/'index.json');self.assertEqual(len(records),2)
  self.assertEqual(records[0]['initial_noise_sha256'],records[1]['initial_noise_sha256'])
  p=batch/'lora/000000.json';meta=read(p)
  self.assertEqual(len(meta),12)
  self.assertNotIn('image_sha256',meta)
  self.assertEqual((p.parent/meta['image']).resolve(),p.with_suffix('.png'))
  self.assertEqual((p.parent/meta['base_model']).resolve(),self.run.parent/'model')
  self.assertEqual((p.parent/meta['adapter_path']).resolve(),self.run/'adapter')
  calls=self.pipe.call_count
  self.assertEqual(generate(self.run,self.config,resume_batch='batch'),batch)
  self.assertEqual(self.pipe.call_count,calls)
  html=(self.run/'report/index.html').read_text()
  self.assertIn('../generations/batch/lora/000000.png',html)
  self.assertIn('../inputs/group/01.png',html)
  self.assertIn('&lt;aircraft&gt;',html)

 def test_index_written_before_sidecar_allows_interrupted_resume(self):
  def interrupted_dump(path,value):
   if Path(path).parent.name=='base' and Path(path).suffix=='.json':raise OSError('interrupted sidecar write')
   dump(path,value)
  with patch('fewshot.generation.dump',side_effect=interrupted_dump):
   with self.assertRaisesRegex(OSError,'interrupted'):generate(self.run,self.config)
  batch=self.run/'generations/batch'
  self.assertEqual(len(read(batch/'index.json')),1)
  self.assertFalse((batch/'base/000000.json').exists())
  generate(self.run,self.config,resume_batch='batch')
  self.assertEqual(read(batch/'complete.json')['noise_pairs_verified'],1)
  self.assertEqual(len(read(batch/'index.json')),2)

 def test_modified_sidecar_rejected(self):
  batch=generate(self.run,self.config);p=batch/'lora/000000.json'
  row=read(p);row['seed']=99;dump(p,row)
  with self.assertRaisesRegex(ValueError,'mismatch'):generate(self.run,self.config,resume_batch='batch')

 def test_modified_image_rejected(self):
  batch=generate(self.run,self.config)
  (batch/'lora/000000.png').write_bytes(b'corrupt')
  with self.assertRaisesRegex(ValueError,'mismatch'):generate(self.run,self.config,resume_batch='batch')

 def test_old_full_sidecars_resume_without_index(self):
  batch=generate(self.run,self.config)
  for row in read(batch/'index.json'):dump(batch/Path(row['image']).with_suffix('.json'),row)
  (batch/'index.json').unlink();calls=self.pipe.call_count
  generate(self.run,self.config,resume_batch='batch')
  self.assertEqual(self.pipe.call_count,calls)
  self.assertEqual(len(read(batch/'index.json')),2)

 def test_missing_index_blocks_resume_but_preview_still_works(self):
  batch=generate(self.run,self.config);(batch/'index.json').unlink()
  with self.assertRaisesRegex(ValueError,'Missing batch index'):generate(self.run,self.config,resume_batch='batch')
  self.assertIn('../generations/batch/lora/000000.png',build(self.run).read_text())

 def test_chimera_conditioning_details_retained_only_when_used(self):
  row={'image':'lora/000000.png','prompt':'aircraft','seed':0,'negative_prompt':'clouds',
       'adapter_heads':['head_0','head_1'],'adapter_fusion_weights':[0.25,0.75],
       'label':'aircraft','label_source':'user_condition_unverified','caption_source_image':None}
  meta=image_metadata(row,self.run/'000000.json')
  self.assertEqual(meta['adapter_fusion_weights'],[0.25,0.75])
  self.assertEqual(meta['adapter_heads'],['head_0','head_1'])
  self.assertEqual(meta['label_source'],'user_condition_unverified')
  self.assertEqual(meta['negative_prompt'],'clouds')
  self.assertNotIn('caption_source_image',meta)


if __name__=='__main__':unittest.main()
