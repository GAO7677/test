import copy,json,tempfile,unittest
from pathlib import Path
from PIL import Image
from fewshot.core import DEFAULTS,inventory,config
from fewshot.report import build

class DataContract(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  self.c=copy.deepcopy(DEFAULTS);self.c['data']['root']=str(self.root);self.c['train']['steps']=400
  for i in range(2):
   p=self.root/f'group{i}';p.mkdir();(p/'caption.txt').write_text(f'aircraft over scene {i}');Image.new('RGB',(32,32),(i*100,0,0)).save(p/'a.png')
 def test_captions_do_not_become_labels(self):
  rows,_=inventory(self.c);self.assertEqual(len(rows),2);self.assertEqual(rows[1]['prompt'],'aircraft over scene 1');self.assertIsNone(rows[0]['label'])
 def test_duplicate_pixels_rejected(self):
  Image.new('RGB',(32,32),(0,0,0)).save(self.root/'group1/a.png')
  with self.assertRaisesRegex(ValueError,'Duplicate'):inventory(self.c)
 def test_empty_caption_rejected(self):
  (self.root/'group1/caption.txt').write_text(' ')
  with self.assertRaisesRegex(ValueError,'Empty caption'):inventory(self.c)
 def test_label_mapping_explicit(self):
  p=self.root/'labels.json';p.write_text(json.dumps({'group0':'medium','group1':'small'}));self.c['data']['labels']=str(p)
  rows,_=inventory(self.c);self.assertEqual([r['label'] for r in rows],['medium','small'])
 def test_order_and_selected_groups(self):
  self.c['data']['groups']=['group1','group0'];rows,_=inventory(self.c);self.assertEqual(rows[0]['group'],'group1')
 def test_incomplete_epoch_rejected(self):
  self.c['train'].update(steps=3,batch_size=1)
  with self.assertRaisesRegex(ValueError,'whole epochs'):inventory(self.c)
 def test_unknown_config_rejected(self):
  p=self.root/'bad.yaml';p.write_text('model: beyondobjects_sd21\ntrain:\n  batc_size: 4\n')
  with self.assertRaisesRegex(ValueError,'Unknown'):config(p)
 def test_report_escapes_prompt(self):
  g=self.root/'generations'/'batch'/'lora';g.mkdir(parents=True);(g/'0.json').write_text(json.dumps(dict(index=0,seed=0,prompt='<script>bad()</script>',kind='lora',image='lora/0.png')))
  (self.root/'status.json').write_text('{}');html=build(self.root).read_text();self.assertIn('&lt;script&gt;',html);self.assertNotIn('<script>',html)
if __name__=='__main__':unittest.main()
