"""Regression checks for native labels and sidecar-aware file counts."""
import sys
import tempfile
import unittest
import json
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.label_schema import label_names
from app.services.version_service import _file_count


class DemoImportChecks(unittest.TestCase):
    def test_native_and_generated_labels(self):
        self.assertEqual(label_names({'annotations': [{'type':'bbox', 'label':'707-320'}], 'category':'707-320'}), ['707-320'])
        self.assertEqual(label_names({'boxes':[{'label':'airplane'}, {'label':'airplane'}]}), ['airplane', 'airplane'])
        self.assertEqual(label_names({'entities':[{'type':'机构','text':'Research Lab'}]}), ['机构'])
        self.assertEqual(label_names({'category':'Small Civil', 'caption':'an aircraft'}), ['Small Civil'])
        self.assertEqual(label_names({'keywords':[{'word':'aircraft'}]}), ['aircraft'])
        self.assertEqual(label_names([{'type':'bbox','xyxy':[0,0,1,1]}]), [])

    def test_sidecars_and_nested_files(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            (root/'a').mkdir();(root/'b').mkdir()
            for rel in ['a/image.jpg','a/image_label.json','b/image.jpg','_manifest.json']:
                (root/rel).write_text('example')
            self.assertEqual(_file_count(root),2)

    def test_demo_manifest_can_rebuild_catalog(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
        from build_material_index import build
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);(root/'example.jpg').write_bytes(b'index fixture')
            metadata=root/'metadata.json'
            metadata.write_text(json.dumps([{'name':'example.jpg', 'caption':'an aircraft', 'categories':['civil']}]))
            index=root/'index.sqlite'
            build({'local_materials':{'demo':{'path':str(root),'metadata':{'kind':'platform_demo_manifest','sources':[str(metadata)]}}}}, index)
            with sqlite3.connect(index) as db:
                self.assertEqual(db.execute('select caption,primary_category from images').fetchone(),('an aircraft','civil'))
                self.assertEqual(db.execute('select count from category_counts').fetchone(),(1,))


if __name__=='__main__':
    unittest.main()
