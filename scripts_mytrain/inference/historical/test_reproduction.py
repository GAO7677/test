"""Packaging parity: preserve original numerical code and separate protocols."""
import ast
from pathlib import Path
import unittest
import tempfile
from argparse import Namespace
import reproduce as r

HERE = Path(__file__).resolve().parent


def node(path, name):
    tree = ast.parse(Path(path).read_text())
    return ast.dump(next(n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name), include_attributes=False)


class HistoricalParity(unittest.TestCase):
    def test_wan_sampler_and_noise_are_unchanged(self):
        for arm in ['clean', 'noisy']:
            for name in ['Experiment', 'sample', 'generate']:
                self.assertEqual(node(HERE / f'wan_{arm}.py', name), node(HERE / 'frozen/wan/worker.py', name))
            self.assertEqual(node(HERE / f'wan_{arm}.py', 'ContextNoise'), node(HERE / 'frozen/wan/noise_adapter.py', 'ContextNoise'))

    def test_nano_sampling_and_default_hook_are_unchanged(self):
        for arm in ['clean', 'noisy']:
            self.assertEqual(node(HERE / f'nano_{arm}.py', 'main'), node(HERE / 'frozen/nano/n75_sweep/nano.py', 'main'))
            for name in ['NanoContext', 'CleanContextAudit']:
                self.assertEqual(node(HERE / f'nano_{arm}.py', name), node(HERE / 'frozen/nano/n75_sweep/hooks.py', name))

    def test_predict_sampling_and_hook_are_unchanged(self):
        for arm in ['clean', 'noisy']:
            for name in ['plan', 'main']:
                self.assertEqual(node(HERE / f'predict25_{arm}.py', name), node(HERE / 'frozen/predict25/infer_bench.py', name))
            self.assertEqual(node(HERE / f'predict25_{arm}.py', 'install_predict'), node(HERE / 'frozen/predict25/hooks.py', 'install_predict'))

    def test_every_frozen_file_has_original_hash(self):
        for row in r.read(HERE / 'source_manifest.json'):
            self.assertEqual(r.sha(HERE / row['file']), row['sha256'], row['file'])

    def test_six_default_references_are_valid(self):
        for backbone in ['wan', 'nano', 'predict25']:
            for arm in ['clean', 'noisy']:
                with self.subTest(backbone=backbone, arm=arm):
                    p = r.plan(backbone, arm, Namespace(reference=None, output=None, config=None, gpu=2))
                    self.assertEqual(p['case'], 'v2v_gap_038')

    def test_clean_entry_rejects_noisy_reference(self):
        for backbone in ['wan', 'nano', 'predict25']:
            args = Namespace(reference=r.default_reference(backbone, 'noisy'), output=None, config=None, gpu=2)
            with self.assertRaisesRegex(ValueError, 'Clean/noisy'):
                r.plan(backbone, 'clean', args)

    def test_nano_old_and_new_input_are_distinct(self):
        nano = r.plan('nano', 'clean', Namespace(reference=None, output=None, config=None, gpu=2))
        old = r.read(HERE / 'frozen/predict25/manifest.json')['cases']['v2v_gap_038']
        self.assertNotEqual(nano['row']['context_video_sha256'], old['context_video_sha256'])

    def test_equal_video_cannot_hide_different_model_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            old = Path(directory) / 'old/results/case/result.json'
            new = Path(directory) / 'new/results/case/result.json'
            r.write(old, dict(video_sha256='same-video', actual_forward=[{'t': 999}]))
            r.write(new, dict(video_sha256='same-video', actual_forward=[{'t': 750}]))
            report = r.compare(old, new)
            self.assertTrue(report['video_bytes_equal'])
            self.assertFalse(report['actual_forward_equal'])
            self.assertEqual(report['status'], 'DIFFERENT_REQUIRES_REVIEW')


if __name__ == '__main__':
    unittest.main()
