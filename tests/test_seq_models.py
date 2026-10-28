"""Synthetic tests for the shared sequence models used by supplementary studies and sequence-model benchmark."""
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from src import seq_models as seq


class SequenceTests(unittest.TestCase):
    def test_notebook_imports_shared_models(self):
        notebook = json.loads((ROOT / 'notebooks/07_optional_studies.ipynb').read_text(encoding='utf-8'))
        nodes = [node for cell in notebook['cells'] if cell['cell_type'] == 'code'
                 for node in ast.walk(ast.parse(''.join(cell['source'])))]
        names = {'CompactGRU', 'CausalBlock', 'CompactTCN'}
        self.assertFalse(any(isinstance(node, ast.ClassDef) and node.name in names for node in nodes))
        imported = {alias.name for node in nodes if isinstance(node, ast.ImportFrom) and node.module == 'src.seq_models'
                    for alias in node.names}
        self.assertTrue({'CompactGRU', 'CompactTCN'} <= imported)
        source = ast.parse((ROOT / 'src/seq_models.py').read_text(encoding='utf-8'))
        self.assertCountEqual([node.name for node in source.body if isinstance(node, ast.ClassDef)], names)

    def test_parameter_counts(self):
        for family, count in [('GRU', 3393), ('TCN', 2465)]:
            model = seq.make_model(family).eval()
            self.assertEqual(sum(p.numel() for p in model.parameters()), count)
            with torch.no_grad():
                prediction = model(torch.zeros(2, 25, 1))
            self.assertEqual(prediction.shape, (2,))
            self.assertTrue(torch.isfinite(prediction).all())

    def test_tcn_causality(self):
        torch.manual_seed(0)
        model = seq.make_model('TCN').eval()
        x = torch.randn(4, 1, 25)
        changed = x.clone()
        changed[:, :, -1] += 10
        with torch.no_grad():
            self.assertTrue(torch.equal(model.blocks(x)[:, :, :-1], model.blocks(changed)[:, :, :-1]))

    def test_ppgr_interpreter_selection(self):
        """Isolated interpreter gate; filesystem and subprocess are mocked."""
        notebook = json.loads((ROOT / 'notebooks/07_optional_studies.ipynb').read_text(encoding='utf-8'))
        source = next(''.join(c['source']) for c in notebook['cells'] if c.get('id') == '5e1353d5')
        code = compile(source, 'ppgr_interpreter_gate', 'exec')
        root = Path('synthetic')
        local, explicit = root / '.venv-ppgr/Scripts/python.exe', root / 'explicit/python.exe'
        cases = [({}, {local}, '1.6.2', None),
                 ({'CGMACROS_PPGR_PYTHON': str(explicit)}, {local, explicit}, '1.6.2', None),
                 ({'CGMACROS_PPGR_PYTHON': str(explicit)}, {local}, '1.6.2', FileNotFoundError),
                 ({}, set(), '1.6.2', FileNotFoundError), ({}, {local}, '2.0.0', ValueError)]
        for environ, available, version, error in cases:
            run = Mock(return_value=SimpleNamespace(stdout=json.dumps({'xgboost': version})))
            namespace = dict(Path=Path, os=SimpleNamespace(environ=environ), PROJECT_ROOT=root,
                             subprocess=SimpleNamespace(run=run), json=json, XGB_VERSION='1.6.2')
            with self.subTest(environ=environ, version=version, error=error), patch.object(Path, 'is_file', lambda p: p in available):
                if error:
                    with self.assertRaises(error):
                        exec(code, namespace)
                    self.assertEqual(run.call_count, 0 if error is FileNotFoundError else 1)
                else:
                    exec(code, namespace)
                    self.assertEqual(namespace['PPGR_PYTHON'], explicit if environ else local)
                    self.assertEqual(run.call_args.args[0][:2], [str(namespace['PPGR_PYTHON']), '-c'])

    def test_training_only_statistics_and_access(self):
        frame = pd.DataFrame({'evaluation_protocol': ['synthetic'] * 2, 'subject': [1, 1],
                              'origin_time': [0, 1], 'task_id': ['t', 't'], 'split': ['train'] * 2,
                              'target_value': [11., 25.]})
        history = np.array([[1., 2.], [4., 8.]])
        stats = seq.training_statistics(frame, history)
        self.assertEqual(stats['history_mean'], history.mean())
        self.assertEqual(stats['residual_sd'], np.std([9., 17.]))
        with self.assertRaises(ValueError):
            seq.training_statistics(frame.assign(split='validation'), history)
        with self.assertRaises(ValueError):
            seq.require_development(frame.assign(split='test'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
