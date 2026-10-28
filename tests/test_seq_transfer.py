"""Sequence-transfer safety tests using synthetic rows only."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import seq_models as seq
from src import seq_residual_eval as experiment
from src.residuals import contrast_row, subject_table


def toy_frame():
    return pd.DataFrame({'evaluation_protocol': ['synthetic'] * 4, 'subject': [1, 1, 2, 2],
                         'origin_time': pd.date_range('2020-01-01', periods=4, freq='h'),
                         'task_id': ['t'] * 4, 'split': ['train'] * 4, 'target_value': [103., 106., 108., 112.],
                         'row_id': np.arange(4)})


class TransferTests(unittest.TestCase):
    def test_verified_shard_returns_the_single_loaded_prediction_frame(self):
        frame = toy_frame()
        with tempfile.TemporaryDirectory() as directory, patch.object(experiment, 'OUT', Path(directory)):
            root = Path(directory)
            predictions = frame.assign(prediction=100.)
            predictions.to_parquet(root / 'pred.parquet', index=False)
            experiment.save_json(root / 'meta.json', {'status': 'complete', 'config_sha256': 'synthetic',
                'output_sha256': {'pred.parquet': experiment.sha(root / 'pred.parquet')},
                'fit_keys_sha256': experiment.key_digest(frame), 'prediction_keys_sha256': experiment.key_digest(frame), 'prediction_file': 'pred.parquet'})
            with patch.object(experiment.pd, 'read_parquet', wraps=pd.read_parquet) as read:
                meta, loaded = experiment.verify_shard(root / 'meta.json', 'synthetic', frame, frame)
                self.assertEqual(read.call_count, 1)
                pd.testing.assert_frame_equal(loaded, predictions, check_exact=True)

    def test_holdout_targets_do_not_change_fixed_epoch_weights(self):
        frame = toy_frame()
        h = np.arange(100., 200.).reshape(4, 25)
        first, _, state1 = seq.train_one('GRU', frame, frame.assign(split='validation'), h, h,
                                       epochs=2, early_stopping=False, device='cpu')
        second, _, state2 = seq.train_one('GRU', frame, frame.assign(split='validation', target_value=-99999.), h, h,
                                        epochs=2, early_stopping=False, device='cpu')
        self.assertEqual(first['statistics'], second['statistics'])
        self.assertTrue(all(torch.equal(state1[k], state2[k]) for k in state1))
        self.assertFalse(first['early_stopping'])
        self.assertEqual(first['epochs_run'], 2)

    def test_overlapping_support_rejected(self):
        held = pd.DataFrame({'subject': [1], 'history_start_time': [pd.Timestamp('2020-01-01 03:00')],
                             'target_time': [pd.Timestamp('2020-01-01 05:30')]})
        fit = held.assign(history_start_time=pd.Timestamp('2020-01-01 05:00'), target_time=pd.Timestamp('2020-01-01 07:30'))
        with self.assertRaises(ValueError):
            experiment.assert_no_support_overlap(fit, held)
        experiment.assert_no_support_overlap(fit.assign(history_start_time=pd.Timestamp('2020-01-01 06:00')), held)

    def test_empty_factor_block_and_delta_sign(self):
        frame = toy_frame().assign(base_prediction=[101., 104., 106., 110.], residual=2.)
        alpha, table, pred = experiment.fit_factors(frame, frame.assign(split='validation'), [])
        self.assertIsNone(alpha)
        self.assertTrue(np.array_equal(pred, frame.base_prediction))
        delta = contrast_row(table, table, n_resamples=2000, seed=42)
        self.assertEqual((delta['d_rmse'], delta['d_rmse_low'], delta['d_rmse_high']), (0., 0., 0.))
        better = subject_table(frame.subject, np.zeros(4))
        self.assertLess(contrast_row(table, better, n_resamples=2000, seed=42)['d_rmse'], 0)

    def test_atomic_completion_and_hash_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'record.json'
            experiment.save_json(path, {'status': 'complete'})
            self.assertEqual(experiment.read_json(path)['status'], 'complete')
            old = experiment.sha(path)
            experiment.save_json(path, {'status': 'changed'})
            self.assertNotEqual(experiment.sha(path), old)
            self.assertFalse(list(Path(directory).glob('*.tmp.*')))

    def test_complete_residual_evaluation_on_common_synthetic_rows(self):
        n = 58
        frame = pd.DataFrame({'evaluation_protocol': [seq.PROTOCOLS['A']] * n,
            'subject': [1, 2] * (n // 2), 'origin_time': pd.date_range('2020-01-01', periods=n, freq='h'),
            'task_id': ['t'] * n, 'split': ['train'] * 16 + ['validation'] * 42,
            'target_value': 110 + np.sin(np.arange(n)), 'base_prediction': np.full(n, 110.),
            'row_id': np.arange(n), 'cohort_core': True, 'x1': np.cos(np.arange(n)), 'x2': np.arange(n) / n})
        frame['residual'] = frame.target_value - frame.base_prediction
        ensemble = frame[['row_id']].assign(prediction=109.9, residual=frame.target_value - 109.9)
        seeds = pd.concat([frame[['row_id', 'split']].assign(prediction=109.9 + (s - 2) / 100, seed=s) for s in range(5)])
        config = {'factor_columns': {'A': {'M': ['x1'], 'T': ['x2'], 'O': ['x1', 'x2']}},
                  'min_validation_windows_per_subject': 20, 'seeds': list(range(5))}
        with tempfile.TemporaryDirectory() as directory, patch.object(experiment, 'OUT', Path(directory)):
            result = experiment.evaluate_protocol('GRU', 'A', frame, ensemble, seeds, config)
        self.assertEqual(len(result['metrics']), 5)
        self.assertEqual(len(result['contrasts']), 4)
        self.assertTrue(all(r['windows'] == 42 and r['subjects'] == 2 for r in result['metrics']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
