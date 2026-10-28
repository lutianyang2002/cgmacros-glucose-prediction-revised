"""Synthetic-only tests for Experiment C. No real source files are loaded."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.factor_construction import conv_gamma, factor
from src.lab_interactions import KEY, MEALS, STATES, check_bmi, isolate_keyed, lab_interaction_columns, lab_state_parameters
from src.lab_experiment import Fitter, proxy_frames


def sample():
    n = 6
    frame = pd.DataFrame({"subject": [1, 1, 2, 2, 3, 3], "origin_time": pd.date_range("2020-01-01", periods=n, freq="5min"), "task_id": "synthetic",
                          "split": ["train"] * 4 + ["validation"] * 2, "cohort_core": True, "residual": [1., 3., -2., -4., 0., 0.],
                          "bmi_kg_m2": [20., 20., 30., 30., 40., 40.]})
    for j, col in enumerate(STATES.values(), 1):
        frame[col] = np.repeat([float(j), float(j + 2), float(j + 20)], 2)
    for i, meal in enumerate(MEALS, 1):
        frame[meal] = np.arange(n, dtype=float) + i
        frame[f"int_{meal}_carbs_x_bmi"] = frame[meal] * frame.bmi_kg_m2
    return frame


class LabTests(unittest.TestCase):
    def test_nonfinite_fit_is_rejected_before_checkpoint_publication(self):
        import tempfile
        from unittest.mock import patch
        from src import lab_experiment as experiment
        with tempfile.TemporaryDirectory() as directory, patch.object(experiment, "OUT", Path(directory)):
            frame = sample().assign(base_prediction=100., target_value=101.)
            fit = Fitter(frame.iloc[:4], frame.iloc[4:], "synthetic", {}, 1e100)
            with patch.object(experiment, "ridge_path_predictions", return_value=np.full((2, 13), np.nan)), patch.object(experiment, "best_alpha", return_value=(0, 1., None)):
                with self.assertRaises(ValueError):
                    fit.fit(["breakfast"])
            self.assertEqual(list(fit.root.iterdir()), [])

    def test_mixed_payload_filtered_before_use(self):
        f = sample()
        keys = f.iloc[:4][KEY + ["split"]]
        raw = f[KEY].assign(value=np.arange(len(f), dtype=float))
        raw.loc[4:, "value"] = 1e200  # synthetic forbidden-scope sentinel
        isolated = isolate_keyed(raw, keys, ["value"])
        np.testing.assert_array_equal(isolated.value, [0., 1., 2., 3.])
        isolated.loc[0, "value"] = -20
        self.assertEqual(raw.loc[0, "value"], 0.)

    def test_conflicting_split_not_hidden(self):
        f = sample()
        raw = f[KEY + ["split"]].assign(value=0.)
        raw.loc[0, "split"] = "test"
        with self.assertRaisesRegex(ValueError, "conflicts"):
            isolate_keyed(raw, f[KEY + ["split"]], ["value"], check_split=True)

    def test_model_rejects_unisolated_test_before_fit(self):
        f = sample()
        f.loc[0, "split"] = "test"
        with self.assertRaises(ValueError):
            Fitter(f, sample().iloc[4:], "must_not_create_checkpoint", {}, 1e100)
        with self.assertRaises(ValueError):
            lab_state_parameters(f)

    def test_unique_training_subject_standardization(self):
        f = sample()
        p = lab_state_parameters(f)
        self.assertEqual(p["training_subjects"], 2)
        self.assertEqual(p["mean"]["fasting_glucose_mg_dl"], 2.)
        self.assertEqual(p["std_ddof0"]["fasting_glucose_mg_dl"], 1.)
        f.loc[f.split.eq("validation"), list(STATES.values())] = 1e9
        f = pd.concat([f, f.iloc[[0]]], ignore_index=True)
        self.assertEqual(lab_state_parameters(f), p)

    def test_twelve_interactions_and_scaling_order(self):
        f = sample()
        p = lab_state_parameters(f)
        km = dict(zip(MEALS, MEALS))
        a = lab_interaction_columns(f, km, p, "C-S1")
        b = lab_interaction_columns(f, km, p, "C-S2")
        self.assertEqual(a.shape, (6, 12))
        np.testing.assert_array_equal(a.lab_breakfast_carbs_x_fasting_glucose, f.breakfast * (f.fasting_glucose_mg_dl - 2))
        np.testing.assert_array_equal(b.lab_breakfast_carbs_x_fasting_glucose, f.breakfast * f.fasting_glucose_mg_dl)
        np.testing.assert_allclose(b.lab_breakfast_carbs_x_fasting_glucose, a.lab_breakfast_carbs_x_fasting_glucose + 2 * f.breakfast)

    def test_bmi_old_multiplication_and_block_scaling(self):
        f = sample()
        checks = check_bmi(f, dict(zip(MEALS, MEALS)))
        self.assertEqual(len(checks), 4)
        self.assertTrue(all(c["passed"] and c["raw_max_abs_diff"] == 0 for c in checks))

    def test_proxy_uses_training_residuals(self):
        f = sample()
        tr, va = proxy_frames(f.iloc[:4], f.iloc[:2].assign(split="validation", residual=1e20), True)
        np.testing.assert_array_equal(tr.identity_proxy, [2., 2., -3., -3.])
        np.testing.assert_array_equal(va.identity_proxy, [2., 2.])

    def test_temporal_causality_of_reused_kernel(self):
        x = np.zeros(200)
        x[20] = 40
        y = x.copy()
        y[100:] = 1e8
        np.testing.assert_array_equal(conv_gamma(x, 20, 2).values[:100], conv_gamma(y, 20, 2).values[:100])
        with self.assertRaises(ValueError):
            factor("bad", "future", "test", -120, 1, np.ones(3))

    def test_mapping_preserves_fixed_key_order(self):
        f = sample()
        raw = f[KEY].assign(value=f.subject * 10).sample(frac=1, random_state=1)
        got = isolate_keyed(raw, f[KEY + ["split"]], ["value"])
        np.testing.assert_array_equal(got.value, f.subject * 10)
        with self.assertRaises(ValueError):
            isolate_keyed(pd.concat([raw, raw.iloc[[0]]]), f[KEY + ["split"]], ["value"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
