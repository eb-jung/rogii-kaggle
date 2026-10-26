import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from rogii.features import aligned_submission, build_features
from rogii.experiment import audit_data, fit_ensemble, predict_ensemble, split_wells, smooth_predictions


def fixture(wid="well"):
    h = pd.DataFrame({"MD": np.arange(20, dtype=float), "X": np.arange(20),
                      "Y": np.ones(20), "Z": np.arange(20) * .1,
                      "GR": np.sin(np.arange(20)) * 3 + 50,
                      "TVT_input": [100., 100.1, 100.2, 100.3, 100.4] + [np.nan]*15,
                      "TVT": 100 + np.arange(20)*.1, "ANCC": np.arange(20)})
    tw = pd.DataFrame({"TVT": np.arange(90., 120.), "GR": np.arange(35., 65.),
                       "Geology": ["unused"]*30})
    return h, tw


class IntegrityTests(unittest.TestCase):
    def test_official_and_duplicate_wells_are_excluded_before_splitting(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            (data / "train").mkdir()
            (data / "test").mkdir()
            h, tw = fixture()
            h.to_csv(data / "train/official__horizontal_well.csv", index=False)
            h.to_csv(data / "train/renamed_official__horizontal_well.csv", index=False)
            h.drop(columns=["TVT", "ANCC"]).to_csv(data / "test/official__horizontal_well.csv", index=False)
            h.X += 100
            h.to_csv(data / "train/a__horizontal_well.csv", index=False)
            h.to_csv(data / "train/b__horizontal_well.csv", index=False)
            pd.DataFrame({"id": ["official_5"], "tvt": [0]}).to_csv(data / "sample_submission.csv", index=False)
            result = audit_data(data)
            self.assertEqual(result["eligible_wells"], ["a"])
            self.assertEqual(result["duplicate_input_wells"], ["b"])
            self.assertEqual(result["excluded_overlap_wells"], ["official", "renamed_official"])

    def test_hidden_targets_and_formation_labels_cannot_change_features(self):
        h, tw = fixture()
        x, ids = build_features(h, tw, "a")
        h["TVT"] = -999999
        h["ANCC"] = np.random.default_rng(3).normal(size=len(h))
        tw["Geology"] = "changed"
        xx, ids2 = build_features(h, tw, "a")
        assert_frame_equal(x, xx)
        self.assertEqual(ids, ids2)
        self.assertFalse({"TVT", "TVT_input", "ANCC"} & set(x.columns))

    def test_target_columns_can_be_absent(self):
        h, tw = fixture()
        x, ids = build_features(h.drop(columns=["TVT", "ANCC"]), tw, "a")
        self.assertEqual(ids, [f"a_{i}" for i in range(5, 20)])
        self.assertEqual(len(x), 15)

    def test_legitimate_context_is_retained(self):
        h, tw = fixture()
        x, _ = build_features(h, tw, "a")
        h.loc[:4, "TVT_input"] += 10
        xx, _ = build_features(h, tw, "a")
        np.testing.assert_allclose(xx.last_known_tvt - x.last_known_tvt, 10)

    def test_nonprefix_context_fails(self):
        h, tw = fixture()
        h.loc[10, "TVT_input"] = 300
        with self.assertRaisesRegex(ValueError, "contiguous"):
            build_features(h, tw, "a")

    def test_duplicate_depth_fails(self):
        h, tw = fixture()
        h.loc[9, "MD"] = h.loc[8, "MD"]
        with self.assertRaisesRegex(ValueError, "strictly"):
            build_features(h, tw, "a")

    def test_submission_alignment(self):
        sample = pd.DataFrame({"id": ["a_8", "a_5"], "tvt": [0, 0]})
        out = aligned_submission(sample, ["a_5", "a_8"], [12., 15.])
        self.assertEqual(out.tvt.tolist(), [15., 12.])
        for ids, values in [(["a_5"], [1.]), (["a_5", "a_5"], [1., 2.]),
                             (["a_5", "a_8"], [1., np.nan])]:
            with self.assertRaises(ValueError):
                aligned_submission(sample, ids, values)

    def test_well_split_is_disjoint_and_order_independent(self):
        ids = [f"w{i}" for i in range(100)]
        a = split_wells(ids)
        self.assertEqual(a, split_wells(ids[::-1]))
        self.assertEqual(len(a["train"]), 80)
        self.assertFalse(set(a["test"]) & set(a["validation"]))
        self.assertFalse(set(a["train"]) & (set(a["validation"]) | set(a["test"])))

    def test_group_folds_and_prediction_reproducibility(self):
        h, tw = fixture()
        x, _ = build_features(h, tw, "a")
        xx = pd.concat([x]*6, ignore_index=True)
        y = np.tile(h.TVT.iloc[5:].to_numpy(), 6)
        groups = np.repeat([f"w{i}" for i in range(6)], len(x))
        m1, folds, oof = fit_ensemble(xx, y, groups, rounds=5)
        m2, _, oof2 = fit_ensemble(xx, y, groups, rounds=5)
        for f in folds:
            self.assertFalse(set(f["train_wells"]) & set(f["validation_wells"]))
        np.testing.assert_array_equal(oof, oof2)
        np.testing.assert_array_equal(predict_ensemble(m1, xx), predict_ensemble(m2, xx))

    def test_smoothing_does_not_cross_wells(self):
        a = np.arange(15, dtype=float)
        b = np.arange(15, dtype=float) + 999
        pred = smooth_predictions(np.r_[a, b], np.array(["a"]*15 + ["b"]*15))
        np.testing.assert_allclose(pred, np.r_[a, b], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
