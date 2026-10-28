"""v3 training helpers: per-bin scales, the round cap, and NaN clocks."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.metrics import apply_interval_by_bin, calibrate_scales_by_bin
from src.schema import BIN_LABELS, V3_PLAYER_FEATURES
from src.train_quantile import FROZEN_PARAMS, SEARCH_GRID, build_matrices, cap_binding


def _splits_with_nan_clocks():
    frame = pd.DataFrame({column: np.ones(6) for column in V3_PLAYER_FEATURES})
    frame["clk_end_frac"] = [np.nan, 0.2, np.nan, 0.4, 0.1, np.nan]
    frame["elo"] = 1500.0
    return {
        "train": frame.iloc[:2].reset_index(drop=True),
        "val": frame.iloc[2:4].reset_index(drop=True),
        "test": frame.iloc[4:].reset_index(drop=True),
    }


class V3TrainTests(unittest.TestCase):
    def test_scales_cover_every_bin_and_fall_back_to_all(self):
        y = np.full(60, 1500.0)
        q10 = np.full(60, 1400.0)
        q50 = np.full(60, 1500.0)
        q90 = np.full(60, 1600.0)
        scales = calibrate_scales_by_bin(y, q10, q50, q90, min_n=50)
        self.assertEqual(set(scales), {*BIN_LABELS, "all"})
        self.assertEqual(scales["800-1000"], scales["all"])

    def test_apply_by_bin_uses_the_bin_of_the_median(self):
        lo, _mid, hi = apply_interval_by_bin(
            np.array([900.0]),
            np.array([1000.0]),
            np.array([1100.0]),
            {**{label: 1.0 for label in BIN_LABELS}, "800-1000": 2.0, "all": 1.0},
        )
        self.assertEqual((lo[0], hi[0]), (800.0, 1200.0))

    def test_cap_binding_flags_a_capped_fit(self):
        capped = SimpleNamespace(best_iteration=399)
        free = SimpleNamespace(best_iteration=120)
        self.assertEqual(cap_binding({"q10": free, "q50": capped, "q90": free}, 400), ["q50"])

    def test_v2_search_defaults_are_unchanged(self):
        self.assertEqual([row["name"] for row in SEARCH_GRID][0], "frozen")
        self.assertEqual(FROZEN_PARAMS["n_estimators"], 400)

    def test_matrices_keep_nan_for_v3_columns(self):
        built = build_matrices(_splits_with_nan_clocks(), with_first_move=False, numeric=V3_PLAYER_FEATURES)
        self.assertTrue(built["matrices"]["train"]["clk_end_frac"].isna().any())
        self.assertEqual(built["feature_names"], V3_PLAYER_FEATURES)
