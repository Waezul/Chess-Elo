"""Per-game v3 features from a move table."""

from __future__ import annotations

import math
import unittest

import pandas as pd

from src.engine import apply_engine_features
from src.features import rows_from_movetext
from src.game_features import extract_game, move_accuracy, side_features, win_percent
from src.moves import MOVE_COLUMNS
from src.schema import ENGINE_FEATURES, V3_CLOCK


def frame(rows):
    return pd.DataFrame(rows, columns=MOVE_COLUMNS)


FOUR = frame([
    ["g", 1, "white", "e2e4", "e2e4", 0, -100, None],
    ["g", 2, "black", "e7e5", "e7e5", -100, -100, None],
    ["g", 3, "white", "g1f3", "d2d4", -100, -100, None],
    ["g", 4, "black", "b8c6", "g8f6", -100, 300, None],
])


class FormulaTests(unittest.TestCase):
    def test_win_percent_anchors(self):
        self.assertAlmostEqual(win_percent(0), 50.0)
        self.assertAlmostEqual(win_percent(1000), 97.54, places=2)
        self.assertAlmostEqual(win_percent(-1000), 2.46, places=2)

    def test_accuracy_is_clamped(self):
        self.assertAlmostEqual(move_accuracy(0), 99.9999, places=3)
        self.assertEqual(move_accuracy(100), 0.0)


class SideFeatureTests(unittest.TestCase):
    def test_subject_and_opponent_losses(self):
        f = side_features(FOUR, "white", "600+5")
        self.assertAlmostEqual(f["cpl_balanced"], 50.0)
        self.assertAlmostEqual(f["opp_cpl_mean"], 200.0)
        self.assertAlmostEqual(f["opp_blunder_rate"], 0.5)
        self.assertAlmostEqual(f["opp_cpl_balanced"], 200.0)
        self.assertAlmostEqual(f["cpl_opening"], 50.0)
        self.assertTrue(math.isnan(f["cpl_late"]))
        self.assertEqual((f["tc_base"], f["tc_increment"]), (600.0, 5.0))

    def test_no_clocks_gives_nan_clock_block(self):
        f = side_features(FOUR, "white", "600+5")
        self.assertEqual(f["has_clock"], 0.0)
        for key in V3_CLOCK[1:]:
            self.assertTrue(math.isnan(f[key]), key)

    def test_unparseable_time_control_turns_clocks_off(self):
        clocked = FOUR.assign(clock_after=[600.0, 600.0, 590.0, 598.0])
        self.assertEqual(side_features(clocked, "white", "-")["has_clock"], 0.0)

    def test_clock_features(self):
        clocked = FOUR.assign(clock_after=[600.0, 600.0, 590.0, 598.0])
        f = side_features(clocked, "white", "600+5")
        self.assertEqual(f["has_clock"], 1.0)
        self.assertAlmostEqual(f["clk_think_frac_mean"], 15.0 / 800.0)
        self.assertAlmostEqual(f["clk_end_frac"], 590.0 / 800.0)
        self.assertAlmostEqual(f["clk_low_rate"], 0.0)

    def test_mate_clamp_keeps_accuracy_in_range(self):
        mates = FOUR.assign(cp_before=[1000, -1000, 1000, -1000], cp_after=[-1000, 1000, -1000, 1000])
        f = side_features(mates, "white", "600+5")
        self.assertTrue(0.0 <= f["acc_mean"] <= 100.0)
        self.assertTrue(0.0 <= f["opp_acc_mean"] <= 100.0)


class ExtractParityTests(unittest.TestCase):
    def test_v2_engine_columns_match_apply_engine_features(self):
        headers = {"White": "A", "Black": "B", "Result": "1-0", "TimeControl": "600+5"}
        lines = ["1. e4 { [%clk 0:10:00] } 1... e5 { [%clk 0:10:00] } 2. Nf3 { [%clk 0:09:50] } 2... Nc6 { [%clk 0:09:58] } 1-0"]
        analyse = lambda board: ([10, 20, -30, -30, 40][board.ply()], None)
        rows, moves = extract_game(headers, lines, analyse)
        legacy = apply_engine_features(headers, lines, rows_from_movetext(headers, lines), analyse)
        for new, old in zip(sorted(rows, key=lambda r: r["color"]), sorted(legacy, key=lambda r: r["color"])):
            for key in ENGINE_FEATURES:
                self.assertAlmostEqual(new[key], old[key], msg=key)
        self.assertEqual(len(moves), 4)
        self.assertEqual(rows[0]["time_control"], "600+5")
