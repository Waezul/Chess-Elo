"""Player rows: one per subset, spread, and seeded clock dropout."""

from __future__ import annotations

import unittest

import pandas as pd

from src.player_table import average_games_v3, drop_clocks, strip_clocks
from src.schema import V3_CLOCK, V3_GAME_FEATURES


def _games(rows: list[tuple]) -> pd.DataFrame:
    records = []
    for player, subset, acc in rows:
        record = {column: 0.0 for column in V3_GAME_FEATURES}
        record.update(
            {
                "player_key": player,
                "subset": subset,
                "elo": 1500.0,
                "split": "train",
                "elo_bin": "1400-1600",
                "acc_mean": float(acc),
                "has_clock": 1.0,
                "clk_end_frac": 0.4,
            }
        )
        records.append(record)
    return pd.DataFrame(records)


GAMES_TWO_SUBSETS = _games([("p", 0, 70), ("p", 1, 80)])
GAMES_ACC_60_AND_80 = _games([("p", 0, 60), ("p", 0, 80)])
MANY_PLAYERS = _games([(f"p{index}", 0, 50) for index in range(10)])


class V3TableTests(unittest.TestCase):
    def test_one_row_per_subset_with_weights_summing_to_one(self):
        players = average_games_v3(GAMES_TWO_SUBSETS)
        self.assertEqual(len(players), 2)
        self.assertAlmostEqual(players.groupby("player_key")["weight"].sum().iloc[0], 1.0)

    def test_spread_columns(self):
        players = average_games_v3(GAMES_ACC_60_AND_80)
        self.assertAlmostEqual(players.loc[0, "acc_mean_std"], 10.0)
        self.assertAlmostEqual(players.loc[0, "acc_mean_min"], 60.0)
        self.assertAlmostEqual(players.loc[0, "acc_mean_max"], 80.0)

    def test_drop_clocks_is_seeded_and_leaves_input_alone(self):
        players = average_games_v3(MANY_PLAYERS)
        a = drop_clocks(players, 0.2, seed=42)
        b = drop_clocks(players, 0.2, seed=42)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(int((a["has_clock"] == 0).sum()), round(0.2 * len(players)))
        self.assertTrue(players["clk_end_frac"].notna().all())

    def test_strip_clocks_blanks_every_clock_feature(self):
        stripped = strip_clocks(average_games_v3(MANY_PLAYERS))
        self.assertTrue(stripped[V3_CLOCK[1:]].isna().all().all())
        self.assertTrue((stripped["has_clock"] == 0).all())
