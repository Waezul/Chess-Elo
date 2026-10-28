"""Eight-game player sample: one bin per player, eight games, no cross-split leak."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.player_sample import select_player_manifest
from src.player_table import average_games
from src.schema import BIN_LABELS, ENGINE_FEATURES, GAME_EXTRA, NUMERIC_FEATURES
from src.train_quantile import build_matrices, player_ship_decision


def _index_row(offset, game_id, white, black, white_elo, black_elo):
    return {
        "offset": offset,
        "game_id": game_id,
        "white_key": white,
        "black_key": black,
        "white_elo": white_elo,
        "black_elo": black_elo,
    }


class PlayerSampleTests(unittest.TestCase):
    def test_tail_bin_claims_a_player_who_also_has_middle_games(self):
        rows = []
        offset = 0
        for index in range(8):
            rows.append(_index_row(offset, f"high-{index}", "t", f"z{index}", 2300, 2300))
            offset += 1
        for index in range(8):
            rows.append(_index_row(offset, f"mid-{index}", "t", f"y{index}", 1500, 1500))
            offset += 1
        manifest, meta = select_player_manifest(
            pd.DataFrame(rows),
            players_per_bin=1,
            games_per_player=8,
            require_all_bins=False,
            seed=42,
        )
        self.assertEqual(set(manifest["elo_bin"]), {BIN_LABELS[7]})
        self.assertEqual(len(manifest), 8)
        self.assertEqual(meta["players"], 1)
        self.assertEqual(manifest["player_key"].nunique(), 1)

    def test_cross_split_game_is_dropped_and_same_split_is_kept(self):
        rows = []
        offset = 0
        for player, outsiders in (
            ("a", ("z1", "z2", "z3")),
            ("b", ("z4", "z5", "z6")),
            ("c", ("z7", "z8")),
            ("d", ("z9", "z10")),
        ):
            for outsider in outsiders:
                rows.append(_index_row(offset, f"{player}-{outsider}", player, outsider, 900, 900))
                offset += 1
        rows.append(_index_row(offset, "a-vs-b", "a", "b", 900, 900))
        index = pd.DataFrame(rows)
        split_map = {"a": "test", "b": "train", "c": "train", "d": "val"}
        manifest, _meta = select_player_manifest(
            index,
            players_per_bin=4,
            games_per_player=2,
            require_all_bins=False,
            split_map=split_map,
            seed=1,
        )
        shared = set(manifest.loc[manifest["player_key"] == "a", "game_id"]) & set(
            manifest.loc[manifest["player_key"] == "b", "game_id"]
        )
        self.assertEqual(shared, set())
        owners = set(manifest.loc[manifest["game_id"] == "a-vs-b", "player_key"])
        self.assertEqual(owners, {"a"})
        self.assertTrue((manifest.groupby("player_key").size() == 2).all())

        same_rows = [
            _index_row(0, "a-z", "a", "z1", 900, 900),
            _index_row(1, "a-vs-b", "a", "b", 900, 900),
            _index_row(2, "b-z", "b", "z2", 900, 900),
            _index_row(3, "c-z3", "c", "z3", 900, 900),
            _index_row(4, "c-z4", "c", "z4", 900, 900),
            _index_row(5, "d-z5", "d", "z5", 900, 900),
            _index_row(6, "d-z6", "d", "z6", 900, 900),
        ]
        kept, _meta = select_player_manifest(
            pd.DataFrame(same_rows),
            players_per_bin=4,
            games_per_player=2,
            require_all_bins=False,
            split_map={"a": "train", "b": "train", "c": "train", "d": "train"},
            seed=1,
        )
        both = kept.loc[kept["game_id"] == "a-vs-b", "player_key"]
        self.assertEqual(set(both), {"a", "b"})

    def test_average_uses_the_mean_elo_and_keeps_the_sampled_bin(self):
        games = pd.DataFrame(
            {
                "player_key": ["p", "p"],
                "elo": [1500, 1600],
                "split": ["train", "train"],
                "elo_bin": ["1400-1600", "1400-1600"],
                "is_white": [1.0, 0.0],
                "cpl_mean": [10.0, 30.0],
            }
        )
        averaged = average_games(games)
        self.assertEqual(len(averaged), 1)
        self.assertAlmostEqual(float(averaged.loc[0, "elo"]), 1550.0)
        self.assertEqual(averaged.loc[0, "elo_bin"], "1400-1600")
        self.assertAlmostEqual(float(averaged.loc[0, "cpl_mean"]), 20.0)
        self.assertAlmostEqual(float(averaged.loc[0, "is_white"]), 0.5)

    def test_player_ship_bar_requires_mae_bias_and_width(self):
        def row(label, bias, coverage, width):
            return {
                "bin": label,
                "n": 140,
                "mae": 100.0,
                "bias": bias,
                "coverage": coverage,
                "median_width": width,
            }

        bins = [row(label, 0.0, 0.8, 300.0) for label in BIN_LABELS]
        ready = player_ship_decision({"macro_mae": 110.0, "bins": bins})
        self.assertTrue(ready["ship_ready"])
        bins[0]["bias"] = 81.0
        biased = player_ship_decision({"macro_mae": 110.0, "bins": bins})
        self.assertFalse(biased["ship_ready"])
        self.assertEqual(biased["tail_bias_bins"], [BIN_LABELS[0]])
        bins[0]["bias"] = 0.0
        bins[3]["median_width"] = 351.0
        wide = player_ship_decision({"macro_mae": 110.0, "bins": bins})
        self.assertFalse(wide["ship_ready"])
        self.assertEqual(wide["wide_bins"], [BIN_LABELS[3]])

    def test_averaged_matrix_has_no_opening_column(self):
        columns = [*GAME_EXTRA, *NUMERIC_FEATURES, *ENGINE_FEATURES]
        frame = pd.DataFrame([{column: 1.0 for column in columns} for _ in range(3)])
        frame["split"] = ["train", "val", "test"]
        frame["elo"] = [1400.0, 1600.0, 1800.0]
        frame["elo_bin"] = ["1400-1600", "1600-1800", "1800-2000"]
        frame["player_key"] = ["a", "b", "c"]
        splits = {name: frame.loc[frame["split"] == name].reset_index(drop=True) for name in ("train", "val", "test")}
        built = build_matrices(splits, with_first_move=False)
        self.assertIsNone(built["encoder"])
        self.assertFalse(any(name.startswith("first_move_") for name in built["feature_names"]))
        self.assertEqual(built["matrices"]["train"].shape[1], len(columns))
        self.assertTrue(np.isfinite(built["matrices"]["val"].to_numpy()).all())


class MergeShardTests(unittest.TestCase):
    def test_merge_skips_a_truncated_shard(self):
        import tempfile
        from pathlib import Path

        from src.analyze_sample import OUTPUT_COLUMNS, _merge

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ckpt"
            ckpt.mkdir()
            good = pd.DataFrame([{column: None for column in OUTPUT_COLUMNS}])
            good["game_id"] = "g"
            good["color"] = "white"
            good["player_key"] = "p"
            good["split"] = "train"
            good["elo"] = 1500
            good.to_parquet(ckpt / "w00_b00000.parquet", index=False)
            (ckpt / "w00_b00001.parquet").write_bytes(b"xxxx")
            frame = _merge(ckpt, Path(tmp) / "out.parquet")
            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.loc[0, "game_id"], "g")
            self.assertFalse((ckpt / "w00_b00001.parquet").exists())


if __name__ == "__main__":
    unittest.main()
