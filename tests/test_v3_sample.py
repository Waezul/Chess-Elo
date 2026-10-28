"""v3 manifest: same held-out games as v2, disjoint train subsets, no held-out opponents."""

from __future__ import annotations

import unittest

import pandas as pd

from src.player_sample_v3 import select_v3_manifest
from src.sample import assert_split_integrity
from src.schema import BIN_LABELS
from tests.test_player8 import _index_row

BIN = BIN_LABELS[0]


def _side(offset, game_id, player, split):
    return {
        "offset": offset,
        "game_id": game_id,
        "color": "white",
        "player_key": player,
        "elo": 900,
        "elo_bin": BIN,
        "bin_index": 0,
        "split": split,
        "thin_bin": False,
    }


V2 = pd.DataFrame(
    [
        _side(0, "gt", "t", "train"),
        _side(1, "gv", "v", "val"),
        _side(2, "ge", "e", "test"),
    ]
)

INDEX = pd.DataFrame(
    [_index_row(100 + i, f"safe{i}", "t", f"o{i}", 900, 900) for i in range(6)]
    + [
        _index_row(200, "vs-v", "t", "v", 900, 900),
        _index_row(201, "vs-e", "t", "e", 900, 910),
    ]
)
GAMES_VS_HELD_OUT = {"vs-v", "vs-e"}
INDEX_ONLY_HELD_OUT_OPPONENTS = pd.DataFrame(
    [
        _index_row(200, "vs-v", "t", "v", 900, 900),
        _index_row(201, "vs-e", "t", "e", 900, 910),
    ]
)


class V3SampleTests(unittest.TestCase):
    def test_held_out_rows_are_copied_from_v2(self):
        manifest, _ = select_v3_manifest(INDEX, V2, train_subsets=3, games_per_player=2)
        held = manifest[manifest["split"] != "train"].drop(columns="subset")
        expected = V2[V2["split"] != "train"]
        pd.testing.assert_frame_equal(
            held.sort_values(["game_id", "color"]).reset_index(drop=True),
            expected.sort_values(["game_id", "color"]).reset_index(drop=True)[held.columns],
        )

    def test_train_never_plays_a_held_out_player(self):
        manifest, _ = select_v3_manifest(INDEX, V2, train_subsets=3, games_per_player=2)
        train_games = set(manifest.loc[manifest["split"] == "train", "game_id"])
        self.assertEqual(train_games & GAMES_VS_HELD_OUT, set())

    def test_subsets_are_disjoint_full_and_capped(self):
        manifest, meta = select_v3_manifest(INDEX, V2, train_subsets=3, games_per_player=2)
        train = manifest[manifest["split"] == "train"]
        sizes = train.groupby(["player_key", "subset"]).size()
        self.assertTrue((sizes == 2).all())
        self.assertLessEqual(train["subset"].max(), 2)
        self.assertEqual(train.groupby("player_key")["game_id"].apply(lambda s: s.is_unique).all(), True)
        assert_split_integrity(manifest)

    def test_player_with_only_held_out_opponents_is_dropped(self):
        manifest, meta = select_v3_manifest(INDEX_ONLY_HELD_OUT_OPPONENTS, V2, games_per_player=2)
        self.assertEqual(meta["train_players_dropped"], 1)
        self.assertNotIn("t", set(manifest["player_key"]))
