"""Smoke the v3 training entry point on a synthetic eight-game table."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.schema import BIN_LABELS, V3_CLOCK, V3_GAME_FEATURES, V3_PLAYER_FEATURES
from src.score import load_bundle, predict_player_v3
from src.train_quantile import SEARCH_GRID_V3
from src.train_v3 import run_v3


def _write_games(folder: Path) -> None:
    rng = np.random.default_rng(1)
    rows = []
    for bin_index, label in enumerate(BIN_LABELS):
        elo = float(800 + bin_index * 200 + 100)
        for player_index in range(40):
            if player_index < 24:
                split = "train"
            elif player_index < 32:
                split = "val"
            else:
                split = "test"
            cpl = (2400.0 - elo) / 16.0
            for _game in range(8):
                row = {column: float(rng.normal(0.0, 0.01)) for column in V3_GAME_FEATURES}
                row.update(
                    {
                        "player_key": f"{label}-{player_index}",
                        "subset": 0,
                        "elo": elo,
                        "split": split,
                        "elo_bin": label,
                        "cpl_mean": cpl,
                        "cpl_median": cpl * 0.8,
                        "eval_volatility": cpl * 0.1,
                        "mistake_rate": min(1.0, cpl / 200.0),
                        "blunder_rate": min(1.0, cpl / 400.0),
                        "best_move_rate": max(0.0, 1.0 - cpl / 150.0),
                        "acc_mean": max(0.0, 100.0 - cpl),
                        "wpl_mean": cpl / 10.0,
                    }
                )
                rows.append(row)
    frame = pd.DataFrame(rows)
    blank = rng.random(len(frame)) < 0.1
    for column in V3_CLOCK:
        if column == "has_clock":
            frame[column] = np.where(blank, 0.0, 1.0)
        else:
            frame[column] = rng.random(len(frame))
            frame.loc[blank, column] = np.nan
    frame.to_parquet(folder / "player_games_v3.parquet", index=False)


class V3RunTests(unittest.TestCase):
    def test_smoke_run_writes_a_rapid8_v3_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            data = folder / "data"
            out = folder / "out"
            data.mkdir()
            _write_games(data)
            run_v3(data, out, grid=SEARCH_GRID_V3[:1])
            card = json.loads((out / "model_card.json").read_text(encoding="utf-8"))
            self.assertEqual(card["revision"], "rapid8-v3")
            self.assertEqual(set(card["interval_scales"]), {*BIN_LABELS, "all"})
            self.assertEqual(card["feature_names"], V3_PLAYER_FEATURES)
            self.assertIn("test_no_clock", card)
            games = pd.read_parquet(data / "player_games_v3.parquet")
            test_players = games.loc[games["split"] == "test", "player_key"]
            one = games.loc[games["player_key"] == test_players.iloc[0]].reset_index(drop=True)
            scored = predict_player_v3(one, load_bundle(out))
            self.assertTrue(np.isfinite(scored[["q10", "q50", "q90"]].to_numpy()).all())
            self.assertLessEqual(scored.loc[0, "q10"], scored.loc[0, "q50"])
            self.assertLessEqual(scored.loc[0, "q50"], scored.loc[0, "q90"])
