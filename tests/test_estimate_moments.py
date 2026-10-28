"""Dashboard moments from the same scripted pass that scores the eight games."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.catalog import load_catalog
from src.estimate import score_selection
from tests.test_api import game

ROOT = Path(__file__).resolve().parents[1] / "model"
# White-POV centipawns at each ply of the first game. White losses are 40, 250, 180, then 10.
GAME0 = [
    0,
    -40,
    -40,
    -290,
    -290,
    -470,
    -470,
    -480,
    -480,
    -490,
    -490,
    -500,
    -500,
    -510,
    -510,
    -520,
    -520,
    -530,
    -530,
    -540,
    -540,
]
BEST = {0: "e2e4", 2: "d2d4", 4: "d2d4"}


def scripted_games():
    seen = {"game": -1}

    def analyse(board):
        if board.ply() == 0:
            seen["game"] += 1
        ply = board.ply()
        if seen["game"] == 0 and ply < len(GAME0):
            return GAME0[ply], BEST.get(ply)
        return 0, None

    return analyse


class EstimateMomentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog(ROOT)

    def test_worst_moves_keep_order_blunder_flag_and_quantiles(self):
        paste = "\n".join(
            game("Rated Rapid game", site=f"https://lichess.org/moment{i}") for i in range(8)
        )
        estimate = score_selection(self.catalog, paste, "Ada", "rapid", list(range(8)), scripted_games())
        moments = estimate["moments"]
        self.assertEqual([row["cp_loss"] for row in moments], [250, 180, 40])
        self.assertEqual([row["blunder"] for row in moments], [True, False, False])
        self.assertEqual([row["ply"] for row in moments], [3, 5, 1])
        self.assertEqual([row["played"] for row in moments], ["Nf3", "Bb5", "e4"])
        self.assertEqual([row["best"] for row in moments], ["d4", "d4", "e4"])
        self.assertEqual([row["game_index"] for row in moments], [0, 0, 0])
        self.assertIsNone(estimate["stats"]["clk_low_rate"])
        self.assertIsNone(estimate["stats"]["blunder_in_time_trouble_rate"])
        for key in ("q10", "q50", "q90"):
            self.assertIn(key, estimate)
            self.assertEqual(estimate[key], estimate[key])
