"""Time-control parsing and one record per mainline move."""

from __future__ import annotations

import io
import unittest

import chess.pgn

from src.moves import analyze_moves, parse_time_control

PGN = (
    '[White "A"]\n[Black "B"]\n[Result "1-0"]\n\n'
    "1. e4 { [%clk 0:10:00] } 1... e5 { [%clk 0:10:00] } "
    "2. Nf3 { [%clk 0:09:50] } 2... Nc6 { [%clk 0:09:58] } 1-0\n"
)


def scripted(values):
    return lambda board: (values[board.ply()], None)


class TimeControlTests(unittest.TestCase):
    def test_parses_base_and_increment(self):
        self.assertEqual(parse_time_control("600+5"), (600, 5))
        self.assertEqual(parse_time_control("600"), (600, 0))

    def test_rejects_missing_and_garbage(self):
        for text in (None, "", "-", "abc", "600+x"):
            self.assertIsNone(parse_time_control(text))


class AnalyzeMovesTests(unittest.TestCase):
    def test_one_record_per_move_with_clock_and_white_pov(self):
        game = chess.pgn.read_game(io.StringIO(PGN))
        rows = analyze_moves(game, scripted([10, 20, -30, -30, 40]), "g1")
        self.assertEqual([r["ply"] for r in rows], [1, 2, 3, 4])
        self.assertEqual([r["color"] for r in rows], ["white", "black", "white", "black"])
        self.assertEqual((rows[2]["cp_before"], rows[2]["cp_after"]), (-30, -30))
        self.assertEqual(rows[2]["clock_after"], 590.0)

    def test_missing_clock_is_none(self):
        game = chess.pgn.read_game(io.StringIO('[Result "*"]\n\n1. e4 e5 *\n'))
        rows = analyze_moves(game, scripted([0, 0, 0]), "g2")
        self.assertTrue(all(r["clock_after"] is None for r in rows))
