"""Behavior features, filters, player split, and centipawn summaries."""

import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np
import pandas as pd

from src.aggregate import aggregate_players, assign_splits, normalized_entropy
from src.engine import analyze_game, cp_white_from_score, summarize_moves
from src.extract import GAME_COLUMNS, extract_path
from src.features import accept_headers, player_key, rows_from_movetext
from src.pgn_stream import iter_records, iter_records_with_offset, parse_headers
from src.schema import NON_FEATURES

MINI = """[Event "Rated Classical game"]
[Site "https://lichess.org/aaaaaaaa"]
[White "Alice"]
[Black "Bob"]
[Result "1-0"]
[WhiteElo "1500"]
[BlackElo "1600"]
[ECO "C60"]
[TimeControl "900+10"]
[Termination "Normal"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. O-O Nf6

[Event "Rated Blitz game"]
[Site "https://lichess.org/bbbbbbbb"]
[White "Alice"]
[Black "Bob"]
[Result "0-1"]
[WhiteElo "1500"]
[BlackElo "1600"]
[TimeControl "180+0"]
[Termination "Normal"]

1. e4 e5

[Event "Rated Correspondence game"]
[Site "https://lichess.org/cccccccc"]
[White "Alice"]
[Black "Bob"]
[Result "1-0"]
[WhiteElo "1500"]
[BlackElo "1600"]
[TimeControl "-"]
[Termination "Normal"]

1. e4 e5

[Event "Rated Classical game"]
[Site "https://lichess.org/dddddddd"]
[White "Cara"]
[Black "Dan"]
[Result "1-0"]
[WhiteElo "1500?"]
[BlackElo "1600"]
[TimeControl "900+10"]
[Termination "Normal"]

1. e4 e5

[Event "Rated Classical game"]
[Site "https://lichess.org/eeeeeeee"]
[White "Eve"]
[Black "Frank"]
[Result "0-1"]
[WhiteElo "500"]
[BlackElo "1600"]
[TimeControl "900+10"]
[Termination "Time forfeit"]

1. e4 e5
"""


class FeatureTests(unittest.TestCase):
    def test_filter_keeps_one_classical_pool(self):
        records = list(iter_records(MINI.splitlines()))
        self.assertEqual(len(records), 5)
        accepted = [parse_headers(headers) for headers, _moves in records if accept_headers(parse_headers(headers))]
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["Site"].rsplit("/", 1)[-1], "aaaaaaaa")

    def test_subject_rows_exclude_names_and_opponent_elo(self):
        headers, moves = next(iter_records(MINI.splitlines()))
        headers = parse_headers(headers)
        rows = rows_from_movetext(headers, moves)
        self.assertEqual(len(rows), 2)
        white, black = rows
        self.assertEqual(white["color"], "white")
        self.assertEqual(white["elo"], 1500)
        self.assertEqual(black["elo"], 1600)
        self.assertEqual(white["player_key"], player_key("alice"))
        self.assertNotEqual(white["player_key"], black["player_key"])
        self.assertEqual(white["eco_family"], "C6")
        self.assertEqual(white["first_move"], "e4")
        self.assertEqual(black["first_move"], "e5")
        self.assertEqual(white["n_subject_moves"], 4)
        self.assertEqual(black["n_subject_moves"], 4)
        self.assertAlmostEqual(white["rate_pawn"], 0.25)
        self.assertAlmostEqual(white["rate_knight"], 0.25)
        self.assertAlmostEqual(white["rate_bishop"], 0.25)
        self.assertAlmostEqual(white["rate_king"], 0.25)
        self.assertEqual(white["castled"], 1.0)
        self.assertEqual(white["castle_kingside"], 1.0)
        self.assertEqual(white["castle_ply"], 7)
        self.assertEqual(black["castled"], 0.0)
        self.assertIsNone(black["castle_ply"])
        self.assertEqual(white["developed_minors_ply10"], 2)
        self.assertEqual(black["developed_minors_ply10"], 2)
        self.assertAlmostEqual(white["mat_opening_mean"], 0.0)
        self.assertIsNone(white["mat_middlegame_mean"])
        self.assertEqual(white["result_subject"], 1.0)
        self.assertEqual(black["result_subject"], 0.0)
        for row in rows:
            self.assertNotIn("Alice", row.values())
            self.assertNotIn("Bob", row.values())
            self.assertNotIn(1600 if row["color"] == "white" else 1500, (row["elo"],))
            blob = " ".join(str(value) for value in row.values())
            self.assertNotIn("lichess.org", blob)
        self.assertEqual(white["game_id"], "aaaaaaaa")
        forbidden = {"White", "Black", "WhiteElo", "BlackElo", "opponent_elo", "WhiteRatingDiff"}
        self.assertTrue(forbidden.isdisjoint(GAME_COLUMNS))
        self.assertTrue({"player_key", "elo", "result_subject", "n_games"}.issubset(NON_FEATURES | {"n_games"}))

    def test_extract_matches_across_workers_and_zst(self):
        import zstandard

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            pgn = folder / "mini.pgn"
            # Repeat the classical game with distinct ids so chunks have work.
            blocks = []
            for index in range(12):
                blocks.append(
                    MINI.split("[Event \"Rated Blitz")[0].replace("aaaaaaaa", f"game{index:02d}").replace(
                        "Alice", f"Player{index % 3}"
                    )
                )
            pgn.write_text("\n".join(blocks), encoding="utf-8")
            single = folder / "one.parquet"
            multi = folder / "multi.parquet"
            extract_path(pgn, single, "classical", 800, 2400, None, 1, 1, checkpoint=False)
            extract_path(pgn, multi, "classical", 800, 2400, None, 3, 1, checkpoint=False)
            left = pd.read_parquet(single).sort_values(["game_id", "color"]).reset_index(drop=True)
            right = pd.read_parquet(multi).sort_values(["game_id", "color"]).reset_index(drop=True)
            self.assertEqual(len(left), 24)
            pd.testing.assert_frame_equal(left, right)
            compressed = folder / "mini.pgn.zst"
            compressor = zstandard.ZstdCompressor()
            compressed.write_bytes(compressor.compress(pgn.read_bytes()))
            zst_out = folder / "zst.parquet"
            extract_path(compressed, zst_out, "classical", 800, 2400, None, 4, 4, checkpoint=False)
            zst_frame = pd.read_parquet(zst_out).sort_values(["game_id", "color"]).reset_index(drop=True)
            pd.testing.assert_frame_equal(left, zst_frame)

    def test_offset_iterator_and_checkpoint_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            pgn = folder / "chunk.pgn"
            blocks = []
            for index in range(6):
                blocks.append(
                    MINI.split('[Event "Rated Blitz')[0]
                    .replace("aaaaaaaa", f"game{index:02d}")
                    .replace("Alice", f"Player{index}")
                )
            pgn.write_text("\n".join(blocks), encoding="utf-8")

            records = list(iter_records_with_offset(str(pgn)))
            self.assertEqual(len(records), 6)
            mid = records[2][2]
            resumed = list(iter_records_with_offset(str(pgn), start=mid))
            self.assertEqual(len(resumed), 3)

            out = folder / "games.parquet"
            first = extract_path(
                pgn, out, "classical", 800, 2400, 2, 1, 1, checkpoint=True, checkpoint_minutes=0
            )
            self.assertTrue(first["completed"])
            self.assertEqual(first["kept"], 2)
            partial_rows = first["rows"]

            # Simulate an interrupted longer run by clearing completion and limiting via resume state.
            second = extract_path(
                pgn, out, "classical", 800, 2400, None, 1, 1, checkpoint=True, checkpoint_minutes=0, fresh=True
            )
            self.assertTrue(second["completed"])
            self.assertEqual(second["kept"], 6)
            self.assertGreater(second["rows"], partial_rows)

    def test_split_then_aggregate_hides_game_count(self):
        rows = []
        for player, elo, opening, result in [
            ("p1", 1500, "C6", 1.0),
            ("p1", 1510, "C6", 0.0),
            ("p1", 1490, "B0", 0.5),
            ("p2", 1800, "A0", 1.0),
        ]:
            row = {column: 0.0 for column in GAME_COLUMNS}
            row.update(
                {
                    "game_id": f"{player}-{elo}",
                    "player_key": player,
                    "color": "white",
                    "elo": elo,
                    "result_subject": result,
                    "eco_family": opening,
                    "first_move": "e4" if player == "p1" else "d4",
                    "split": "train",
                }
            )
            rows.append(row)
        games = pd.DataFrame(rows)
        players = aggregate_players(games, min_games=2)
        self.assertEqual(players["player_key"].tolist(), ["p1"])
        self.assertEqual(players.loc[0, "elo"], 1500.0)
        self.assertEqual(players.loc[0, "n_games"], 3)
        self.assertGreater(players.loc[0, "opening_entropy"], 0)
        self.assertEqual(players.loc[0, "first_move_entropy"], 0.0)
        self.assertNotIn("n_games_mean", players.columns)
        mapping = assign_splits(["a", "b", "c", "d", "e"], seed=42, test_size=0.2, val_size=0.2)
        self.assertEqual(set(mapping.values()), {"train", "val", "test"})
        self.assertEqual(normalized_entropy(pd.Series(["e4", "e4"])), 0.0)


class EngineMathTests(unittest.TestCase):
    def test_cpl_summary_and_subject_only(self):
        summary = summarize_moves([0, 100, 250], [10, 20, 30], [True, False, False])
        self.assertAlmostEqual(summary["cpl_mean"], 350 / 3)
        self.assertEqual(summary["cpl_median"], 100)
        self.assertAlmostEqual(summary["mistake_rate"], 2 / 3)
        self.assertAlmostEqual(summary["blunder_rate"], 1 / 3)
        self.assertAlmostEqual(summary["best_move_rate"], 1 / 3)
        self.assertAlmostEqual(summary["eval_volatility"], float(np.std([10, 20, 30], ddof=0)))

        class Score:
            def __init__(self, cp):
                self._cp = cp

            def is_mate(self):
                return False

            def score(self, mate_score=1000):
                return self._cp

            def mate(self):
                return None

        class Pov:
            def white(self):
                return Score(50)

        self.assertEqual(cp_white_from_score(Pov()), 50)

        headers, moves = next(iter_records(MINI.splitlines()))
        headers = parse_headers(headers)
        # Build a python-chess game from the same movetext the extractor uses.
        from src.features import rows_from_movetext as _rows

        self.assertEqual(len(_rows(headers, moves)), 2)
        import io

        import chess.pgn

        text = "\n".join(f'[{k} "{headers[k]}"]' for k in ("White", "Black", "Result"))
        text += "\n\n" + "\n".join(moves) + "\n"
        game = chess.pgn.read_game(io.StringIO(text))

        def analyse(board: chess.Board):
            # Constant eval => zero loss. e2e4 matches only White's first move.
            return 0, "e2e4"

        white_rows = analyze_game(game, {"white"}, analyse)
        self.assertEqual(len(white_rows), 1)
        self.assertEqual(white_rows[0]["color"], "white")
        self.assertAlmostEqual(white_rows[0]["cpl_mean"], 0.0)
        self.assertAlmostEqual(white_rows[0]["best_move_rate"], 0.25)

        from src.engine import apply_engine_features

        attached = apply_engine_features(headers, moves, rows_from_movetext(headers, moves), analyse)
        self.assertAlmostEqual(attached[0]["cpl_mean"], 0.0)
        self.assertAlmostEqual(attached[0]["best_move_rate"], 0.25)
        self.assertAlmostEqual(attached[1]["best_move_rate"], 0.0)
        self.assertIn("cpl_mean", GAME_COLUMNS)
        self.assertIn("best_move_rate", GAME_COLUMNS)


if __name__ == "__main__":
    unittest.main()
