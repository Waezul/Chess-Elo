"""Stockfish pass writes v3 side rows and one move table."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

import chess.pgn
import pandas as pd

from src.analyze_sample import OUTPUT_COLUMNS_V3, analyze_games
from src.engine import _local_stockfish
from src.pgn_stream import iter_games


def _game(site: str) -> str:
    return (
        f'[Event "Rated Rapid game"]\n'
        f'[Site "https://lichess.org/{site}"]\n'
        f'[White "Ann"]\n'
        f'[Black "Ben"]\n'
        f'[Result "1-0"]\n'
        f'[WhiteElo "1500"]\n'
        f'[BlackElo "1520"]\n'
        f'[TimeControl "600+5"]\n'
        f"\n"
        "1. e4 { [%clk 0:10:00] } 1... e5 { [%clk 0:10:00] } "
        "2. Nf3 { [%clk 0:09:50] } 2... Nc6 { [%clk 0:09:58] } 1-0\n"
        "\n"
    )


def _plies(text: str) -> int:
    game = chess.pgn.read_game(io.StringIO(text))
    return sum(1 for _ in game.mainline())


class V3AnalyzeTests(unittest.TestCase):
    @unittest.skipUnless(_local_stockfish(), "Stockfish binary is not installed")
    def test_v3_rows_moves_and_clock_share(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            text = _game("gameaaa1") + _game("gamebbb2")
            pgn = folder / "two.pgn"
            pgn.write_text(text, encoding="utf-8")
            located = list(iter_games(str(pgn)))
            self.assertEqual(len(located), 2)
            rows = []
            for start, headers, _moves, _next in located:
                site = [line for line in headers if line.startswith("[Site ")][0]
                game_id = site.split("/")[-1].rstrip('"]')
                for color, subset in (("white", 0), ("black", 1 if game_id == "gameaaa1" else 2)):
                    rows.append(
                        {
                            "offset": start,
                            "game_id": game_id,
                            "color": color,
                            "player_key": color,
                            "elo": 1500,
                            "elo_bin": "1400-1600",
                            "bin_index": 3,
                            "split": "train",
                            "thin_bin": False,
                            "subset": subset,
                        }
                    )
            manifest = folder / "manifest.parquet"
            pd.DataFrame(rows).to_parquet(manifest, index=False)
            output = folder / "player_games_v3.parquet"
            moves_path = folder / "moves_v3.parquet"
            stats = analyze_games(
                pgn,
                manifest,
                output,
                workers=1,
                stockfish=_local_stockfish(),
                depth=1,
                feature_version=3,
                moves_output=moves_path,
            )
            frame = pd.read_parquet(output)
            moves = pd.read_parquet(moves_path)
            self.assertEqual(list(frame.columns), OUTPUT_COLUMNS_V3)
            self.assertEqual(
                set(zip(frame["game_id"], frame["color"])),
                {("gameaaa1", "white"), ("gameaaa1", "black"), ("gamebbb2", "white"), ("gamebbb2", "black")},
            )
            subsets = {(row.game_id, row.color): int(row.subset) for row in frame.itertuples(index=False)}
            self.assertEqual(subsets[("gameaaa1", "white")], 0)
            self.assertEqual(subsets[("gameaaa1", "black")], 1)
            self.assertEqual(subsets[("gamebbb2", "black")], 2)
            self.assertEqual(len(moves), 2 * _plies(_game("gameaaa1")))
            self.assertTrue((frame["has_clock"] == 1.0).all())
            self.assertEqual(stats["feature_version"], 3)
            self.assertEqual(stats["has_clock_share"], 1.0)
            meta = json.loads((folder / "player_games_v3_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["feature_version"], 3)
            self.assertEqual(meta["has_clock_share"], 1.0)
            self.assertTrue(meta["stockfish"])
