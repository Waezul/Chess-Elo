"""Tail-first sample, header index, interval scale, and the quantile bundle."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.aggregate import assign_splits
from src.index_headers import index_pgn
from src.metrics import bin_index, bin_label, calibrate_scale, coverage, within_tolerance
from src.features import game_id_from_site
from src.pgn_stream import iter_games, parse_headers, read_game_at
from src.pipeline import main as pipeline_main
from src.sample import assert_split_integrity, select_manifest
from src.schema import BIN_LABELS, ENGINE_FEATURES, FEATURE_DESCRIPTIONS, GAME_EXTRA, NUMERIC_FEATURES
from src.score import average_quantiles, describe_feature


def _game(site: str, white: str, black: str, white_elo: int, black_elo: int, event: str = "Rated Classical game") -> str:
    return (
        f'[Event "{event}"]\n'
        f'[Site "https://lichess.org/{site}"]\n'
        f'[White "{white}"]\n'
        f'[Black "{black}"]\n'
        f'[Result "1-0"]\n'
        f'[WhiteElo "{white_elo}"]\n'
        f'[BlackElo "{black_elo}"]\n'
        f'[TimeControl "900+10"]\n'
        f'[Termination "Normal"]\n'
        f"\n"
        f"1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. O-O Nf6\n"
        f"\n"
    )


def _index_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class BinAndSampleTests(unittest.TestCase):
    def test_bin_edges(self):
        self.assertEqual(bin_index(800), 0)
        self.assertEqual(bin_index(1000), 0)
        self.assertEqual(bin_index(1001), 1)
        self.assertEqual(bin_index(1200), 1)
        self.assertEqual(bin_index(2200), 6)
        self.assertEqual(bin_index(2201), 7)
        self.assertEqual(bin_index(2400), 7)
        self.assertIsNone(bin_index(799))
        self.assertIsNone(bin_index(2401))
        self.assertEqual(bin_label(1600), "1400-1600")
        self.assertEqual(len(BIN_LABELS), 8)

    def test_player_cap_and_middle_is_not_kept_whole(self):
        rows = []
        for index in range(6):
            rows.append(
                {
                    "offset": index,
                    "game_id": f"g{index}",
                    "white_key": "solo",
                    "black_key": f"opp{index}",
                    "white_elo": 1500,
                    "black_elo": 1510,
                }
            )
        manifest, meta = select_manifest(
            _index_frame(rows),
            games_per_bin=100,
            per_player=4,
            thin_players=500,
            split_map={"solo": "train", **{f"opp{i}": "train" for i in range(6)}},
        )
        white = manifest.loc[manifest["player_key"] == "solo"]
        self.assertEqual(len(white), 4)
        middle = next(row for row in meta["bins"] if row["bin"] == "1400-1600")
        self.assertFalse(middle["thin"])
        self.assertFalse(middle["keep_all"])

    def test_middle_respects_side_quota_even_when_few_players(self):
        rows = []
        for index in range(5):
            rows.append(
                {
                    "offset": index,
                    "game_id": f"g{index}",
                    "white_key": f"w{index}",
                    "black_key": f"b{index}",
                    "white_elo": 1500,
                    "black_elo": 1520,
                }
            )
        keys = [f"w{i}" for i in range(5)] + [f"b{i}" for i in range(5)]
        manifest, _meta = select_manifest(
            _index_frame(rows),
            games_per_bin=2,
            per_player=4,
            thin_players=500,
            split_map={key: "train" for key in keys},
        )
        self.assertEqual(int((manifest["elo_bin"] == "1400-1600").sum()), 2)

    def test_thin_tail_keeps_every_side(self):
        rows = []
        offset = 0
        for player in range(3):
            for game in range(5):
                rows.append(
                    {
                        "offset": offset,
                        "game_id": f"g{offset}",
                        "white_key": f"low{player}",
                        "black_key": f"mid{offset}",
                        "white_elo": 900,
                        "black_elo": 1500,
                    }
                )
                offset += 1
        keys = {row["white_key"] for row in rows} | {row["black_key"] for row in rows}
        manifest, meta = select_manifest(
            _index_frame(rows),
            games_per_bin=2,
            per_player=1,
            thin_players=10,
            split_map={key: "train" for key in keys},
        )
        low = manifest.loc[manifest["elo_bin"] == "800-1000"]
        self.assertEqual(len(low), 15)
        self.assertTrue(bool(low["thin_bin"].all()))
        tail = next(row for row in meta["bins"] if row["bin"] == "800-1000")
        self.assertTrue(tail["thin"])
        self.assertEqual(tail["index_players"], 3)

    def test_cross_split_keeps_the_tail_side(self):
        index = _index_frame(
            [
                {
                    "offset": 10,
                    "game_id": "g1",
                    "white_key": "tail_player",
                    "black_key": "mid_player",
                    "white_elo": 900,
                    "black_elo": 1500,
                }
            ]
        )
        manifest, meta = select_manifest(
            index,
            games_per_bin=10,
            per_player=4,
            thin_players=0,
            split_map={"tail_player": "test", "mid_player": "train"},
        )
        self.assertEqual(manifest["color"].tolist(), ["white"])
        self.assertEqual(manifest["split"].tolist(), ["test"])
        self.assertEqual(meta["dropped_cross_split_sides"], 1)
        assert_split_integrity(manifest)

    def test_same_split_keeps_both_sides(self):
        index = _index_frame(
            [
                {
                    "offset": 3,
                    "game_id": "g1",
                    "white_key": "a",
                    "black_key": "b",
                    "white_elo": 2100,
                    "black_elo": 1500,
                }
            ]
        )
        manifest, meta = select_manifest(
            index,
            thin_players=0,
            split_map={"a": "val", "b": "val"},
        )
        self.assertEqual(len(manifest), 2)
        self.assertEqual(set(manifest["split"]), {"val"})
        self.assertEqual(meta["dropped_cross_split_sides"], 0)

    def test_bad_split_frame_is_rejected(self):
        frame = pd.DataFrame(
            {
                "player_key": ["a", "a"],
                "game_id": ["g1", "g2"],
                "split": ["train", "test"],
            }
        )
        with self.assertRaises(ValueError):
            assert_split_integrity(frame)


class IntervalTests(unittest.TestCase):
    def test_scale_moves_coverage_toward_the_target(self):
        y = np.arange(100, dtype=float)
        q50 = np.full(100, 50.0)
        q10 = np.full(100, 40.0)
        q90 = np.full(100, 60.0)
        scale, _covered = calibrate_scale(y, q10, q50, q90, target=0.8)
        from src.metrics import apply_interval

        low, _mid, high = apply_interval(q10, q50, q90, scale)
        self.assertGreater(coverage(y, low, high), coverage(y, q10, q90))
        self.assertGreater(scale, 1.0)
        self.assertAlmostEqual(coverage(y, low, high), 0.8, delta=0.03)

    def test_feature_blurbs_cover_the_signals_the_pilot_used(self):
        for name in (
            "avg_legal_moves",
            "mat_middlegame_mean",
            "best_move_rate",
            "cpl_mean",
            "mistake_rate",
            "blunder_rate",
        ):
            self.assertIn(name, FEATURE_DESCRIPTIONS)
            self.assertGreater(len(describe_feature(name)), 30)
        self.assertIn("e4", describe_feature("first_move_e4"))

    def test_average_quantiles(self):
        averaged = average_quantiles(pd.DataFrame({"q10": [1000.0, 1200.0], "q50": [1400.0, 1600.0], "q90": [1800.0, 2000.0]}))
        self.assertEqual(averaged["q50"], 1500.0)
        self.assertEqual(averaged["n_games"], 2)


class IndexPipelineTests(unittest.TestCase):
    def _pgn(self, folder: Path) -> Path:
        blocks = [
            _game("game00", "Ann", "Ben", 900, 1500),
            _game("game01", "Cia", "Dan", 2100, 1600),
            _game("blitz1", "Ann", "Ben", 900, 1500, event="Rated Blitz game"),
            _game("game02", "Eve", "Fay", 1300, 1700),
            _game("low", "Gus", "Hua", 500, 1500),
        ]
        path = folder / "mini.pgn"
        path.write_text("\n".join(blocks), encoding="utf-8")
        return path

    def test_iter_games_resume_and_index_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            pgn = self._pgn(folder)
            games = list(iter_games(str(pgn)))
            self.assertGreaterEqual(len(games), 5)
            rest = list(iter_games(str(pgn), start=games[2][3]))
            self.assertEqual(rest[0][0], games[3][0])
            self.assertEqual(len(rest), len(games) - 3)

            output = folder / "header_index.parquet"
            first = index_pgn(pgn, output, fresh=True, checkpoint_seconds=0)
            self.assertTrue(first["completed"])
            second = index_pgn(pgn, output, checkpoint_seconds=0)
            self.assertTrue(second["completed"])
            frame = pd.read_parquet(output)
            self.assertEqual(set(frame["game_id"]), {"game00", "game01", "game02"})
            self.assertEqual(len(frame), 3)
            for row in frame.itertuples(index=False):
                parsed = read_game_at(str(pgn), int(row.offset))
                self.assertIsNotNone(parsed)
                headers = parse_headers(parsed[0])
                self.assertEqual(game_id_from_site(headers["Site"]), row.game_id)

    def test_pipeline_writes_a_player_disjoint_sample_without_stockfish(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            pgn = self._pgn(folder)
            pipeline_main(
                [
                    "--input",
                    str(pgn),
                    "--output-dir",
                    str(folder),
                    "--no-stockfish",
                    "--workers",
                    "1",
                    "--checkpoint-seconds",
                    "0",
                    "--thin-players",
                    "0",
                ]
            )
            manifest = pd.read_parquet(folder / "sample_manifest.parquet")
            games = pd.read_parquet(folder / "games_sample.parquet")
            self.assertFalse(manifest.empty)
            self.assertEqual(len(games), len(manifest))
            self.assertTrue({"split", "elo_bin", "thin_bin", "cpl_mean"}.issubset(games.columns))
            assert_split_integrity(games)
            self.assertTrue(set(games["game_id"]).issubset({"game00", "game01", "game02"}))
            pipeline_main(
                [
                    "--input",
                    str(pgn),
                    "--output-dir",
                    str(folder),
                    "--no-stockfish",
                    "--workers",
                    "1",
                ]
            )
            again = pd.read_parquet(folder / "games_sample.parquet")
            self.assertEqual(len(again), len(games))


class QuantileBundleTests(unittest.TestCase):
    def test_fit_scores_and_describes_features(self):
        rng = np.random.default_rng(0)
        n = 240
        numeric = [*GAME_EXTRA, *NUMERIC_FEATURES, *ENGINE_FEATURES]
        frame = pd.DataFrame({column: np.zeros(n) for column in numeric})
        cpl = rng.uniform(0.05, 0.85, n)
        frame["cpl_mean"] = cpl
        frame["cpl_median"] = cpl * 0.8
        frame["mistake_rate"] = np.clip(cpl * 0.6, 0, 1)
        frame["blunder_rate"] = np.clip(cpl * 0.3, 0, 1)
        frame["best_move_rate"] = 1 - cpl
        frame["is_white"] = rng.integers(0, 2, n)
        frame["first_move"] = rng.choice(["e4", "d4", "c5", "e5"], n)
        frame["elo"] = 2100 - 1100 * cpl
        frame["elo_bin"] = [bin_label(value) for value in frame["elo"]]
        frame["player_key"] = [f"p{index // 4}" for index in range(n)]
        frame["game_id"] = [f"g{index}" for index in range(n)]
        frame["color"] = "white"
        mapping = assign_splits(frame["player_key"].unique().tolist(), seed=42, test_size=0.2, val_size=0.2)
        frame["split"] = frame["player_key"].map(mapping)

        from src.train_quantile import (
            build_matrices,
            engine_linear_macro,
            feature_report,
            fit_quantile_trio,
            gate_sample,
            predict_quantiles,
            ship_decision,
            summarize_predictions,
            write_bundle,
        )
        from src.metrics import calibrate_scale as fit_scale
        from src.score import load_bundle, predict_rows

        splits = gate_sample(frame)
        built = build_matrices(splits)
        params = {
            "n_estimators": 40,
            "max_depth": 3,
            "learning_rate": 0.15,
            "min_child_weight": 1,
            "reg_lambda": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "tree_method": "hist",
            "early_stopping_rounds": 8,
            "n_jobs": 1,
        }
        models = fit_quantile_trio(
            built["matrices"]["train"],
            splits["train"]["elo"].to_numpy(dtype=float),
            built["matrices"]["val"],
            splits["val"]["elo"].to_numpy(dtype=float),
            splits["val"]["elo_bin"].to_numpy(),
            built["feature_names"],
            params=params,
        )
        q10, q50, q90 = predict_quantiles(models, built["matrices"]["val"])
        scale, _val_coverage = fit_scale(
            splits["val"]["elo"].to_numpy(dtype=float), q10, q50, q90, target=0.8
        )
        summary = summarize_predictions(
            splits["test"]["elo"].to_numpy(dtype=float),
            splits["test"]["elo_bin"].to_numpy(),
            *predict_quantiles(models, built["matrices"]["test"]),
            scale,
        )
        mean_pred = np.full(len(splits["test"]), float(splits["train"]["elo"].mean()))
        mean_mae = float(np.mean(np.abs(mean_pred - splits["test"]["elo"].to_numpy(dtype=float))))
        self.assertLess(summary["mae"], mean_mae)
        self.assertGreater(scale, 0)
        decision = ship_decision(summary)
        self.assertIn("ship_ready", decision)

        _table, top = feature_report(
            models["q50"],
            built["matrices"]["val"],
            splits["val"]["elo"].to_numpy(dtype=float),
            n_repeats=2,
            max_rows=80,
            top_k=5,
        )
        self.assertEqual(len(top), 5)
        self.assertTrue(all(item["description"] for item in top))
        linear = engine_linear_macro(splits["train"], splits["val"])
        self.assertGreater(linear, 0)

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            card = {
                "feature_names": built["feature_names"],
                "numeric_features": built["numeric_features"],
                "interval_scale": scale,
                "watched_features": [item["feature"] for item in top],
                "top_features": top,
                "test": {key: summary[key] for key in ("coverage", "macro_mae", "mae", "bins")},
                "ship_ready": decision["ship_ready"],
                "wide_bins": decision["wide_bins"],
            }
            write_bundle(directory, models, built["encoder"], card)
            bundle = load_bundle(directory)
            scored = predict_rows(splits["test"].head(8), bundle, contributions=True)
            self.assertTrue((scored["q10"] <= scored["q50"]).all())
            self.assertTrue((scored["q50"] <= scored["q90"]).all())
            self.assertEqual(len(scored.loc[0, "contributions"]), 5)
            self.assertIn("description", scored.loc[0, "contributions"][0])


class SearchGridTests(unittest.TestCase):
    def test_grid_is_the_frozen_point_and_five_neighbors(self):
        from src.train_quantile import FROZEN_PARAMS, SEARCH_GRID

        names = [row["name"] for row in SEARCH_GRID]
        self.assertEqual(
            names,
            ["frozen", "depth-3", "depth-5", "min-child-5", "min-child-20", "lr-0.1"],
        )
        frozen = SEARCH_GRID[0]
        self.assertEqual(frozen["max_depth"], FROZEN_PARAMS["max_depth"])
        self.assertEqual(frozen["min_child_weight"], FROZEN_PARAMS["min_child_weight"])
        self.assertEqual(frozen["learning_rate"], FROZEN_PARAMS["learning_rate"])
        moved = [
            (
                row["max_depth"] != FROZEN_PARAMS["max_depth"],
                row["min_child_weight"] != FROZEN_PARAMS["min_child_weight"],
                row["learning_rate"] != FROZEN_PARAMS["learning_rate"],
            )
            for row in SEARCH_GRID[1:]
        ]
        self.assertTrue(all(sum(flags) == 1 for flags in moved))

    def test_choose_search_row_ignores_within_150_and_keeps_frozen_ties(self):
        from src.train_quantile import choose_search_row

        rows = [
            {"name": "frozen", "frozen": True, "val_macro_mae": 80.0, "val_within_150": 0.40},
            {"name": "depth-5", "frozen": False, "val_macro_mae": 80.0, "val_within_150": 0.90},
            {"name": "lr-0.1", "frozen": False, "val_macro_mae": 79.0, "val_within_150": 0.10},
        ]
        self.assertEqual(choose_search_row(rows)["name"], "lr-0.1")
        self.assertEqual(choose_search_row(rows[:2])["name"], "frozen")

    def test_within_tolerance_counts_the_closed_band(self):
        actual = np.array([1000.0, 1000.0, 1000.0])
        guessed = np.array([1000.0, 1150.0, 1151.0])
        self.assertAlmostEqual(within_tolerance(actual, guessed, 150), 2 / 3)


if __name__ == "__main__":
    unittest.main()
