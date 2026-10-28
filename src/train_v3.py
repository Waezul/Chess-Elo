"""Fit rapid8-v3: more rows, clock dropout, and one interval scale per bin.

Colab calls these functions from ``notebooks/train_player8_v3.ipynb``.
Validation chooses the row. Test is scored once, then again with clocks stripped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import calibrate_scales_by_bin, sort_quantiles, within_tolerance
from src.player_table import average_games_v3, drop_clocks, strip_clocks
from src.schema import (
    BIN_LABELS,
    CLOCK_DROPOUT,
    MACRO_BIN_MIN_N,
    MONOTONE_V3,
    PLAYER_MACRO_MAE_MAX,
    PLAYER_TAIL_BIAS_MAX,
    TAIL_BIN_INDEXES,
    V3_FEATURE_SETS,
    V3_PLAYER_FEATURES,
    V3_REVISION,
)
from src.train_quantile import (
    FROZEN_PARAMS_V3,
    SEARCH_GRID_V3,
    ablation_validation,
    build_matrices,
    cap_binding,
    engine_linear_macro,
    feature_report,
    gate_sample,
    predict_quantiles,
    search_validation,
    summarize_predictions_by_bin,
    write_bundle,
)


def v3_ship_decision(summary: dict) -> dict:
    """Macro MAE and tail bias only. Width and coverage are reported and do not block."""
    tails = {BIN_LABELS[index] for index in TAIL_BIN_INDEXES}
    biased = []
    for row in summary["bins"]:
        if int(row["n"]) < MACRO_BIN_MIN_N:
            continue
        label = str(row["bin"])
        if label in tails and abs(float(row["bias"])) > PLAYER_TAIL_BIAS_MAX:
            biased.append(label)
    macro = float(summary["macro_mae"])
    return {
        "ship_ready": bool(macro <= PLAYER_MACRO_MAE_MAX and not biased),
        "tail_bias_bins": biased,
        "macro_mae_max": PLAYER_MACRO_MAE_MAX,
        "tail_bias_max": PLAYER_TAIL_BIAS_MAX,
        "ship_bar": (
            "test macro MAE <= 120 and tail bias within ±80. "
            "Width and coverage are reported and do not block ship_ready."
        ),
    }


def _limit_players(frame: pd.DataFrame, max_players: int | None, seed: int) -> pd.DataFrame:
    """Keep at most ``max_players`` whole players in each split and rating bin."""
    if max_players is None or frame.empty:
        return frame
    rng = np.random.default_rng(seed)
    pieces = []
    for (_split, _elo_bin), group in frame.groupby(["split", "elo_bin"], sort=False):
        players = np.array(sorted(group["player_key"].astype(str).unique()))
        if len(players) > int(max_players):
            players = rng.choice(players, size=int(max_players), replace=False)
        pieces.append(group.loc[group["player_key"].astype(str).isin(set(map(str, players)))])
    return pd.concat(pieces, ignore_index=True) if pieces else frame.iloc[0:0]


def load_v3(data_dir: Path, max_players: int | None = None, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Average eight-game subsets, then blank clocks on 20% of the train rows."""
    games = pd.read_parquet(Path(data_dir) / "player_games_v3.parquet")
    if "subset" not in games.columns:
        games["subset"] = 0
    counts = games.groupby(["player_key", "subset"]).size()
    keep = counts[counts == 8].reset_index()[["player_key", "subset"]]
    games = games.merge(keep, on=["player_key", "subset"], how="inner")
    players = average_games_v3(games)
    players = _limit_players(players, max_players, seed)
    before = int((players["has_clock"] == 0).sum()) if "has_clock" in players.columns else 0
    splits = gate_sample(players)
    splits["train"] = drop_clocks(splits["train"], CLOCK_DROPOUT, seed).reset_index(drop=True)
    after = int((splits["train"]["has_clock"] == 0).sum()) if len(splits["train"]) else 0
    print(
        f"[load] train={len(splits['train']):,} val={len(splits['val']):,} test={len(splits['test']):,} "
        f"clock_dropout_rows={after - before}",
        flush=True,
    )
    return splits


def fit_v3(splits: dict[str, pd.DataFrame], grid=SEARCH_GRID_V3, base: dict | None = None) -> dict:
    """Search on validation, then report the ablation. Test labels are not read."""
    base = dict(FROZEN_PARAMS_V3 if base is None else base)
    built = build_matrices(splits, with_first_move=False, numeric=list(V3_PLAYER_FEATURES))
    y_train = splits["train"]["elo"].to_numpy(dtype=float)
    y_val = splits["val"]["elo"].to_numpy(dtype=float)
    val_bins = splits["val"]["elo_bin"].to_numpy()
    weight = splits["train"]["weight"].to_numpy(dtype=float)
    print("[search] validation search, then ablation", flush=True)
    searched = search_validation(
        built["matrices"]["train"],
        y_train,
        built["matrices"]["val"],
        y_val,
        val_bins,
        built["feature_names"],
        grid=grid,
        base=base,
        sample_weight=weight,
        monotone=MONOTONE_V3,
    )
    ablation = ablation_validation(
        built["matrices"]["train"],
        y_train,
        built["matrices"]["val"],
        y_val,
        val_bins,
        V3_FEATURE_SETS,
        weight,
        params=base,
    )
    q10, q50, q90 = predict_quantiles(searched["models"], built["matrices"]["val"])
    scales = calibrate_scales_by_bin(y_val, q10, q50, q90)
    linear = engine_linear_macro(splits["train"], splits["val"])
    cap = cap_binding(searched["models"], int(base["n_estimators"]))
    print(f"[search] linear_mae={linear:.3f} cap={cap or 'none'}", flush=True)
    print(ablation.to_string(index=False), flush=True)
    return {
        "models": searched["models"],
        "chosen": searched["chosen"],
        "rows": searched["rows"],
        "ablation": ablation,
        "linear_mae": float(linear),
        "scales": scales,
        "cap": cap,
        "base": base,
    }


def _score(models: dict, frame: pd.DataFrame, scales: dict) -> tuple[dict, float]:
    matrix = frame[list(V3_PLAYER_FEATURES)].astype(float)
    q10, q50, q90 = predict_quantiles(models, matrix)
    _low, mid, _high = sort_quantiles(q10, q50, q90)
    y_true = frame["elo"].to_numpy(dtype=float)
    summary = summarize_predictions_by_bin(y_true, frame["elo_bin"].to_numpy(), q10, q50, q90, scales)
    return summary, float(within_tolerance(y_true, mid, 150.0))


def score_test_once(fitted: dict, splits: dict[str, pd.DataFrame]) -> dict:
    """Score the held-out players once, and once more with every clock blanked."""
    print("[test] scoring test once", flush=True)
    summary, within = _score(fitted["models"], splits["test"], fitted["scales"])
    no_clock, _within_plain = _score(fitted["models"], strip_clocks(splits["test"]), fitted["scales"])
    decision = v3_ship_decision(summary)
    print(
        f"[test] macro_mae={summary['macro_mae']:.3f} within_150={within:.3f} "
        f"no_clock_macro_mae={no_clock['macro_mae']:.3f} ship_ready={decision['ship_ready']}",
        flush=True,
    )
    return {"summary": summary, "decision": decision, "within_150": within, "no_clock": no_clock}


def validation_report(fitted: dict, splits: dict[str, pd.DataFrame]):
    """Permutation importance of the chosen median on validation players."""
    built = build_matrices(splits, with_first_move=False, numeric=list(V3_PLAYER_FEATURES))
    return feature_report(
        fitted["models"]["q50"],
        built["matrices"]["val"],
        splits["val"]["elo"].to_numpy(dtype=float),
    )


def test_points(fitted: dict, splits: dict[str, pd.DataFrame]) -> tuple[np.ndarray, np.ndarray]:
    """Actual Elo and the predicted median for the scatter. Does not refit."""
    built = build_matrices(splits, with_first_move=False, numeric=list(V3_PLAYER_FEATURES))
    q10, q50, q90 = predict_quantiles(fitted["models"], built["matrices"]["test"])
    _low, mid, _high = sort_quantiles(q10, q50, q90)
    return splits["test"]["elo"].to_numpy(dtype=float), np.asarray(mid, dtype=float)


def save_v3(out_dir: Path, fitted: dict, tested: dict, splits: dict[str, pd.DataFrame], meta: dict) -> Path:
    """Write the bundle. Importance is fit on validation, not on the test players."""
    print("[save] writing bundle", flush=True)
    built = build_matrices(splits, with_first_move=False, numeric=list(V3_PLAYER_FEATURES))
    importance, top = feature_report(
        fitted["models"]["q50"],
        built["matrices"]["val"],
        splits["val"]["elo"].to_numpy(dtype=float),
    )
    card = {
        "revision": V3_REVISION,
        "feature_version": 3,
        "pool": "rapid",
        "source_label": "Lichess rapid, September 2026",
        "games_per_player": 8,
        "test_players": int(splits["test"]["player_key"].nunique()),
        "feature_names": list(V3_PLAYER_FEATURES),
        "numeric_features": list(V3_PLAYER_FEATURES),
        "interval_scales": fitted["scales"],
        "monotone": MONOTONE_V3,
        "clock_dropout": CLOCK_DROPOUT,
        "stockfish": meta.get("stockfish"),
        "search": {"chosen": fitted["chosen"], "rows": fitted["rows"]},
        "ablation": fitted["ablation"].to_dict(orient="records"),
        "cap_binding": fitted["cap"],
        "test_bins": tested["summary"]["bins"],
        "test_macro_mae": tested["summary"]["macro_mae"],
        "test_no_clock": tested["no_clock"],
        "within_150": tested["within_150"],
        "ship_ready": tested["decision"]["ship_ready"],
        "ship_bar": tested["decision"]["ship_bar"],
        "top_features": top,
        "watched_features": [item["feature"] for item in top],
        "linear_mae": fitted["linear_mae"],
    }
    return write_bundle(Path(out_dir), fitted["models"], None, card, importance)


def run_v3(data_dir, out_dir, max_players: int | None = None, grid=SEARCH_GRID_V3, smoke: bool = False) -> dict:
    """Load, fit, score test once, and save. ``--smoke`` uses a short grid."""
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    base = dict(FROZEN_PARAMS_V3)
    use_grid = grid
    if smoke:
        use_grid = tuple(SEARCH_GRID_V3[:2])
        base["n_estimators"] = 300
    splits = load_v3(data_dir, max_players=max_players)
    fitted = fit_v3(splits, grid=use_grid, base=base)
    tested = score_test_once(fitted, splits)
    meta_path = data_dir / "player_games_v3_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    path = save_v3(out_dir, fitted, tested, splits, meta)
    return {"splits": splits, "fitted": fitted, "tested": tested, "card": path}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train rapid8-v3 from player_games_v3.parquet.")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-players", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    run_v3(args.data_dir, args.out_dir, max_players=args.max_players, smoke=args.smoke)


if __name__ == "__main__":
    main()
