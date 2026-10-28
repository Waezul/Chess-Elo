"""Score saved quantile models and name the features that moved a guess."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import apply_interval, apply_interval_by_bin
from src.schema import FEATURE_DESCRIPTIONS


def describe_feature(name: str) -> str:
    if name in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[name]
    prefix = "first_move_"
    if name.startswith(prefix):
        return f"This side's first move was {name[len(prefix):]}."
    return "Column in the game model."


def design_matrix(
    frame: pd.DataFrame,
    feature_names: list[str],
    numeric_features: list[str],
    encoder,
) -> pd.DataFrame:
    """Rebuild the training matrix. Missing one-hot columns stay zero."""
    missing = [column for column in numeric_features if column not in frame.columns]
    if missing:
        raise ValueError(f"missing features: {missing}")
    numeric = frame[numeric_features].reset_index(drop=True).astype(float)
    if encoder is None:
        return numeric.reindex(columns=list(feature_names), fill_value=0.0)
    if "first_move" not in frame.columns:
        raise ValueError("missing features: ['first_move']")
    encoded = pd.DataFrame(
        encoder.transform(frame[["first_move"]].fillna("missing").astype(str)),
        columns=list(encoder.get_feature_names_out(["first_move"])),
    )
    design = pd.concat([numeric, encoded], axis=1)
    return design.reindex(columns=list(feature_names), fill_value=0.0)


def load_bundle(directory: Path) -> dict:
    import joblib
    import xgboost as xgb

    card = json.loads((directory / "model_card.json").read_text(encoding="utf-8"))
    models = {}
    for name, filename in card["models"].items():
        booster = xgb.Booster()
        booster.load_model(str(directory / filename))
        models[name] = booster
    encoder = joblib.load(directory / card["encoder"])
    return {"card": card, "models": models, "encoder": encoder}


def _predict(booster, matrix: pd.DataFrame) -> np.ndarray:
    import xgboost as xgb

    dmatrix = xgb.DMatrix(matrix.to_numpy(), feature_names=list(matrix.columns))
    return np.asarray(booster.predict(dmatrix), dtype=float).reshape(-1)


def predict_rows(frame: pd.DataFrame, bundle: dict, contributions: bool = False) -> pd.DataFrame:
    """Return q10, q50, q90 after the validation interval scale, one row per input row."""
    card = bundle["card"]
    matrix = design_matrix(frame, card["feature_names"], card["numeric_features"], bundle["encoder"])
    q10 = _predict(bundle["models"]["q10"], matrix)
    q50 = _predict(bundle["models"]["q50"], matrix)
    q90 = _predict(bundle["models"]["q90"], matrix)
    scales = card.get("interval_scales")
    if isinstance(scales, dict) and scales:
        low, mid, high = apply_interval_by_bin(q10, q50, q90, scales)
    else:
        low, mid, high = apply_interval(q10, q50, q90, float(card.get("interval_scale", 1.0)))
    scored = pd.DataFrame({"q10": low, "q50": mid, "q90": high})
    if contributions:
        scored["contributions"] = top_contributions(
            bundle["models"]["q50"],
            matrix,
            list(card.get("watched_features", [])),
        )
    return scored


def top_contributions(booster, matrix: pd.DataFrame, watched: list[str], k: int = 5) -> list[list[dict]]:
    """Per-row contributions for the studied features. Positive raises the guess."""
    import xgboost as xgb

    dmatrix = xgb.DMatrix(matrix.to_numpy(), feature_names=list(matrix.columns))
    values = np.asarray(booster.predict(dmatrix, pred_contribs=True), dtype=float)
    names = list(matrix.columns)
    if values.shape[1] != len(names) + 1:
        raise RuntimeError("contribution width does not match the feature list")
    watch = watched or names
    rows = []
    for line in values:
        ranked = []
        for name in watch:
            if name not in names:
                continue
            ranked.append(
                {
                    "feature": name,
                    "contribution": float(line[names.index(name)]),
                    "description": describe_feature(name),
                }
            )
        ranked.sort(key=lambda item: abs(item["contribution"]), reverse=True)
        rows.append(ranked[:k])
    return rows


def average_quantiles(scored: pd.DataFrame) -> dict[str, float | int]:
    """Several games from one player: average the three quantiles."""
    if scored.empty:
        raise ValueError("no games to average")
    return {
        "q10": float(scored["q10"].mean()),
        "q50": float(scored["q50"].mean()),
        "q90": float(scored["q90"].mean()),
        "n_games": int(len(scored)),
    }


def predict_eight_games(games: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Average the features of eight games, then score that one row.

    Averaging eight one-game guesses is a different model. Those guesses share
    the same tail bias, so their mean stays biased.
    """
    from src.player_table import average_games

    expected = int(bundle["card"].get("games_per_player") or 0)
    if "player_key" in games.columns and int(games["player_key"].nunique()) > 1:
        raise ValueError("score one player at a time")
    if expected and len(games) != expected:
        raise ValueError(f"this model scores {expected} games at a time, got {len(games)}")
    return predict_rows(average_games(games), bundle)


def predict_player_v3(games: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Average one player's eight games with the v3 table, then score that row."""
    from src.player_table import average_games_v3

    expected = int(bundle["card"].get("games_per_player") or 0)
    if "player_key" in games.columns and int(games["player_key"].nunique()) > 1:
        raise ValueError("score one player at a time")
    if expected and len(games) != expected:
        raise ValueError(f"this model scores {expected} games at a time, got {len(games)}")
    table = games.copy()
    if "subset" not in table.columns:
        table["subset"] = 0
    return predict_rows(average_games_v3(table), bundle)
