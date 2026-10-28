"""Fit the site model: three quantiles, a small validation search, feature card.

Colab imports this module from ``notebooks/train_site.ipynb``. The search picks
tree size on validation macro MAE. Test rows are not used to choose the
configuration, the interval scale, or the features.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from src.metrics import (
    apply_interval,
    apply_interval_by_bin,
    calibrate_scale,
    coverage,
    macro_bin_mae,
    per_bin_report,
    within_tolerance,
)
from src.sample import assert_split_integrity
from src.schema import (
    BIN_LABELS,
    COVERAGE_TARGET,
    DEFAULT_DEPTH,
    DEFAULT_SEED,
    ENGINE_FEATURES,
    GAME_EXTRA,
    MACRO_BIN_MIN_N,
    MONOTONE,
    MONOTONE_V3,
    NUMERIC_FEATURES,
    PLAYER_BIN_COVERAGE_MIN,
    PLAYER_MACRO_MAE_MAX,
    PLAYER_TAIL_BIAS_MAX,
    PLAYER_WIDTH_MAX,
    TAIL_BIN_INDEXES,
)
from src.score import describe_feature

FROZEN_PARAMS = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "min_child_weight": 10,
    "reg_lambda": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "early_stopping_rounds": 30,
    "n_jobs": 1,
}

# rapid8-v3. The round cap is high enough that early stopping, not the cap, should decide.
FROZEN_PARAMS_V3 = {
    "n_estimators": 4000,
    "max_depth": 5,
    "learning_rate": 0.03,
    "min_child_weight": 10,
    "reg_lambda": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "early_stopping_rounds": 150,
    "n_jobs": -1,
}

# One axis moves at a time off the v3 frozen point.
SEARCH_GRID_V3: tuple[dict, ...] = (
    {"name": "frozen", "max_depth": 5, "min_child_weight": 10, "learning_rate": 0.03, "colsample_bytree": 0.8},
    {"name": "depth-4", "max_depth": 4, "min_child_weight": 10, "learning_rate": 0.03, "colsample_bytree": 0.8},
    {"name": "depth-6", "max_depth": 6, "min_child_weight": 10, "learning_rate": 0.03, "colsample_bytree": 0.8},
    {"name": "min-child-30", "max_depth": 5, "min_child_weight": 30, "learning_rate": 0.03, "colsample_bytree": 0.8},
    {"name": "lr-0.05", "max_depth": 5, "min_child_weight": 10, "learning_rate": 0.05, "colsample_bytree": 0.8},
    {"name": "colsample-0.5", "max_depth": 5, "min_child_weight": 10, "learning_rate": 0.03, "colsample_bytree": 0.5},
)

# One axis moves at a time off the frozen point. The notebook prints this list; it does not keep a second copy.
SEARCH_GRID: tuple[dict, ...] = (
    {"name": "frozen", "max_depth": 4, "min_child_weight": 10, "learning_rate": 0.05},
    {"name": "depth-3", "max_depth": 3, "min_child_weight": 10, "learning_rate": 0.05},
    {"name": "depth-5", "max_depth": 5, "min_child_weight": 10, "learning_rate": 0.05},
    {"name": "min-child-5", "max_depth": 4, "min_child_weight": 5, "learning_rate": 0.05},
    {"name": "min-child-20", "max_depth": 4, "min_child_weight": 20, "learning_rate": 0.05},
    {"name": "lr-0.1", "max_depth": 4, "min_child_weight": 10, "learning_rate": 0.1},
)


def gate_sample(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Require a player-disjoint game table with Stockfish columns filled in."""
    required = ("train", "val", "test")
    if "split" not in frame.columns:
        raise RuntimeError("games_sample.parquet has no split column")
    unknown = set(frame["split"].astype(str).unique()) - set(required)
    if unknown:
        raise RuntimeError(f"unexpected split labels: {sorted(unknown)}")
    counts = frame["split"].value_counts()
    empty = [name for name in required if int(counts.get(name, 0)) == 0]
    if empty:
        raise RuntimeError(f"split(s) {empty} are empty. The sample needs more players.")
    assert_split_integrity(frame)
    missing = [column for column in ENGINE_FEATURES if column not in frame.columns]
    if missing:
        raise RuntimeError(f"missing Stockfish columns: {missing}")
    nulls = [column for column in ENGINE_FEATURES if frame[column].isna().any()]
    if nulls:
        raise RuntimeError(f"Stockfish columns contain nulls: {nulls}")
    return {name: frame.loc[frame["split"] == name].reset_index(drop=True) for name in required}


def build_matrices(
    splits: dict[str, pd.DataFrame],
    with_first_move: bool = True,
    numeric: list[str] | None = None,
) -> dict:
    """One-hot the subject's first move on train only, then align every split.

    The eight-game model passes ``with_first_move=False``. Its rows are already
    averages, so a single opening is not a feature. ``numeric`` replaces the v2
    column list and is left with its NaNs intact.
    """
    numeric = [*GAME_EXTRA, *NUMERIC_FEATURES, *ENGINE_FEATURES] if numeric is None else list(numeric)
    if not with_first_move:
        matrices = {
            name: frame[numeric].reset_index(drop=True).astype(float)
            for name, frame in splits.items()
        }
        feature_names = list(matrices["train"].columns)
        return {
            "matrices": matrices,
            "encoder": None,
            "feature_names": feature_names,
            "numeric_features": numeric,
        }
    encoder = OneHotEncoder(handle_unknown="ignore", max_categories=30, sparse_output=False)
    encoder.fit(splits["train"][["first_move"]].fillna("missing").astype(str))

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        base = frame[numeric].reset_index(drop=True).astype(float)
        coded = pd.DataFrame(
            encoder.transform(frame[["first_move"]].fillna("missing").astype(str)),
            columns=list(encoder.get_feature_names_out(["first_move"])),
        )
        return pd.concat([base, coded], axis=1)

    matrices = {name: transform(frame) for name, frame in splits.items()}
    feature_names = list(matrices["train"].columns)
    for name, matrix in matrices.items():
        if list(matrix.columns) != feature_names:
            raise RuntimeError(f"{name} feature order drifted from the training matrix")
    return {
        "matrices": matrices,
        "encoder": encoder,
        "feature_names": feature_names,
        "numeric_features": numeric,
    }


def _macro_metric(bin_labels: np.ndarray):
    """Validation metric. XGBoost's sklearn wrapper calls ``f(y_true, y_pred) -> float``."""
    labels = np.asarray(bin_labels)

    def macro_bin_mae_eval(y_true, y_pred):
        if hasattr(y_true, "get_label"):
            actual = np.asarray(y_true.get_label(), dtype=float).reshape(-1)
            pred = np.asarray(y_pred, dtype=float).reshape(-1)
        else:
            actual = np.asarray(y_true, dtype=float).reshape(-1)
            pred = np.asarray(y_pred, dtype=float).reshape(-1)
        if len(pred) != len(labels):
            value = float(np.mean(np.abs(pred - actual)))
        else:
            value = macro_bin_mae(actual, pred, labels)
        return float(value)

    return macro_bin_mae_eval


def fit_quantile_trio(
    train_x: pd.DataFrame,
    y_train: np.ndarray,
    val_x: pd.DataFrame,
    y_val: np.ndarray,
    val_bins: np.ndarray,
    feature_names: list[str],
    params: dict | None = None,
    seed: int = DEFAULT_SEED,
    sample_weight: np.ndarray | None = None,
    monotone: dict | None = None,
) -> dict[str, XGBRegressor]:
    """Fit 10%, 50%, and 90%. The median early-stops on macro per-bin MAE.

    ``sample_weight`` applies to the training rows only. The validation set
    stays unweighted so the early-stopping score is one player, one vote.
    """
    params = dict(FROZEN_PARAMS if params is None else params)
    rounds = int(params.pop("early_stopping_rounds", 30))
    constraint_map = MONOTONE if monotone is None else monotone
    constraints = tuple(int(constraint_map.get(name, 0)) for name in feature_names)
    models: dict[str, XGBRegressor] = {}
    for name, alpha in (("q10", 0.1), ("q50", 0.5), ("q90", 0.9)):
        metric = _macro_metric(val_bins) if name == "q50" else None
        kwargs = dict(params)
        model_kwargs = {
            **kwargs,
            "objective": "reg:quantileerror",
            "quantile_alpha": alpha,
            "monotone_constraints": constraints,
            "early_stopping_rounds": rounds,
            "random_state": seed,
        }
        if metric is not None:
            model_kwargs["eval_metric"] = metric
        model = XGBRegressor(**model_kwargs)
        fit_kwargs: dict = {"eval_set": [(val_x, y_val)], "verbose": False}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        model.fit(train_x, y_train, **fit_kwargs)
        models[name] = model
        best = getattr(model, "best_iteration", None)
        score = getattr(model, "best_score", None)
        try:
            score_txt = f"{float(score):.3f}"
        except (TypeError, ValueError):
            score_txt = "n/a" if score is None else str(score)
        print(f"[fit] {name} alpha={alpha} best_iteration={best} best_score={score_txt}", flush=True)
    return models


def predict_quantiles(models: dict[str, XGBRegressor], matrix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from src.metrics import sort_quantiles

    return sort_quantiles(
        models["q10"].predict(matrix),
        models["q50"].predict(matrix),
        models["q90"].predict(matrix),
    )


def _is_frozen(spec: dict, base: dict | None = None) -> bool:
    reference = FROZEN_PARAMS if base is None else base
    for key, value in spec.items():
        if key == "name" or key not in reference:
            continue
        current = reference[key]
        if isinstance(current, float) or isinstance(value, float):
            if float(value) != float(current):
                return False
        elif int(value) != int(current):
            return False
    return True


def choose_search_row(rows: list[dict]) -> dict:
    """Lowest validation macro MAE. An exact tie keeps the frozen row, then the earlier row.

    Overall MAE, within-150, and coverage are not part of the key.
    """
    if not rows:
        raise ValueError("search rows are empty")

    def sort_key(item: tuple[int, dict]) -> tuple:
        index, row = item
        return (float(row["val_macro_mae"]), 0 if row.get("frozen") else 1, index)

    _index, chosen = min(enumerate(rows), key=sort_key)
    return chosen


def _best_iteration(model) -> int | None:
    iteration = getattr(model, "best_iteration", None)
    return None if iteration is None else int(iteration)


def search_validation(
    train_x: pd.DataFrame,
    y_train: np.ndarray,
    val_x: pd.DataFrame,
    y_val: np.ndarray,
    val_bins: np.ndarray,
    feature_names: list[str],
    seed: int = DEFAULT_SEED,
    grid: tuple | list | None = None,
    base: dict | None = None,
    sample_weight: np.ndarray | None = None,
    monotone: dict | None = None,
) -> dict:
    """Fit each pre-declared row on train. Choose on validation macro MAE.

    The returned models are the chosen row only. Test labels are not an argument.
    """
    y_val = np.asarray(y_val, dtype=float).reshape(-1)
    grid = SEARCH_GRID if grid is None else grid
    base = dict(FROZEN_PARAMS if base is None else base)
    rows: list[dict] = []
    fitted: dict[str, dict[str, XGBRegressor]] = {}
    for spec in grid:
        params = dict(base)
        for key, value in spec.items():
            if key != "name":
                params[key] = value
        print(
            f"[search] {spec['name']} depth={params['max_depth']} "
            f"min_child_weight={params['min_child_weight']} lr={params['learning_rate']}",
            flush=True,
        )
        models = fit_quantile_trio(
            train_x,
            y_train,
            val_x,
            y_val,
            val_bins,
            feature_names,
            params=params,
            seed=seed,
            sample_weight=sample_weight,
            monotone=monotone,
        )
        q10, q50, q90 = predict_quantiles(models, val_x)
        scale, covered = calibrate_scale(y_val, q10, q50, q90)
        row = {
            "name": str(spec["name"]),
            "max_depth": int(params["max_depth"]),
            "min_child_weight": int(params["min_child_weight"]),
            "learning_rate": float(params["learning_rate"]),
            "frozen": _is_frozen(spec, base),
            "val_macro_mae": float(macro_bin_mae(y_val, q50, val_bins)),
            "val_mae": float(np.mean(np.abs(np.asarray(q50, dtype=float) - y_val))),
            "val_within_150": float(within_tolerance(y_val, q50, 150.0)),
            "val_coverage": float(covered),
            "interval_scale": float(scale),
            "best_iteration_q10": _best_iteration(models["q10"]),
            "best_iteration_q50": _best_iteration(models["q50"]),
            "best_iteration_q90": _best_iteration(models["q90"]),
        }
        rows.append(row)
        fitted[row["name"]] = models
        print(
            f"[search] {row['name']} val_macro_mae={row['val_macro_mae']:.3f} "
            f"val_mae={row['val_mae']:.3f} val_within_150={row['val_within_150']:.3f} "
            f"val_coverage={row['val_coverage']:.3f}",
            flush=True,
        )
    chosen = choose_search_row(rows)
    models = fitted.pop(str(chosen["name"]))
    fitted.clear()
    print(
        f"[search] chose {chosen['name']} on validation macro MAE. "
        "within-150 did not choose the row.",
        flush=True,
    )
    return {"rows": rows, "chosen": chosen, "models": models}


def cap_binding(models: dict, n_estimators: int) -> list[str]:
    """Quantiles whose best round reached the estimator cap."""
    flagged = []
    limit = int(n_estimators) - 1
    for name in ("q10", "q50", "q90"):
        best = _best_iteration(models[name])
        if best is not None and best >= limit:
            flagged.append(name)
    return flagged


def ablation_validation(
    train: pd.DataFrame,
    y_train,
    val: pd.DataFrame,
    y_val,
    val_bins,
    feature_sets: dict[str, list[str]],
    sample_weight,
    params: dict | None = None,
) -> pd.DataFrame:
    """Fit the frozen v3 point on each growing feature set. Report only; it does not choose."""
    params = dict(FROZEN_PARAMS_V3 if params is None else params)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    y_val = np.asarray(y_val, dtype=float).reshape(-1)
    rows = []
    for name, features in feature_sets.items():
        columns = list(features)
        models = fit_quantile_trio(
            train[columns],
            y_train,
            val[columns],
            y_val,
            val_bins,
            columns,
            params=params,
            sample_weight=sample_weight,
            monotone=MONOTONE_V3,
        )
        _q10, q50, _q90 = predict_quantiles(models, val[columns])
        rows.append(
            {
                "set": name,
                "n_features": len(columns),
                "val_macro_mae": float(macro_bin_mae(y_val, q50, val_bins)),
                "val_within_150": float(within_tolerance(y_val, q50, 150.0)),
            }
        )
    return pd.DataFrame(rows)


def summarize_predictions_by_bin(
    y_true: np.ndarray,
    bin_labels: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    scales: dict,
) -> dict:
    """Same report as ``summarize_predictions``, with one scale per predicted bin."""
    low, mid, high = apply_interval_by_bin(q10, q50, q90, scales)
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    return {
        "coverage": coverage(y_true, low, high),
        "macro_mae": macro_bin_mae(y_true, mid, bin_labels),
        "mae": float(np.mean(np.abs(mid - y_true))),
        "bins": per_bin_report(y_true, mid, low, high, bin_labels),
    }


def summarize_predictions(
    y_true: np.ndarray,
    bin_labels: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    scale: float,
) -> dict:
    low, mid, high = apply_interval(q10, q50, q90, scale)
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    return {
        "coverage": coverage(y_true, low, high),
        "macro_mae": macro_bin_mae(y_true, mid, bin_labels),
        "mae": float(np.mean(np.abs(mid - y_true))),
        "bins": per_bin_report(y_true, mid, low, high, bin_labels),
    }


def ship_decision(summary: dict) -> dict:
    """Overall coverage near the target is ready. Wide bins stay labeled on the card."""
    wide = []
    for row in summary["bins"]:
        if int(row["n"]) < 20:
            continue
        if float(row["coverage"]) < 0.70 or float(row["median_width"]) >= 400:
            wide.append(str(row["bin"]))
    return {
        "ship_ready": bool(summary["coverage"] >= 0.75),
        "wide_bins": wide,
    }


def player_ship_decision(summary: dict) -> dict:
    """Ship bar for the eight-game model. The one-game coverage rule stays separate."""
    tails = {BIN_LABELS[index] for index in TAIL_BIN_INDEXES}
    wide = []
    biased = []
    for row in summary["bins"]:
        if int(row["n"]) < MACRO_BIN_MIN_N:
            continue
        label = str(row["bin"])
        if float(row["coverage"]) < PLAYER_BIN_COVERAGE_MIN or float(row["median_width"]) > PLAYER_WIDTH_MAX:
            wide.append(label)
        if label in tails and abs(float(row["bias"])) > PLAYER_TAIL_BIAS_MAX:
            biased.append(label)
    macro = float(summary["macro_mae"])
    return {
        "ship_ready": bool(macro <= PLAYER_MACRO_MAE_MAX and not wide and not biased),
        "wide_bins": wide,
        "tail_bias_bins": biased,
        "macro_mae_max": PLAYER_MACRO_MAE_MAX,
        "tail_bias_max": PLAYER_TAIL_BIAS_MAX,
        "width_max": PLAYER_WIDTH_MAX,
        "bin_coverage_min": PLAYER_BIN_COVERAGE_MIN,
    }


def engine_linear_macro(train: pd.DataFrame, val: pd.DataFrame) -> float:
    """Median baseline: linear regression on the Stockfish block, scored by macro bin MAE."""
    model = LinearRegression()
    model.fit(train[list(ENGINE_FEATURES)].to_numpy(dtype=float), train["elo"].to_numpy(dtype=float))
    pred = model.predict(val[list(ENGINE_FEATURES)].to_numpy(dtype=float))
    return macro_bin_mae(val["elo"].to_numpy(dtype=float), pred, val["elo_bin"].to_numpy())


def decile_curve(values: pd.Series, elo: pd.Series) -> list[dict]:
    """Mean actual Elo inside up to ten quantile bins of one feature."""
    clean = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce"), "elo": pd.to_numeric(elo, errors="coerce")})
    clean = clean.dropna()
    if clean["value"].nunique() < 3 or len(clean) < 10:
        return []
    clean = clean.copy()
    clean["decile"] = pd.qcut(clean["value"], 10, duplicates="drop")
    grouped = (
        clean.groupby("decile", observed=True)
        .agg(value_mean=("value", "mean"), elo_mean=("elo", "mean"), n=("elo", "size"))
        .reset_index(drop=True)
    )
    return [
        {
            "value_mean": float(row.value_mean),
            "elo_mean": float(row.elo_mean),
            "n": int(row.n),
        }
        for row in grouped.itertuples(index=False)
    ]


def feature_report(
    model: XGBRegressor,
    matrix: pd.DataFrame,
    elo: np.ndarray,
    n_repeats: int = 5,
    seed: int = DEFAULT_SEED,
    max_rows: int = 4000,
    top_k: int = 5,
) -> tuple[pd.DataFrame, list[dict]]:
    """Permutation importance on validation, plus an Elo curve for the top features."""
    from sklearn.inspection import permutation_importance

    elo = np.asarray(elo, dtype=float).reshape(-1)
    sample = matrix
    sample_elo = elo
    if len(matrix) > max_rows:
        rng = np.random.default_rng(seed)
        take = np.sort(rng.choice(len(matrix), size=max_rows, replace=False))
        sample = matrix.iloc[take]
        sample_elo = elo[take]
    result = permutation_importance(
        model,
        sample,
        sample_elo,
        scoring="neg_mean_absolute_error",
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=1,
    )
    table = (
        pd.DataFrame(
            {
                "feature": list(sample.columns),
                "importance": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    table["description"] = table["feature"].map(describe_feature)
    top = []
    for row in table.head(top_k).itertuples(index=False):
        top.append(
            {
                "feature": str(row.feature),
                "importance": float(row.importance),
                "importance_std": float(row.importance_std),
                "description": describe_feature(str(row.feature)),
                "curve": decile_curve(matrix[row.feature], pd.Series(elo)),
            }
        )
    return table, top


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and value != value:
        return None
    return value


def write_bundle(
    directory: Path,
    models: dict[str, XGBRegressor],
    encoder,
    card: dict,
    importance: pd.DataFrame | None = None,
) -> Path:
    import joblib

    directory.mkdir(parents=True, exist_ok=True)
    filenames = {"q10": "model_q10.json", "q50": "model_q50.json", "q90": "model_q90.json"}
    for name, filename in filenames.items():
        models[name].save_model(directory / filename)
    joblib.dump(encoder, directory / "encoder.joblib")
    if importance is not None:
        importance.to_csv(directory / "permutation_importance.csv", index=False)
        card["importance_csv"] = "permutation_importance.csv"
    card["models"] = filenames
    card["encoder"] = "encoder.joblib"
    card["depth"] = DEFAULT_DEPTH
    card["coverage_target"] = COVERAGE_TARGET
    path = directory / "model_card.json"
    path.write_text(json.dumps(_jsonable(card), indent=2), encoding="utf-8")
    print(f"[fit] wrote {path}", flush=True)
    return path
