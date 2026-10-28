"""Per-bin error and 10-90% interval coverage for the site model.

Early stopping uses the macro average of per-bin MAE so a crowded middle bin
cannot hide the tails. The interval scale is chosen on validation only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.schema import BIN_LABELS, COVERAGE_TARGET, ELO_MAX, ELO_MIN, MACRO_BIN_MIN_N, TAIL_BIN_INDEXES


def bin_index(elo: float) -> int | None:
    """Right-closed 200-Elo bin. 800 and 1000 land in 800-1000; 2400 lands in 2200-2400."""
    try:
        value = int(elo)
    except (TypeError, ValueError):
        return None
    if value < ELO_MIN or value > ELO_MAX:
        return None
    if value == ELO_MIN:
        return 0
    index = (value - 1 - ELO_MIN) // 200
    if index < 0 or index >= len(BIN_LABELS):
        return None
    return int(index)


def bin_label(elo: float) -> str | None:
    index = bin_index(elo)
    if index is None:
        return None
    return BIN_LABELS[index]


def is_tail(index: int) -> bool:
    return index in TAIL_BIN_INDEXES


def sort_quantiles(
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Force q10 <= q50 <= q90 on each row. Crossed trees happen on small leaves."""
    stacked = np.sort(
        np.column_stack(
            [
                np.asarray(q10, dtype=float).reshape(-1),
                np.asarray(q50, dtype=float).reshape(-1),
                np.asarray(q90, dtype=float).reshape(-1),
            ]
        ),
        axis=1,
    )
    return stacked[:, 0], stacked[:, 1], stacked[:, 2]


def apply_interval(
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Widen or tighten the two sides of the median by ``scale``. 1 leaves the quantiles as fit."""
    low, mid, high = sort_quantiles(q10, q50, q90)
    lo = mid - scale * (mid - low)
    hi = mid + scale * (high - mid)
    return lo, mid, hi


def within_tolerance(y_true: np.ndarray, y_pred: np.ndarray, tolerance: float = 150.0) -> float:
    """Share of rows whose absolute error is at most ``tolerance`` Elo.

    Reported beside macro MAE. It does not choose the search row or the interval scale.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(np.abs(y_pred - y_true) <= float(tolerance)))


def coverage(y_true: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    if len(y_true) == 0:
        return float("nan")
    inside = (y_true >= np.asarray(low, dtype=float).reshape(-1)) & (
        y_true <= np.asarray(high, dtype=float).reshape(-1)
    )
    return float(np.mean(inside))


def calibrate_scale(
    y_true: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    target: float = COVERAGE_TARGET,
) -> tuple[float, float]:
    """Pick the scale whose validation coverage is closest to ``target``.

    Ties keep the tighter interval. The search grid is fixed; this is not a
    model refit.
    """
    scales = np.round(np.arange(0.25, 4.01, 0.05), 2)
    best: tuple[float, float, float] | None = None
    for scale in scales:
        low, _, high = apply_interval(q10, q50, q90, float(scale))
        covered = coverage(y_true, low, high)
        gap = abs(covered - target)
        if best is None or gap < best[0] - 1e-12 or (abs(gap - best[0]) <= 1e-12 and scale < best[1]):
            best = (gap, float(scale), covered)
    assert best is not None
    return best[1], best[2]


def calibrate_scales_by_bin(
    y_true: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    target: float = COVERAGE_TARGET,
    min_n: int = 50,
) -> dict[str, float]:
    """One interval scale per predicted rating bin. Small bins use the overall scale."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    q10 = np.asarray(q10, dtype=float).reshape(-1)
    q50 = np.asarray(q50, dtype=float).reshape(-1)
    q90 = np.asarray(q90, dtype=float).reshape(-1)
    labels = np.array([bin_label(float(np.clip(value, 801, 2400))) for value in q50])
    all_scale, _covered = calibrate_scale(y_true, q10, q50, q90, target)
    scales = {label: float(all_scale) for label in BIN_LABELS}
    scales["all"] = float(all_scale)
    for label in BIN_LABELS:
        mask = labels == label
        if int(mask.sum()) < int(min_n):
            continue
        scale, _bin_covered = calibrate_scale(y_true[mask], q10[mask], q50[mask], q90[mask], target)
        scales[label] = float(scale)
    return scales


def apply_interval_by_bin(
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    scales: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Widen each row with the scale of its median's rating bin."""
    low, mid, high = sort_quantiles(q10, q50, q90)
    fallback = float(scales["all"])
    chosen = np.empty(len(mid), dtype=float)
    for index, value in enumerate(mid):
        label = bin_label(float(np.clip(value, 801, 2400)))
        chosen[index] = float(scales[label]) if label in scales else fallback
    lo = mid - chosen * (mid - low)
    hi = mid + chosen * (high - mid)
    return lo, mid, hi


def _bin_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bin_labels: np.ndarray,
) -> list[tuple[str, int, float]]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    labels = np.asarray(bin_labels).reshape(-1)
    rows: list[tuple[str, int, float]] = []
    for label in pd.unique(labels):
        mask = labels == label
        count = int(mask.sum())
        if count == 0:
            continue
        mae = float(np.mean(np.abs(y_pred[mask] - y_true[mask])))
        rows.append((str(label), count, mae))
    return rows


def macro_bin_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bin_labels: np.ndarray,
    min_n: int = MACRO_BIN_MIN_N,
) -> float:
    """Unweighted mean of per-bin MAE. Bins under ``min_n`` are skipped when any bin qualifies."""
    rows = _bin_rows(y_true, y_pred, bin_labels)
    usable = [mae for _, count, mae in rows if count >= min_n]
    if not usable:
        usable = [mae for _, _, mae in rows]
    if not usable:
        return float("nan")
    return float(np.mean(usable))


def per_bin_report(
    y_true: np.ndarray,
    mid: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    bin_labels: np.ndarray,
) -> list[dict]:
    """One record per rating bin that appears in ``bin_labels``, in label order."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    mid = np.asarray(mid, dtype=float).reshape(-1)
    low = np.asarray(low, dtype=float).reshape(-1)
    high = np.asarray(high, dtype=float).reshape(-1)
    labels = np.asarray(bin_labels).reshape(-1)
    order = {label: index for index, label in enumerate(BIN_LABELS)}
    records = []
    for label in sorted(pd.unique(labels), key=lambda item: order.get(str(item), 99)):
        mask = labels == label
        count = int(mask.sum())
        if count == 0:
            continue
        residual = mid[mask] - y_true[mask]
        width = high[mask] - low[mask]
        records.append(
            {
                "bin": str(label),
                "n": count,
                "mae": float(np.mean(np.abs(residual))),
                "bias": float(np.mean(residual)),
                "coverage": coverage(y_true[mask], low[mask], high[mask]),
                "median_width": float(np.median(width)),
            }
        )
    return records
