"""v3 features for one side of one game, from the shared move table.

``extract_game`` is the function the live API and the training pass both call.
A missing phase or a missing clock is NaN, never zero.
"""

from __future__ import annotations

import math

import chess.engine
import pandas as pd

from src.engine import move_accuracy, read_game, summarize_moves, win_percent
from src.features import rows_from_movetext
from src.moves import MOVE_COLUMNS, analyze_moves, parse_time_control
from src.schema import (
    BALANCED_CP,
    BLUNDER_CP,
    CALM_CLOCK_FRAC,
    INSTANT_SECONDS,
    LATE_PLY,
    LOW_CLOCK_FRAC,
    MISTAKE_CP,
    OPENING_PLY,
    V3_CLOCK,
    V3_GAME_ENGINE,
    V3_OPPONENT,
    V3_TC,
)

__all__ = ["extract_game", "move_accuracy", "side_features", "win_percent"]


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _loss(cp_before: float, cp_after: float, color: str) -> float:
    if color == "white":
        return max(0.0, float(cp_before) - float(cp_after))
    return max(0.0, float(cp_after) - float(cp_before))


def _mover(cp_white: float, color: str) -> float:
    return float(cp_white) if color == "white" else -float(cp_white)


def _quality(side: pd.DataFrame) -> dict[str, float]:
    losses: list[float] = []
    wpls: list[float] = []
    accs: list[float] = []
    blunders: list[float] = []
    balanced: list[float] = []
    balanced_mistakes: list[float] = []
    opening: list[float] = []
    late: list[float] = []
    for row in side.itertuples(index=False):
        color = str(row.color)
        before = _mover(row.cp_before, color)
        after = _mover(row.cp_after, color)
        loss = _loss(row.cp_before, row.cp_after, color)
        wpl = max(0.0, win_percent(before) - win_percent(after))
        losses.append(loss)
        wpls.append(wpl)
        accs.append(move_accuracy(wpl))
        blunders.append(1.0 if loss >= BLUNDER_CP else 0.0)
        if abs(before) <= BALANCED_CP:
            balanced.append(loss)
            balanced_mistakes.append(1.0 if loss >= MISTAKE_CP else 0.0)
        ply = int(row.ply)
        if ply <= OPENING_PLY:
            opening.append(loss)
        if ply > LATE_PLY:
            late.append(loss)
    return {
        "cpl_mean": _mean(losses),
        "wpl_mean": _mean(wpls),
        "acc_mean": _mean(accs),
        "blunder_rate": _mean(blunders),
        "cpl_balanced": _mean(balanced),
        "mistake_rate_balanced": _mean(balanced_mistakes),
        "cpl_opening": _mean(opening),
        "cpl_late": _mean(late),
    }


def _clock_block(side: pd.DataFrame, parsed: tuple[int, int] | None) -> dict[str, float]:
    blank = {key: float("nan") for key in (*V3_CLOCK[1:], *V3_TC)}
    blank["has_clock"] = 0.0
    if parsed is None:
        return blank
    base, increment = parsed
    timed = {
        "has_clock": 0.0,
        **{key: float("nan") for key in V3_CLOCK[1:]},
        "tc_base": float(base),
        "tc_increment": float(increment),
    }
    if side.empty or bool(side["clock_after"].isna().any()):
        return timed
    est = float(base) + 40.0 * float(increment)
    if est <= 0:
        return timed
    thinks: list[float] = []
    low: list[float] = []
    calm: list[float] = []
    instant: list[float] = []
    previous: float | None = None
    last_clock: float | None = None
    for row in side.sort_values("ply").itertuples(index=False):
        clock_after = float(row.clock_after)
        clock_before = float(base) if previous is None else previous
        think = max(0.0, clock_before - clock_after + float(increment))
        if previous is not None:
            thinks.append(think)
        if int(row.ply) > OPENING_PLY:
            instant.append(1.0 if think < INSTANT_SECONDS else 0.0)
        low.append(1.0 if clock_after < LOW_CLOCK_FRAC * est else 0.0)
        if clock_before >= CALM_CLOCK_FRAC * est:
            calm.append(_loss(row.cp_before, row.cp_after, str(row.color)))
        previous = clock_after
        last_clock = clock_after
    think_mean = _mean(thinks)
    if not thinks or not math.isfinite(think_mean) or think_mean == 0.0:
        cv = float("nan")
    else:
        variance = sum((value - think_mean) ** 2 for value in thinks) / len(thinks)
        cv = math.sqrt(variance) / think_mean
    return {
        "has_clock": 1.0,
        "clk_think_frac_mean": float("nan") if not thinks else think_mean / est,
        "clk_think_cv": cv,
        "clk_instant_rate": _mean(instant),
        "clk_low_rate": _mean(low),
        "clk_end_frac": float("nan") if last_clock is None else last_clock / est,
        "cpl_calm": _mean(calm),
        "tc_base": float(base),
        "tc_increment": float(increment),
    }


def side_features(moves: pd.DataFrame, color: str, time_control: str | None) -> dict[str, float]:
    """Quality, opponent, clock, and time-control columns for ``color``."""
    subject = moves.loc[moves["color"] == color]
    opponent = moves.loc[moves["color"] != color]
    own = _quality(subject)
    opp = _quality(opponent)
    clocks = _clock_block(subject, parse_time_control(time_control))
    features = {
        "wpl_mean": own["wpl_mean"],
        "acc_mean": own["acc_mean"],
        "cpl_balanced": own["cpl_balanced"],
        "mistake_rate_balanced": own["mistake_rate_balanced"],
        "cpl_opening": own["cpl_opening"],
        "cpl_late": own["cpl_late"],
        "opp_cpl_mean": opp["cpl_mean"],
        "opp_wpl_mean": opp["wpl_mean"],
        "opp_acc_mean": opp["acc_mean"],
        "opp_blunder_rate": opp["blunder_rate"],
        "opp_cpl_balanced": opp["cpl_balanced"],
        **clocks,
    }
    expected = [*V3_GAME_ENGINE, *V3_OPPONENT, *V3_CLOCK, *V3_TC]
    return {key: float(features[key]) for key in expected}


def _engine_by_color(moves: list[dict]) -> dict[str, dict[str, float]]:
    buckets: dict[str, dict[str, list]] = {}
    for row in moves:
        color = str(row["color"])
        bucket = buckets.setdefault(color, {"loss": [], "eval": [], "best": []})
        bucket["loss"].append(_loss(row["cp_before"], row["cp_after"], color))
        bucket["eval"].append(_mover(row["cp_before"], color))
        bucket["best"].append(row["best_uci"] is not None and row["uci"] == row["best_uci"])
    return {
        color: summarize_moves(bucket["loss"], bucket["eval"], bucket["best"])
        for color, bucket in buckets.items()
        if bucket["loss"]
    }


def extract_game(
    headers: dict[str, str],
    move_lines: list[str],
    analyse,
) -> tuple[list[dict], list[dict]]:
    """Behavior row, v2 engine columns, and v3 columns for each side that moved."""
    game = read_game(headers, move_lines)
    rows = rows_from_movetext(headers, move_lines)
    if game is None or not rows:
        return [], []
    game_id = str(rows[0].get("game_id") or "")
    try:
        moves = analyze_moves(game, analyse, game_id)
    except chess.engine.EngineTerminatedError:
        raise
    except chess.engine.EngineError:
        return [], []
    if not moves:
        return [], []
    by_color = _engine_by_color(moves)
    table = pd.DataFrame(moves, columns=MOVE_COLUMNS)
    time_control = headers.get("TimeControl")
    enriched: list[dict] = []
    for row in rows:
        color = str(row["color"])
        stats = by_color.get(color)
        if not stats:
            continue
        updated = dict(row)
        updated.update(stats)
        updated.update(side_features(table, color, time_control))
        updated["time_control"] = time_control
        enriched.append(updated)
    if not enriched:
        return [], []
    return enriched, moves
