"""Turn eight accepted games into one estimate. The caller supplies the engine."""

from __future__ import annotations

import io
import threading

import chess.engine
import pandas as pd

from src.catalog import Catalog
from src.checklist import check_pgn, estimated_seconds, subject_color
from src.engine import EngineSession, engine_pass, move_accuracy, win_percent
from src.features import rows_from_movetext
from src.player_table import average_games
from src.pgn_stream import iter_records, parse_headers
from src.schema import BLUNDER_CP, MISTAKE_CP

_OPENING_PLY = 20
_LATE_PLY = 60
_LOW_CLOCK_FRAC = 0.10


class SelectionError(Exception):
    def __init__(self, error: str) -> None:
        self.error = error
        super().__init__(error)


class EngineDownError(Exception):
    pass


def _accuracy(record: dict) -> float:
    lost = max(0.0, win_percent(record["cp_before_mover"]) - win_percent(record["cp_after_mover"]))
    return move_accuracy(lost)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _clock_rates(moves: list[dict]) -> tuple[float | None, float | None]:
    """Null unless every subject move has a clock and a parseable time control."""
    if not moves or any(move["clock_after"] is None or move["est"] is None for move in moves):
        return None, None
    short = [move["clock_after"] < _LOW_CLOCK_FRAC * move["est"] for move in moves]
    low_rate = sum(short) / len(short)
    blunders = [is_short for move, is_short in zip(moves, short) if move["cp_loss"] >= BLUNDER_CP]
    if not blunders:
        return low_rate, None
    return low_rate, sum(blunders) / len(blunders)


def _castling(rows: list[dict]) -> dict:
    kingside = queenside = uncastled = 0
    plies = []
    for row in rows:
        if row.get("castled"):
            if row.get("castle_kingside") == 1 or row.get("castle_kingside") == 1.0:
                kingside += 1
            else:
                queenside += 1
            ply = row.get("castle_ply")
            if ply is not None and ply == ply:
                plies.append(float(ply))
        else:
            uncastled += 1
    return {
        "castle_kingside": kingside,
        "castle_queenside": queenside,
        "uncastled": uncastled,
        "castle_ply_mean": _mean(plies),
    }


def _dashboard(rows: list[dict], subject_moves: list[dict], opponent_moves: list[dict]) -> tuple[dict, list[dict]]:
    opening = [move["cp_loss"] for move in subject_moves if move["ply"] <= _OPENING_PLY]
    late = [move["cp_loss"] for move in subject_moves if move["ply"] > _LATE_PLY]
    low_rate, trouble = _clock_rates(subject_moves)
    stats = {
        "acc": _mean([_accuracy(move) for move in subject_moves]),
        "blunder_rate": _mean([1.0 if move["cp_loss"] >= BLUNDER_CP else 0.0 for move in subject_moves]),
        "mistake_rate": _mean([1.0 if move["cp_loss"] >= MISTAKE_CP else 0.0 for move in subject_moves]),
        "best_move_rate": _mean([1.0 if move["best_flag"] else 0.0 for move in subject_moves]),
        "cpl_opening": _mean(opening),
        "cpl_late": _mean(late),
        "n_games": len(rows),
        "opp_acc": _mean([_accuracy(move) for move in opponent_moves]),
        "clk_low_rate": low_rate,
        "blunder_in_time_trouble_rate": trouble,
        **_castling(rows),
    }
    worst = sorted(subject_moves, key=lambda move: move["cp_loss"], reverse=True)[:3]
    moments = [
        {
            "game_index": move["game_index"],
            "ply": move["ply"],
            "played": move["played"],
            "best": move["best"],
            "cp_loss": move["cp_loss"],
            "blunder": move["cp_loss"] >= BLUNDER_CP,
        }
        for move in worst
    ]
    return stats, moments


def score_selection(catalog: Catalog, pgn: str, subject: str, pool: str, indexes: list[int], analyse) -> dict:
    """Average the eight games, then score that one row. Ratings stay out of the result."""
    required = int(catalog.bundles[pool]["card"]["games_per_player"])
    if len(indexes) != required:
        raise SelectionError("wrong_count")
    checked = {game["index"]: game for game in check_pgn(pgn, subject, pool)}
    records = list(iter_records(io.StringIO(pgn)))
    rows = []
    subject_moves = []
    opponent_moves = []
    for index in indexes:
        game = checked.get(index)
        if game is None or not game["accepted"]:
            raise SelectionError("bad_indexes")
        headers = parse_headers(records[index][0])
        move_lines = records[index][1]
        color = subject_color(headers, subject)
        extracted = rows_from_movetext(headers, move_lines)
        extracted, moves = engine_pass(headers, move_lines, extracted, analyse)
        picked = [
            row
            for row in extracted
            if row.get("color") == color and row.get("cpl_mean") is not None
        ]
        if len(picked) != 1:
            raise SelectionError("unreadable")
        rows.append(picked[0])
        est = estimated_seconds(headers.get("TimeControl"))
        for move in moves:
            tagged = {**move, "game_index": index, "est": est}
            if move["color"] == color:
                subject_moves.append(tagged)
            else:
                opponent_moves.append(tagged)
    averaged = average_games(pd.DataFrame(rows))
    if len(averaged) != 1:
        raise SelectionError("unreadable")
    quantiles = catalog.score_frame(pool, averaged)
    for key in ("q10", "q50", "q90"):
        if quantiles[key] != quantiles[key]:
            raise SelectionError("unreadable")
    card = catalog.bundles[pool]["card"]
    meta_revision = catalog.summary(pool)["revision"]
    stats, moments = _dashboard(rows, subject_moves, opponent_moves)
    return {
        "pool": pool,
        "revision": meta_revision,
        "ship_ready": bool(card["ship_ready"]),
        "lab": not bool(card["ship_ready"]),
        "n_games": len(indexes),
        "depth": int(card["depth"]),
        "q10": quantiles["q10"],
        "q50": quantiles["q50"],
        "q90": quantiles["q90"],
        "drivers": quantiles["drivers"],
        "stats": stats,
        "moments": moments,
        "limits": catalog.limits(pool, quantiles["q50"]),
    }


def stockfish_estimator(catalog: Catalog, binary: str, hash_mb: int = 128):
    """One Stockfish process per worker thread, opened on the first job."""
    local = threading.local()

    def estimate(pool: str, pgn: str, subject: str, indexes: list[int]) -> dict:
        depth = int(catalog.bundles[pool]["card"]["depth"])
        session = getattr(local, "session", None)
        if session is None or getattr(local, "depth", None) != depth:
            if session is not None:
                session.close()
            try:
                session = EngineSession(binary, depth, 1, hash_mb, 50_000)
            except (OSError, chess.engine.EngineError) as exc:
                raise EngineDownError from exc
            local.session = session
            local.depth = depth
        try:
            return score_selection(catalog, pgn, subject, pool, indexes, session.analyse)
        except chess.engine.EngineTerminatedError as exc:
            local.session = None
            raise EngineDownError from exc

    return estimate
