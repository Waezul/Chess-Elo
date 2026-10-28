"""One row per mainline move, plus the clock the PGN recorded after it.

Centipawns stay in white's point of view. ``analyze_game`` walks this same
table, so the live page and the v3 sample cannot drift apart.
"""

from __future__ import annotations

import chess
import chess.engine
import chess.pgn

from src.engine import AnalyseFn

MOVE_COLUMNS = ["game_id", "ply", "color", "uci", "best_uci", "cp_before", "cp_after", "clock_after"]


def parse_time_control(text: str | None) -> tuple[int, int] | None:
    """``600+5`` is 600 seconds plus 5. ``600`` has no increment. Anything else is missing."""
    if text is None:
        return None
    raw = str(text).strip()
    if not raw or raw == "-":
        return None
    parts = raw.split("+")
    if len(parts) == 1 and parts[0].isdigit():
        return int(parts[0]), 0
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return int(parts[0]), int(parts[1])
    return None


def _clock_seconds(node: chess.pgn.GameNode) -> float | None:
    clock = node.clock()
    if clock is None:
        return None
    if hasattr(clock, "total_seconds"):
        return float(clock.total_seconds())
    return float(clock)


def analyze_moves(game: chess.pgn.Game, analyse: AnalyseFn, game_id: str) -> list[dict]:
    """One dict per mainline move. Empty on ``EngineError``. ``EngineTerminatedError`` is re-raised."""
    board = game.board()
    rows: list[dict] = []
    try:
        for node in game.mainline():
            move = node.move
            if move is None:
                continue
            ply = board.ply() + 1
            color = "white" if board.turn == chess.WHITE else "black"
            cp_before, best_uci = analyse(board)
            uci = move.uci()
            board.push(move)
            cp_after, _after_best = analyse(board)
            rows.append(
                {
                    "game_id": game_id,
                    "ply": int(ply),
                    "color": color,
                    "uci": uci,
                    "best_uci": best_uci,
                    "cp_before": int(cp_before),
                    "cp_after": int(cp_after),
                    "clock_after": _clock_seconds(node),
                }
            )
    except chess.engine.EngineTerminatedError:
        raise
    except chess.engine.EngineError:
        return []
    return rows
