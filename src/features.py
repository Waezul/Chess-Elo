"""Subject-only behavior features for one rated game.

Each accepted game produces up to two rows, one per side that moved.
Counts use only that side's plies. Material is sampled on those plies in
the opening and in the middlegame, excluding the last six plies so the
final material tally does not stand in for the result.
"""

from __future__ import annotations

import hashlib
import io
from typing import Any

import chess
import chess.pgn

from src.schema import ELO_MAX, ELO_MIN, POOLS, TERMINATION

_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}
_PIECE_RATE = {
    chess.PAWN: "rate_pawn",
    chess.KNIGHT: "rate_knight",
    chess.BISHOP: "rate_bishop",
    chess.ROOK: "rate_rook",
    chess.QUEEN: "rate_queen",
    chess.KING: "rate_king",
}
_START_MINORS = {
    chess.WHITE: (chess.B1, chess.G1, chess.C1, chess.F1),
    chess.BLACK: (chess.B8, chess.G8, chess.C8, chess.F8),
}
_STANDARD_START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def player_key(name: str) -> str:
    """Stable id that is not the username."""
    return hashlib.sha256(name.strip().lower().encode("utf-8")).hexdigest()[:16]


def parse_elo(raw: str | None) -> int | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text.isdigit():
        return None
    return int(text)


def game_id_from_site(site: str) -> str:
    site = (site or "").strip().rstrip("/")
    if not site:
        return ""
    return site.split("/")[-1]


def accept_headers(
    headers: dict[str, str],
    pool: str = "classical",
    elo_min: int = ELO_MIN,
    elo_max: int = ELO_MAX,
) -> bool:
    """One rating pool, normal finish, both Elos inside the band.

    Provisional ratings (a trailing ``?``) are rejected. Correspondence
    and every other time-control pool are rejected so clock mix cannot
    stand in for skill.
    """
    if pool not in POOLS:
        raise ValueError(f"pool must be one of {POOLS}")
    event = headers.get("Event", "").lower()
    if "rated" not in event or pool not in event:
        return False
    if "correspondence" in event:
        return False
    if headers.get("Termination") != TERMINATION:
        return False
    if headers.get("TimeControl", "-").strip() in {"-", ""}:
        return False
    if not headers.get("White", "").strip() or not headers.get("Black", "").strip():
        return False
    variant = headers.get("Variant", "Standard")
    if variant.lower() not in {"standard", "chess"}:
        return False
    white = parse_elo(headers.get("WhiteElo"))
    black = parse_elo(headers.get("BlackElo"))
    if white is None or black is None:
        return False
    return elo_min <= white <= elo_max and elo_min <= black <= elo_max


def _result_subject(result: str | None, color: str) -> float | None:
    if result == "1-0":
        return 1.0 if color == "white" else 0.0
    if result == "0-1":
        return 0.0 if color == "white" else 1.0
    if result == "1/2-1/2":
        return 0.5
    return None


def _eco_family(eco: str | None) -> str | None:
    if not eco:
        return None
    token = eco.strip().upper()
    if len(token) < 2:
        return token or None
    return token[:2]


def _developed_minors(board: chess.Board, color: chess.Color) -> int:
    start = set(_START_MINORS[color])
    count = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if (
            piece
            and piece.color == color
            and piece.piece_type in (chess.KNIGHT, chess.BISHOP)
            and square not in start
        ):
            count += 1
    return count


def _material_pov(board: chess.Board, color: chess.Color) -> int:
    own = 0
    opp = 0
    for piece in board.piece_map().values():
        value = _VALUES.get(piece.piece_type, 0)
        if piece.color == color:
            own += value
        else:
            opp += value
    return own - opp


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _row_for_side(
    color_name: str,
    color: chess.Color,
    acc: dict[str, Any],
    final_ply: int,
    developed_at_10: dict[chess.Color, int | None],
    final_developed: dict[chess.Color, int],
    headers: dict[str, str],
    game_id: str,
) -> dict[str, Any] | None:
    moves = acc["moves"]
    if moves <= 0:
        return None
    rates = {name: 0.0 for name in _PIECE_RATE.values()}
    for piece_type, count in acc["piece_counts"].items():
        rates[_PIECE_RATE[piece_type]] = count / moves  # piece-type share of this side's moves
    # Opening is the first 20 plies. Middlegame drops the last 6 so the result does not leak in.
    opening = [mat for ply, mat in acc["material"] if ply <= 20]
    middlegame = [mat for ply, mat in acc["material"] if 20 < ply <= final_ply - 6]
    developed = developed_at_10[color]
    if developed is None:
        developed = final_developed[color]
    elo_header = "WhiteElo" if color == chess.WHITE else "BlackElo"
    name_header = "White" if color == chess.WHITE else "Black"
    first = acc["first_move"]
    return {
        "game_id": game_id,  # join key only; not a model input
        "player_key": player_key(headers[name_header]),  # hashed name, for grouping players
        "color": color_name,  # which side these counts belong to
        "elo": parse_elo(headers.get(elo_header)),  # regression target for this side
        "is_white": 1 if color == chess.WHITE else 0,  # color flag for the game-level model
        "result_subject": _result_subject(headers.get("Result"), color_name),  # held out of the default model
        "eco_family": _eco_family(headers.get("ECO")),  # coarse opening family, not the full ECO code
        "first_move": first,  # this side's first move, for repertoire entropy
        "game_plies": final_ply,  # game length
        "n_subject_moves": moves,  # moves by this side
        "rate_pawn": rates["rate_pawn"],
        "rate_knight": rates["rate_knight"],
        "rate_bishop": rates["rate_bishop"],
        "rate_rook": rates["rate_rook"],
        "rate_queen": rates["rate_queen"],
        "rate_king": rates["rate_king"],
        "rate_capture": acc["captures"] / moves,  # capture frequency for this side
        "rate_check": acc["checks"] / moves,  # check frequency for this side
        "rate_promotion": acc["promotions"] / moves,  # promotion frequency for this side
        "castled": 1.0 if acc["castled"] else 0.0,  # did this side castle
        "castle_kingside": acc["castle_kingside"],  # which side they castled
        "castle_ply": acc["castle_ply"],  # when they castled
        "mat_opening_mean": _mean(opening),  # material balance in the opening
        "mat_middlegame_mean": _mean(middlegame),  # material balance before the ending
        "developed_minors_ply10": developed,  # minor-piece development by ply 10
        "avg_legal_moves": acc["legal_sum"] / moves,  # how wide their positions were
    }


def _blank_acc() -> dict[str, Any]:
    return {
        "moves": 0,
        "piece_counts": {piece: 0 for piece in _PIECE_RATE},
        "captures": 0,
        "checks": 0,
        "promotions": 0,
        "castled": False,
        "castle_kingside": None,
        "castle_ply": None,
        "legal_sum": 0,
        "material": [],
        "first_move": None,
    }


def rows_from_game(
    game: chess.pgn.Game,
    headers: dict[str, str],
    game_id: str | None = None,
) -> list[dict[str, Any]]:
    board = game.board()
    if not board.fen().startswith(_STANDARD_START):
        return []
    if not game_id:
        game_id = game_id_from_site(headers.get("Site", ""))
    acc = {chess.WHITE: _blank_acc(), chess.BLACK: _blank_acc()}
    developed_at_10: dict[chess.Color, int | None] = {chess.WHITE: None, chess.BLACK: None}
    for move in game.mainline_moves():
        color = board.turn
        side = acc[color]
        piece = board.piece_at(move.from_square)
        if piece is None:
            return []
        side["legal_sum"] += board.legal_moves.count()  # choices available before the move
        if side["first_move"] is None:
            # Strip check markers so Nf6 and Nf6+ count as the same repertoire move.
            san = board.san(move).replace("+", "").replace("#", "")
            side["first_move"] = san
        side["piece_counts"][piece.piece_type] += 1  # this side only; the opponent is a separate row
        if board.is_capture(move):
            side["captures"] += 1
        if board.gives_check(move):
            side["checks"] += 1
        if move.promotion:
            side["promotions"] += 1
        if board.is_castling(move):
            side["castled"] = True
            side["castle_kingside"] = 1.0 if board.is_kingside_castling(move) else 0.0
            side["castle_ply"] = board.ply() + 1  # ply number of the castling move
        board.push(move)
        side["moves"] += 1
        side["material"].append((board.ply(), _material_pov(board, color)))  # own material minus opponent
        if board.ply() == 10:
            # Snapshot both sides once, at the same moment, for the development feature.
            developed_at_10[chess.WHITE] = _developed_minors(board, chess.WHITE)
            developed_at_10[chess.BLACK] = _developed_minors(board, chess.BLACK)
    final_ply = board.ply()
    final_developed = {
        chess.WHITE: _developed_minors(board, chess.WHITE),
        chess.BLACK: _developed_minors(board, chess.BLACK),
    }
    rows = []
    for color_name, color in (("white", chess.WHITE), ("black", chess.BLACK)):
        if color == chess.WHITE and "White" not in headers:
            continue
        if color == chess.BLACK and "Black" not in headers:
            continue
        row = _row_for_side(
            color_name,
            color,
            acc[color],
            final_ply,
            developed_at_10,
            final_developed,
            headers,
            game_id,
        )
        if row is not None:
            rows.append(row)
    return rows


def _pgn_quote(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def rows_from_movetext(headers: dict[str, str], movetext_lines: list[str]) -> list[dict[str, Any]]:
    # Names are required to hash a player id. Only the hash is returned.
    body = "\n".join(
        f'[{key} "{_pgn_quote(value)}"]'
        for key, value in headers.items()
        if key in {"White", "Black", "Result"}
    )
    text = body + "\n\n" + "\n".join(movetext_lines) + "\n"
    game = chess.pgn.read_game(io.StringIO(text))
    if game is None:
        return []
    game_id = game_id_from_site(headers.get("Site", ""))
    if not game_id:
        game_id = player_key("\n".join(movetext_lines))
    return rows_from_game(game, headers, game_id)
