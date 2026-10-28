"""Header check for a pasted PGN. No Stockfish and no ratings in the result."""

from __future__ import annotations

import io

from src.features import rows_from_movetext
from src.pgn_stream import iter_records, parse_headers

# Lichess speed from estimated seconds: base + 40 * increment.
# Under 3 minutes bullet, under 8 blitz, under 25 rapid, and the rest classical.
_BULLET_SECONDS = 180
_BLITZ_SECONDS = 480
_RAPID_SECONDS = 1500
_MIN_MOVES = 10


def parse_time_control(text: str | None) -> tuple[int, int] | None:
    """Return ``(base_seconds, increment_seconds)`` or None when the header is not a clock."""
    raw = (text or "").strip()
    if raw in {"", "-"}:
        return None
    piece = raw.split(":")[-1]
    parts = piece.split("+")
    if len(parts) > 2 or not parts[0].isdigit():
        return None
    base = int(parts[0])
    increment = 0
    if len(parts) == 2:
        if not parts[1].isdigit():
            return None
        increment = int(parts[1])
    return base, increment


def estimated_seconds(text: str | None) -> int | None:
    parsed = parse_time_control(text)
    if parsed is None:
        return None
    base, increment = parsed
    return base + 40 * increment


def pool_from_event(event: str) -> str | None:
    """The speed named in a Lichess event, or None when the header does not name one."""
    text = event.lower()
    if "correspondence" in text:
        return "correspondence"
    # "ultrabullet" contains "bullet", so the longer name has to win.
    for name in ("ultrabullet", "classical", "rapid", "blitz", "bullet"):
        if name in text:
            return name
    return None


def pool_from_clock(text: str | None) -> str | None:
    seconds = estimated_seconds(text)
    if seconds is None:
        return None
    if seconds < _BULLET_SECONDS:
        return "bullet"
    if seconds < _BLITZ_SECONDS:
        return "blitz"
    if seconds < _RAPID_SECONDS:
        return "rapid"
    return "classical"


def _is_foreign(headers: dict[str, str]) -> bool:
    return "chess.com" in headers.get("Site", "").lower()


def _is_unrated(headers: dict[str, str]) -> bool:
    event = headers.get("Event", "").lower()
    if "correspondence" in event or "unrated" in event or "casual" in event:
        return True
    if "rated" not in event:
        return True
    return headers.get("TimeControl", "").strip() in {"", "-"}


def _is_standard(headers: dict[str, str]) -> bool:
    variant = headers.get("Variant", "Standard").strip().lower()
    return variant in {"", "standard", "chess"}


def _is_finished(headers: dict[str, str]) -> bool:
    if headers.get("Termination", "Normal") != "Normal":
        return False
    return headers.get("Result", "*") in {"1-0", "0-1", "1/2-1/2"}


def subject_color(headers: dict[str, str], subject: str) -> str | None:
    """Which side the subject played. Matching ignores case and surrounding space."""
    wanted = subject.strip().casefold()
    if not wanted:
        return None
    white = headers.get("White", "").strip().casefold() == wanted
    black = headers.get("Black", "").strip().casefold() == wanted
    if white:
        return "white"
    if black:
        return "black"
    return None


def _move_count(row: dict | None) -> int | None:
    if row is None:
        return None
    plies = row.get("game_plies")
    if plies is None:
        return None
    return (int(plies) + 1) // 2


def classify_game(
    index: int,
    headers: dict[str, str],
    move_lines: list[str],
    subject: str,
    pool: str,
) -> dict:
    """One checklist row. Ratings are read by nothing in this function's result."""
    color = subject_color(headers, subject)
    rows = rows_from_movetext(headers, move_lines)
    own = next((row for row in rows if row.get("color") == color), None) if color else None
    event_pool = pool_from_event(headers.get("Event", ""))
    detected = event_pool or pool_from_clock(headers.get("TimeControl"))

    if not _is_standard(headers):
        reason = "variant"
    elif _is_foreign(headers):
        reason = "wrong_pool"
    elif _is_unrated(headers):
        reason = "unrated"
    elif detected != pool:
        reason = "wrong_pool"
    elif color is None:
        reason = "name_missing"
    elif not _is_finished(headers):
        reason = "unfinished"
    elif own is None or int(own.get("n_subject_moves") or 0) < _MIN_MOVES:
        reason = "too_short"
    else:
        reason = "ok"

    accepted = reason == "ok"
    return {
        "index": index,
        "accepted": accepted,
        "reason": reason,
        "color": color if accepted else None,
        "white": headers.get("White", ""),
        "black": headers.get("Black", ""),
        "date": headers.get("Date") or None,
        "opening": headers.get("Opening") or None,
        "result": headers.get("Result") or None,
        "moves": _move_count(own) if own is not None else _move_count(rows[0] if rows else None),
    }


def check_pgn(pgn: str, subject: str, pool: str) -> list[dict]:
    """Every game in the paste, in order. An empty paste yields an empty list."""
    games = []
    for index, (header_lines, move_lines) in enumerate(iter_records(io.StringIO(pgn))):
        games.append(classify_game(index, parse_headers(header_lines), move_lines, subject, pool))
    return games
