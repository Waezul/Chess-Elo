"""Stream PGN text, including zstd, and yield games without a full decompress."""

from __future__ import annotations

import io
import os
import re
from collections.abc import Iterator
from typing import BinaryIO, TextIO

_HEADER = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$')


def parse_headers(lines: list[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for line in lines:
        match = _HEADER.match(line.strip())
        if not match:
            continue
        value = match.group(2).replace(r"\"", '"').replace(r"\\", "\\")
        headers[match.group(1)] = value
    return headers


def iter_records(lines: Iterator[str]) -> Iterator[tuple[list[str], list[str]]]:
    """Yield (header lines, movetext lines) for each PGN game."""
    headers: list[str] = []
    moves: list[str] = []
    mode = "seek"
    for raw in lines:
        line = raw.strip().lstrip("\ufeff")
        if not line:
            if mode == "header":
                mode = "moves"
            continue
        if line.startswith("["):
            if mode == "moves":
                if headers:
                    yield headers, moves
                headers, moves = [], []
            mode = "header"
            headers.append(line)
            continue
        if mode == "seek":
            continue
        mode = "moves"
        moves.append(line)
    if headers:
        yield headers, moves


def iter_records_with_offset(path: str, start: int = 0) -> Iterator[tuple[list[str], list[str], int]]:
    """Yield (headers, moves, resume_offset) from an uncompressed PGN.

    ``resume_offset`` is the byte position of the first line of the *next* game
    (or EOF). Seek there on resume. Only valid for plain ``.pgn`` files.
    """
    if path.endswith(".zst"):
        raise ValueError("byte-offset resume requires an uncompressed .pgn")
    headers: list[str] = []
    moves: list[str] = []
    mode = "seek"
    with open(path, "rb") as handle:
        if start > 0:
            handle.seek(start)
        while True:
            line_start = handle.tell()
            raw = handle.readline()
            if not raw:
                if headers:
                    yield headers, moves, handle.tell()
                return
            line = raw.decode("utf-8", "replace").strip().lstrip("\ufeff")
            if not line:
                if mode == "header":
                    mode = "moves"
                continue
            if line.startswith("["):
                if mode == "moves" and headers:
                    yield headers, moves, line_start
                    headers, moves = [line], []
                    mode = "header"
                    continue
                mode = "header"
                headers.append(line)
                continue
            if mode == "seek":
                continue
            mode = "moves"
            moves.append(line)


def _decode_line(raw: bytes) -> str:
    return raw.decode("utf-8", "replace").strip().lstrip("\ufeff")


def iter_games(path: str, start: int = 0) -> Iterator[tuple[int, list[str], list[str], int]]:
    """Yield ``(game_start, headers, moves, next_offset)`` for an uncompressed PGN.

    ``next_offset`` is the byte position of the next game's first header line.
    Resume by passing it as ``start``. Stockfish is not involved.
    """
    if path.endswith(".zst"):
        raise ValueError("byte-offset indexing requires an uncompressed .pgn")
    headers: list[str] = []
    moves: list[str] = []
    mode = "seek"
    game_start: int | None = None
    with open(path, "rb") as handle:
        if start > 0:
            handle.seek(start)
        while True:
            line_start = handle.tell()
            raw = handle.readline()
            if not raw:
                if headers and game_start is not None:
                    yield game_start, headers, moves, handle.tell()
                return
            line = _decode_line(raw)
            if not line:
                if mode == "header":
                    mode = "moves"
                continue
            if line.startswith("["):
                if mode == "moves" and headers and game_start is not None:
                    yield game_start, headers, moves, line_start
                    headers, moves = [line], []
                    game_start = line_start
                    mode = "header"
                    continue
                if mode != "header":
                    game_start = line_start
                mode = "header"
                headers.append(line)
                continue
            if mode == "seek":
                continue
            mode = "moves"
            moves.append(line)


def read_game_from_handle(handle: BinaryIO) -> tuple[list[str], list[str]] | None:
    """Read one game from the handle's current position. The next game's header is consumed."""
    headers: list[str] = []
    moves: list[str] = []
    mode = "seek"
    while True:
        raw = handle.readline()
        if not raw:
            break
        line = _decode_line(raw)
        if not line:
            if mode == "header":
                mode = "moves"
            continue
        if line.startswith("["):
            if mode == "moves":
                break
            mode = "header"
            headers.append(line)
            continue
        if mode == "seek":
            continue
        mode = "moves"
        moves.append(line)
    if not headers:
        return None
    return headers, moves


def read_game_at(path: str, offset: int) -> tuple[list[str], list[str]] | None:
    """Read the single game whose first header byte is ``offset``."""
    with open(path, "rb") as handle:
        handle.seek(offset)
        return read_game_from_handle(handle)


def open_pgn_text(path: str) -> tuple[TextIO, BinaryIO | None]:
    """Return a text stream and an optional binary handle that must be closed."""
    if path.endswith(".zst"):
        import zstandard

        binary = open(path, "rb")
        reader = zstandard.ZstdDecompressor().stream_reader(binary)
        text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace", newline="")
        return text, binary
    return open(path, "r", encoding="utf-8", errors="replace", newline=""), None


def _decode(raw: bytes) -> str:
    return raw.decode("utf-8", "replace")


def iter_chunk_lines(path: str, start: int, end: int, is_last: bool) -> Iterator[str]:
    """Lines for a byte span of an uncompressed PGN, aligned to `[Event` boundaries."""
    file_end = os.path.getsize(path) if is_last else end
    with open(path, "rb") as handle:
        if start > 0:
            handle.seek(start - 1)
            at_line_start = handle.read(1) == b"\n"
            handle.seek(start)
            if not at_line_start:
                handle.readline()
            while True:
                pos = handle.tell()
                raw = handle.readline()
                if not raw:
                    return
                if raw.startswith(b"[Event ") and (is_last or pos < file_end):
                    yield _decode(raw)
                    break
                if not is_last and pos >= file_end:
                    return
        while True:
            pos = handle.tell()
            raw = handle.readline()
            if not raw:
                return
            if raw.startswith(b"[Event ") and not is_last and pos >= file_end:
                return
            yield _decode(raw)


def chunk_spans(path: str, workers: int) -> list[tuple[int, int, bool]]:
    size = os.path.getsize(path)
    workers = max(1, workers)
    span = max(1, size // workers)
    spans: list[tuple[int, int, bool]] = []
    for index in range(workers):
        start = index * span
        is_last = index == workers - 1
        end = size if is_last else (index + 1) * span
        if start >= size:
            break
        spans.append((start, end, is_last))
    return spans
