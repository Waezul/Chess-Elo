"""Stream a Lichess PGN and write one subject-only feature row per side.

Header fields decide whether a game is kept. Movetext is parsed only after
that filter. Names are hashed into ``player_key`` and are not stored.

Checkpointed runs (default for uncompressed single-worker extracts) flush
parquet shards as they go, print elapsed/coverage progress, and can resume
after Ctrl+C from the last saved byte offset.

Examples
--------
python -m src.extract --input games.pgn --output data/games.parquet --max-games 5000
python -m src.extract --input games.pgn --output data/games.parquet --max-games 8000 --segments 8
python -m src.extract --input games.pgn.zst --output data/games.parquet --workers 4
python -m src.extract --input games.pgn --output data/games.parquet --with-stockfish
# resume after interrupt (same command; auto-detects checkpoint):
python -m src.extract --input games.pgn --output data/games.parquet --with-stockfish
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.engine import EngineSession, apply_engine_features
from src.features import accept_headers, rows_from_movetext
from src.pgn_stream import (
    chunk_spans,
    iter_chunk_lines,
    iter_records,
    iter_records_with_offset,
    open_pgn_text,
    parse_headers,
)
from src.schema import ELO_MAX, ELO_MIN, ENGINE_FEATURES, GAME_COLUMNS, NUMERIC_FEATURES, POOLS

STATE_NAME = "state.json"


def checkpoint_dir_for(output: Path) -> Path:
    return output.parent / f"{output.stem}_ckpt"


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


def _source_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "source": str(path.resolve()),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
    }


def _load_state(ckpt_dir: Path) -> dict[str, Any] | None:
    path = ckpt_dir / STATE_NAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(ckpt_dir: Path, state: dict[str, Any]) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / STATE_NAME
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)


def _shard_path(ckpt_dir: Path, index: int) -> Path:
    return ckpt_dir / f"shard_{index:05d}.parquet"


def _merge_shards(ckpt_dir: Path, output: Path, shard_count: int) -> int:
    frames = []
    for index in range(shard_count):
        part = _shard_path(ckpt_dir, index)
        if part.exists() and part.stat().st_size > 0:
            frames.append(pd.read_parquet(part))
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=GAME_COLUMNS)
    write_games(frame, output)
    return int(len(frame))


def _flush_shard(
    ckpt_dir: Path,
    shard_index: int,
    rows: list[dict[str, Any]],
) -> int:
    if not rows:
        return shard_index
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=GAME_COLUMNS)
    path = _shard_path(ckpt_dir, shard_index)
    frame.to_parquet(path, index=False)
    print(f"[extract] checkpoint shard {shard_index:05d} rows={len(frame):,} -> {path}", flush=True)
    return shard_index + 1


def iter_feature_rows(
    lines,
    pool: str,
    elo_min: int,
    elo_max: int,
    max_games: int | None,
    analyse_position=None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    stats = {"scanned": 0, "kept": 0, "parse_errors": 0, "engine_rows": 0}
    step = 10 if analyse_position is not None else 500
    for header_lines, move_lines in iter_records(lines):
        stats["scanned"] += 1
        headers = parse_headers(header_lines)
        if not accept_headers(headers, pool=pool, elo_min=elo_min, elo_max=elo_max):
            continue
        if max_games is not None and stats["kept"] >= max_games:
            break
        try:
            extracted = rows_from_movetext(headers, move_lines)
            extracted = apply_engine_features(headers, move_lines, extracted, analyse_position)
        except Exception:
            stats["parse_errors"] += 1
            continue
        if not extracted:
            stats["parse_errors"] += 1
            continue
        stats["kept"] += 1
        stats["engine_rows"] += sum(1 for row in extracted if row.get("cpl_mean") is not None)
        rows.extend(extracted)
        if stats["kept"] % step == 0:
            print(f"kept {stats['kept']} games, scanned {stats['scanned']}", flush=True)
    return rows, stats


def extract_path_checkpointed(
    path: Path,
    output: Path,
    pool: str,
    elo_min: int,
    elo_max: int,
    max_games: int | None,
    stockfish: str | None,
    depth: int,
    threads: int,
    hash_mb: int,
    cache_size: int,
    checkpoint_minutes: float,
    fresh: bool,
) -> dict[str, Any]:
    """Single-process extract with progress, shard flushes, and resume."""
    if path.name.endswith(".zst"):
        raise SystemExit("checkpoint/resume needs an uncompressed .pgn (not .zst)")

    # <=0 means flush/print after every kept game (tests / debug only).
    checkpoint_seconds = 0.0 if checkpoint_minutes <= 0 else float(checkpoint_minutes) * 60.0
    ckpt_dir = checkpoint_dir_for(output)
    fingerprint = _source_fingerprint(path)
    file_size = fingerprint["source_size"]
    state = None if fresh else _load_state(ckpt_dir)

    if state and not state.get("completed"):
        same_source = (
            state.get("source") == fingerprint["source"]
            and state.get("source_size") == fingerprint["source_size"]
            and state.get("source_mtime_ns") == fingerprint["source_mtime_ns"]
            and state.get("pool") == pool
            and state.get("elo_min") == elo_min
            and state.get("elo_max") == elo_max
            and state.get("max_games") == max_games
            and bool(state.get("stockfish_enabled")) == bool(stockfish)
        )
        if not same_source:
            raise SystemExit(
                f"checkpoint at {ckpt_dir} does not match this run. "
                "Re-run with --fresh to start over, or restore the original input/settings."
            )
        print(
            f"[extract] RESUME from byte {_format_bytes(int(state['byte_offset']))} "
            f"({100 * state['byte_offset'] / file_size:.2f}% of file)  "
            f"kept so far={state['kept']:,}  shards={state['shard_count']}",
            flush=True,
        )
    else:
        if fresh and ckpt_dir.exists():
            for item in ckpt_dir.glob("shard_*.parquet"):
                item.unlink()
            state_path = ckpt_dir / STATE_NAME
            if state_path.exists():
                state_path.unlink()
            print(f"[extract] --fresh: cleared checkpoint {ckpt_dir}", flush=True)
        state = {
            **fingerprint,
            "pool": pool,
            "elo_min": elo_min,
            "elo_max": elo_max,
            "max_games": max_games,
            "stockfish_enabled": bool(stockfish),
            "byte_offset": 0,
            "scanned": 0,
            "kept": 0,
            "parse_errors": 0,
            "engine_rows": 0,
            "shard_count": 0,
            "completed": False,
            "rows": 0,
        }
        interval_txt = "every kept game" if checkpoint_seconds <= 0 else f"every {checkpoint_minutes:g} min"
        print(
            f"[extract] START checkpointed run  file={_format_bytes(file_size)}  "
            f"ckpt={ckpt_dir}  save/print={interval_txt}",
            flush=True,
        )

    stop_requested = {"flag": False}

    def _request_stop(signum, _frame) -> None:
        if stop_requested["flag"]:
            return
        stop_requested["flag"] = True
        print(
            f"\n[extract] signal {signum} received - finishing current game, then saving checkpoint...",
            flush=True,
        )

    previous_int = signal.getsignal(signal.SIGINT)
    previous_term = signal.getsignal(signal.SIGTERM) if hasattr(signal, "SIGTERM") else None
    signal.signal(signal.SIGINT, _request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _request_stop)

    session = (
        EngineSession(stockfish, depth, threads, hash_mb, cache_size) if stockfish else None
    )
    analyse = None if session is None else session.analyse
    buffer: list[dict[str, Any]] = []
    started = time.monotonic()
    last_checkpoint = started
    byte_offset = int(state["byte_offset"])
    interrupted = False

    def _checkpoint_now(reason: str) -> None:
        nonlocal buffer, last_checkpoint
        if buffer:
            added = len(buffer)
            state["shard_count"] = _flush_shard(ckpt_dir, int(state["shard_count"]), buffer)
            buffer = []
            state["rows"] = int(state.get("rows", 0)) + added
        state["byte_offset"] = byte_offset
        _save_state(ckpt_dir, state)
        pending = 0
        state_view = {**state, "rows": int(state.get("rows", 0)) + pending}
        _print_progress(started, file_size, byte_offset, state_view)
        last_checkpoint = time.monotonic()
        print(f"[extract] checkpoint saved ({reason})", flush=True)

    try:
        for header_lines, move_lines, resume_offset in iter_records_with_offset(
            str(path), start=byte_offset
        ):
            state["scanned"] = int(state["scanned"]) + 1
            byte_offset = resume_offset
            headers = parse_headers(header_lines)
            if accept_headers(headers, pool=pool, elo_min=elo_min, elo_max=elo_max):
                try:
                    extracted = rows_from_movetext(headers, move_lines)
                    extracted = apply_engine_features(headers, move_lines, extracted, analyse)
                except Exception:
                    state["parse_errors"] = int(state["parse_errors"]) + 1
                    extracted = None
                if extracted is None:
                    pass
                elif not extracted:
                    state["parse_errors"] = int(state["parse_errors"]) + 1
                else:
                    state["kept"] = int(state["kept"]) + 1
                    state["engine_rows"] = int(state["engine_rows"]) + sum(
                        1 for row in extracted if row.get("cpl_mean") is not None
                    )
                    buffer.extend(extracted)
                    if max_games is not None and int(state["kept"]) >= max_games:
                        # Stop before pulling the next game so resume offsets stay correct.
                        break

            now = time.monotonic()
            due = checkpoint_seconds <= 0 or (now - last_checkpoint) >= checkpoint_seconds
            if due and (buffer or checkpoint_seconds > 0):
                # Time-based: flush+print on interval. Zero interval: after each kept game in buffer.
                if checkpoint_seconds <= 0:
                    if buffer:
                        _checkpoint_now("per-game")
                else:
                    _checkpoint_now(f"{checkpoint_minutes:g}min interval")
            if stop_requested["flag"]:
                interrupted = True
                break

        _checkpoint_now("final" if not interrupted else "interrupt")
        state["completed"] = (not interrupted) and (
            byte_offset >= file_size or (max_games is not None and int(state["kept"]) >= max_games)
        )

        rows = _merge_shards(ckpt_dir, output, int(state["shard_count"]))
        state["rows"] = rows
        if session is not None:
            state["stockfish"] = session.name
            state["cache_hits"] = session.cache.hits
            state["cache_misses"] = session.cache.misses
        _save_state(ckpt_dir, state)

        if interrupted:
            print(
                f"[extract] STOPPED - kept data written to {output} "
                f"({rows:,} rows, {state['kept']:,} games). "
                f"Re-run the same command to RESUME from "
                f"{100 * byte_offset / file_size:.2f}%.",
                flush=True,
            )
        else:
            print(
                f"[extract] COMPLETE - {rows:,} rows from {state['kept']:,} games -> {output}",
                flush=True,
            )
        return {
            "scanned": int(state["scanned"]),
            "kept": int(state["kept"]),
            "parse_errors": int(state["parse_errors"]),
            "rows": rows,
            "engine_rows": int(state["engine_rows"]),
            "stockfish": state.get("stockfish"),
            "cache_hits": int(state.get("cache_hits", 0)),
            "cache_misses": int(state.get("cache_misses", 0)),
            "completed": bool(state["completed"]),
            "byte_offset": byte_offset,
            "file_size": file_size,
            "checkpoint_dir": str(ckpt_dir),
            "elapsed_sec": time.monotonic() - started,
        }
    finally:
        signal.signal(signal.SIGINT, previous_int)
        if previous_term is not None:
            signal.signal(signal.SIGTERM, previous_term)
        if session is not None:
            session.close()


def _print_progress(started: float, file_size: int, byte_offset: int, state: dict[str, Any]) -> None:
    elapsed = time.monotonic() - started
    frac = byte_offset / file_size if file_size else 0.0
    kept = int(state["kept"])
    scanned = int(state["scanned"])
    rate = kept / elapsed if elapsed > 0 else 0.0
    eta = (elapsed / frac - elapsed) if frac > 0.001 else float("nan")
    eta_txt = _format_duration(eta) if eta == eta and eta < 1e8 else "n/a"
    print(
        f"[extract] elapsed={_format_duration(elapsed)}  "
        f"file={100 * frac:.2f}% ({_format_bytes(byte_offset)}/{_format_bytes(file_size)})  "
        f"scanned={scanned:,}  kept={kept:,}  rows={int(state.get('rows', 0)):,}  "
        f"errors={int(state['parse_errors']):,}  "
        f"kept/min={60 * rate:.2f}  ETA={eta_txt}",
        flush=True,
    )


def _extract_span(payload: dict[str, Any]) -> dict[str, Any]:
    """Worker entry. Writes one parquet part and returns counts."""
    session = None
    analyse = None
    if payload.get("stockfish"):
        session = EngineSession(
            payload["stockfish"],
            payload["depth"],
            payload["threads"],
            payload["hash_mb"],
            payload["cache_size"],
        )
        analyse = session.analyse
    try:
        rows, stats = iter_feature_rows(
            iter_chunk_lines(payload["path"], payload["start"], payload["end"], payload["is_last"]),
            pool=payload["pool"],
            elo_min=payload["elo_min"],
            elo_max=payload["elo_max"],
            max_games=payload["max_games"],
            analyse_position=analyse,
        )
    finally:
        if session is not None:
            stats_name = session.name
            stats_hits = session.cache.hits
            stats_misses = session.cache.misses
            session.close()
        else:
            stats_name = None
            stats_hits = 0
            stats_misses = 0
    part = payload["part"]
    frame = pd.DataFrame(rows, columns=GAME_COLUMNS)
    if len(frame):
        frame.to_parquet(part, index=False)
    elif os.path.exists(part):
        os.remove(part)
    return {
        "part": part if len(frame) else None,
        "rows": int(len(frame)),
        "stockfish": stats_name,
        "cache_hits": stats_hits,
        "cache_misses": stats_misses,
        **stats,
    }


def write_games(frame: pd.DataFrame, output: Path) -> None:
    # One parquet row per side. Behavior columns and Stockfish columns share this file.
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)


def extract_path(
    path: Path,
    output: Path,
    pool: str,
    elo_min: int,
    elo_max: int,
    max_games: int | None,
    workers: int,
    segments: int,
    stockfish: str | None = None,
    depth: int = 10,
    threads: int = 1,
    hash_mb: int = 128,
    cache_size: int = 200_000,
    checkpoint: bool = True,
    checkpoint_minutes: float = 30.0,
    fresh: bool = False,
) -> dict[str, Any]:
    zst = path.name.endswith(".zst")
    if zst and (workers > 1 or segments > 1):
        print("zstd input is streamed in one process", flush=True)
        workers = 1
        segments = 1
    # A capped sample stays single-process so it is the first accepted games.
    # --segments N instead takes a quota from N slices spread through the file,
    # which is what produces repeat players in a month-long chronological dump.
    if segments > 1:
        if max_games is None:
            raise SystemExit("--segments requires --max-games")
        workers = segments
        checkpoint = False
    elif max_games is not None and not stockfish:
        workers = 1

    use_checkpoint = checkpoint and not zst and workers <= 1 and segments <= 1
    if checkpoint and not use_checkpoint:
        print(
            "[extract] checkpoint/resume disabled for this mode "
            "(needs uncompressed .pgn, workers=1, segments=1). "
            "For overnight+resume use: --workers 1 (default when checkpointing).",
            flush=True,
        )
    if checkpoint and workers > 1 and segments <= 1 and not zst:
        print("[extract] forcing workers=1 so progress + resume checkpoints work", flush=True)
        workers = 1
        use_checkpoint = True

    if use_checkpoint:
        return extract_path_checkpointed(
            path,
            output,
            pool=pool,
            elo_min=elo_min,
            elo_max=elo_max,
            max_games=max_games,
            stockfish=stockfish,
            depth=depth,
            threads=threads,
            hash_mb=hash_mb,
            cache_size=cache_size,
            checkpoint_minutes=checkpoint_minutes,
            fresh=fresh,
        )

    session = (
        EngineSession(stockfish, depth, threads, hash_mb, cache_size)
        if stockfish and workers <= 1
        else None
    )
    try:
        if workers <= 1:
            text, binary = open_pgn_text(str(path))
            try:
                rows, stats = iter_feature_rows(
                    text,
                    pool,
                    elo_min,
                    elo_max,
                    max_games,
                    analyse_position=None if session is None else session.analyse,
                )
            finally:
                text.close()
                if binary is not None:
                    binary.close()
            frame = pd.DataFrame(rows, columns=GAME_COLUMNS)
            write_games(frame, output)
            stats["rows"] = int(len(frame))
            stats["completed"] = True
            if session is not None:
                stats["stockfish"] = session.name
                stats["cache_hits"] = session.cache.hits
                stats["cache_misses"] = session.cache.misses
            return stats

        spans = chunk_spans(str(path), workers)
        payloads = []
        parts = []
        if max_games is None:
            quotas = [None] * len(spans)
        else:
            base, remainder = divmod(max_games, len(spans))
            quotas = [base + (1 if index < remainder else 0) for index in range(len(spans))]
        for index, (start, end, is_last) in enumerate(spans):
            part = str(output.with_suffix(f".part{index}.parquet"))
            parts.append(part)
            payloads.append(
                {
                    "path": str(path),
                    "start": start,
                    "end": end,
                    "is_last": is_last,
                    "pool": pool,
                    "elo_min": elo_min,
                    "elo_max": elo_max,
                    "max_games": quotas[index],
                    "part": part,
                    "stockfish": stockfish,
                    "depth": depth,
                    "threads": threads,
                    "hash_mb": hash_mb,
                    "cache_size": cache_size,
                }
            )
        import multiprocessing as mp

        import src.extract as extract_mod

        # Use the importable module so Windows spawn can pickle the worker
        # when this file is launched with `python -m src.extract`.
        mp.freeze_support()
        with mp.Pool(len(payloads)) as pool_exec:
            results = pool_exec.map(extract_mod._extract_span, payloads)
        frames = [pd.read_parquet(item["part"]) for item in results if item["part"]]
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=GAME_COLUMNS)
        if max_games is not None and segments <= 1:
            keep_ids = list(dict.fromkeys(frame["game_id"].tolist()))[:max_games]
            frame = frame[frame["game_id"].isin(keep_ids)]
        write_games(frame, output)
        for part in parts:
            if os.path.exists(part):
                os.remove(part)
        return {
            "scanned": int(sum(item["scanned"] for item in results)),
            "kept": int(frame["game_id"].nunique()) if len(frame) else 0,
            "parse_errors": int(sum(item["parse_errors"] for item in results)),
            "rows": int(len(frame)),
            "engine_rows": int(sum(item.get("engine_rows", 0) for item in results)),
            "stockfish": next((item.get("stockfish") for item in results if item.get("stockfish")), None),
            "cache_hits": int(sum(item.get("cache_hits", 0) for item in results)),
            "cache_misses": int(sum(item.get("cache_misses", 0) for item in results)),
            "completed": True,
        }
    finally:
        if session is not None:
            session.close()


def build_meta(path: Path, args: argparse.Namespace, stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": str(path),
        "pool": args.pool,
        "elo_min": args.elo_min,
        "elo_max": args.elo_max,
        "termination": "Normal",
        "max_games": args.max_games,
        "workers": args.workers,
        "segments": args.segments,
        "excluded_from_features": [
            "player names",
            "opponent elo",
            "rating diff",
            "site url",
            "game result",
        ],
        "numeric_features": [*NUMERIC_FEATURES, *ENGINE_FEATURES],
        "stockfish_depth": args.depth,
        "mistake_cp": 100,
        "blunder_cp": 200,
        **stats,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract subject-only chess behavior features.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pool", default="classical", choices=POOLS)
    parser.add_argument("--elo-min", type=int, default=ELO_MIN)
    parser.add_argument("--elo-max", type=int, default=ELO_MAX)
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--segments",
        type=int,
        default=1,
        help="With --max-games, take a quota from this many slices spread through the file.",
    )
    parser.add_argument("--meta", type=Path, default=None)
    parser.add_argument(
        "--with-stockfish",
        action="store_true",
        help="Score subject moves with Stockfish and store centipawn features on each row.",
    )
    parser.add_argument("--stockfish", default=None, help="Path to the Stockfish binary, or omit to search and download.")
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash", type=int, default=128, dest="hash_mb")
    parser.add_argument("--cache-size", type=int, default=200_000)
    parser.add_argument(
        "--checkpoint-minutes",
        type=float,
        default=30.0,
        help="Flush shards + print progress every N minutes (default 30). Use 0 for every kept game.",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable shard checkpointing / resume (old in-memory behavior).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore and clear any existing checkpoint; start from the beginning.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.input.exists():
        raise SystemExit(f"input not found: {args.input}")
    binary = None
    if args.with_stockfish or args.stockfish:
        from src.engine import ensure_stockfish

        binary = ensure_stockfish(args.stockfish or "auto")
    stats = extract_path(
        args.input,
        args.output,
        pool=args.pool,
        elo_min=args.elo_min,
        elo_max=args.elo_max,
        max_games=args.max_games,
        workers=args.workers,
        segments=args.segments,
        stockfish=binary,
        depth=args.depth,
        threads=args.threads,
        hash_mb=args.hash_mb,
        cache_size=args.cache_size,
        checkpoint=not args.no_checkpoint,
        checkpoint_minutes=args.checkpoint_minutes,
        fresh=args.fresh,
    )
    meta_path = args.meta or args.output.with_name("extract_meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(build_meta(args.input, args, stats), indent=2), encoding="utf-8")
    print(
        f"wrote {stats['rows']} rows from {stats['kept']} games "
        f"(scanned {stats['scanned']}, parse errors {stats['parse_errors']}) -> {args.output}",
        flush=True,
    )
    if stats.get("completed") is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
