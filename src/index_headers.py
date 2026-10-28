"""Pass 1: index accepted games by byte offset. No Stockfish.

Resume with the same command. The final parquet is written only when the
file has been scanned, so a sample cannot start from a partial month.
"""

from __future__ import annotations

import argparse
import json
import shutil
import signal
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.features import accept_headers, game_id_from_site, parse_elo, player_key
from src.pgn_stream import iter_games, parse_headers
from src.progress import format_bytes, format_duration
from src.schema import ELO_MAX, ELO_MIN

INDEX_COLUMNS = ["offset", "game_id", "white_key", "black_key", "white_elo", "black_elo"]
STATE_NAME = "state.json"


def checkpoint_dir_for(output: Path) -> Path:
    return output.parent / f"{output.stem}_ckpt"


def meta_path_for(output: Path) -> Path:
    return output.with_name(f"{output.stem}_meta.json")


def _load_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)


def _shard_path(ckpt_dir: Path, index: int) -> Path:
    return ckpt_dir / f"shard_{index:05d}.parquet"


def _seen_ids(ckpt_dir: Path, shard_count: int) -> set[str]:
    seen: set[str] = set()
    for index in range(shard_count):
        path = _shard_path(ckpt_dir, index)
        if path.exists() and path.stat().st_size > 0:
            seen.update(pd.read_parquet(path, columns=["game_id"])["game_id"].astype(str))
    return seen


def _flush(ckpt_dir: Path, shard_index: int, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return shard_index
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    path = _shard_path(ckpt_dir, shard_index)
    frame.to_parquet(path, index=False)
    print(f"[index] shard {shard_index:05d} rows={len(frame):,}", flush=True)
    return shard_index + 1


def _merge(ckpt_dir: Path, output: Path, shard_count: int) -> int:
    frames = []
    for index in range(shard_count):
        path = _shard_path(ckpt_dir, index)
        if path.exists() and path.stat().st_size > 0:
            frames.append(pd.read_parquet(path))
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=INDEX_COLUMNS)
    if len(frame):
        frame = frame.drop_duplicates("game_id", keep="first")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    return int(len(frame))


def _fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "source": str(path.resolve()),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
    }


def is_complete(output: Path) -> bool:
    meta = meta_path_for(output)
    return output.exists() and meta.exists() and bool(json.loads(meta.read_text(encoding="utf-8")).get("completed"))


def index_pgn(
    path: Path,
    output: Path,
    pool: str = "classical",
    elo_min: int = ELO_MIN,
    elo_max: int = ELO_MAX,
    fresh: bool = False,
    checkpoint_seconds: float = 30.0,
) -> dict[str, Any]:
    """Scan ``path`` and write one index row per accepted game."""
    if path.name.endswith(".zst"):
        raise SystemExit("header index needs an uncompressed .pgn so games can be seeked later")
    if not path.exists():
        raise SystemExit(f"PGN not found: {path}")

    ckpt_dir = checkpoint_dir_for(output)
    state_path = ckpt_dir / STATE_NAME
    fingerprint = _fingerprint(path)
    if fresh and ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    if fresh and output.exists():
        output.unlink()
    meta_path = meta_path_for(output)
    if fresh and meta_path.exists():
        meta_path.unlink()

    state = _load_state(state_path)
    if state and state.get("completed") and output.exists() and not fresh:
        if state.get("source_size") != fingerprint["source_size"] or state.get("pool") != pool:
            raise SystemExit("index checkpoint does not match this PGN or pool. Re-run with --fresh.")
        if not meta_path.exists():
            meta = {
                "completed": True,
                "rows": int(state.get("rows", state.get("kept", 0))),
                "scanned": int(state.get("scanned", 0)),
                "duplicates": int(state.get("duplicates", 0)),
                "pool": pool,
                "elo_min": elo_min,
                "elo_max": elo_max,
                "source": fingerprint["source"],
            }
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[index] already complete -> {output}", flush=True)
        return state
    if state is None or fresh:
        state = {
            **fingerprint,
            "pool": pool,
            "elo_min": elo_min,
            "elo_max": elo_max,
            "byte_offset": 0,
            "scanned": 0,
            "kept": 0,
            "duplicates": 0,
            "shard_count": 0,
            "completed": False,
        }
    elif state.get("source_size") != fingerprint["source_size"] or state.get("pool") != pool:
        raise SystemExit("index checkpoint does not match this PGN or pool. Re-run with --fresh.")

    file_size = fingerprint["source_size"]
    seen = _seen_ids(ckpt_dir, int(state["shard_count"]))
    buffer: list[dict[str, Any]] = []
    byte_offset = int(state["byte_offset"])
    started = time.monotonic()
    last_flush = started
    stop = {"flag": False}

    def _request_stop(signum, _frame) -> None:
        stop["flag"] = True
        print(f"\n[index] signal {signum}, saving checkpoint after this game", flush=True)

    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _request_stop)

    def _checkpoint(reason: str) -> None:
        nonlocal buffer, last_flush
        if buffer:
            state["shard_count"] = _flush(ckpt_dir, int(state["shard_count"]), buffer)
            buffer = []
        state["byte_offset"] = byte_offset
        _save_state(state_path, state)
        elapsed = time.monotonic() - started
        frac = byte_offset / file_size if file_size else 0.0
        eta = (elapsed / frac - elapsed) if frac > 0.001 else float("nan")
        eta_txt = format_duration(eta) if eta == eta and eta < 1e8 else "n/a"
        print(
            f"[index] {reason}  elapsed={format_duration(elapsed)}  "
            f"file={100 * frac:.2f}% ({format_bytes(byte_offset)}/{format_bytes(file_size)})  "
            f"scanned={int(state['scanned']):,}  kept={int(state['kept']):,}  ETA={eta_txt}",
            flush=True,
        )
        last_flush = time.monotonic()

    interrupted = False
    try:
        for game_start, header_lines, _moves, next_offset in iter_games(str(path), start=byte_offset):
            byte_offset = next_offset
            state["scanned"] = int(state["scanned"]) + 1
            headers = parse_headers(header_lines)
            if accept_headers(headers, pool=pool, elo_min=elo_min, elo_max=elo_max):
                game_id = game_id_from_site(headers.get("Site", "")) or f"offset-{game_start}"
                if game_id in seen:
                    state["duplicates"] = int(state["duplicates"]) + 1
                else:
                    white_elo = parse_elo(headers.get("WhiteElo"))
                    black_elo = parse_elo(headers.get("BlackElo"))
                    if white_elo is None or black_elo is None:
                        continue
                    buffer.append(
                        {
                            "offset": int(game_start),
                            "game_id": game_id,
                            "white_key": player_key(headers["White"]),
                            "black_key": player_key(headers["Black"]),
                            "white_elo": int(white_elo),
                            "black_elo": int(black_elo),
                        }
                    )
                    seen.add(game_id)
                    state["kept"] = int(state["kept"]) + 1
            now = time.monotonic()
            due = checkpoint_seconds <= 0 or (now - last_flush) >= checkpoint_seconds
            if due and (buffer or checkpoint_seconds > 0):
                _checkpoint("checkpoint")
            if stop["flag"]:
                interrupted = True
                break
    except KeyboardInterrupt:
        interrupted = True
        print("[index] interrupted, saving checkpoint", flush=True)
    finally:
        signal.signal(signal.SIGINT, previous)

    state["completed"] = (not interrupted) and byte_offset >= file_size
    _checkpoint("paused" if interrupted else "final")
    if state["completed"]:
        rows = _merge(ckpt_dir, output, int(state["shard_count"]))
        state["rows"] = rows
        _save_state(state_path, state)
        meta = {
            "completed": True,
            "rows": rows,
            "scanned": int(state["scanned"]),
            "duplicates": int(state["duplicates"]),
            "pool": pool,
            "elo_min": elo_min,
            "elo_max": elo_max,
            "source": fingerprint["source"],
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        for shard in ckpt_dir.glob("shard_*.parquet"):
            shard.unlink()
        print(f"[index] COMPLETE {rows:,} games -> {output}", flush=True)
    else:
        frac = byte_offset / file_size if file_size else 0.0
        print(
            f"[index] STOPPED at {100 * frac:.2f}%. "
            "Re-run the same command to resume. Sampling waits until the index completes.",
            flush=True,
        )
    return state


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Index a PGN by header. Does not run Stockfish.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pool", default="classical")
    parser.add_argument("--elo-min", type=int, default=ELO_MIN)
    parser.add_argument("--elo-max", type=int, default=ELO_MAX)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--checkpoint-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    state = index_pgn(
        args.input,
        args.output,
        pool=args.pool,
        elo_min=args.elo_min,
        elo_max=args.elo_max,
        fresh=args.fresh,
        checkpoint_seconds=args.checkpoint_seconds,
    )
    if not state.get("completed"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
