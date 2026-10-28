"""Pass 3: Stockfish and behavior features for the selected games only.

Workers seek to the indexed offsets. Each worker flushes batches, so a
stopped run resumes from the games already written. Re-run the same command.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from src.engine import EngineSession, apply_engine_features, ensure_stockfish
from src.features import rows_from_movetext
from src.game_features import extract_game
from src.moves import MOVE_COLUMNS
from src.pgn_stream import parse_headers, read_game_from_handle
from src.schema import DEFAULT_DEPTH, GAME_COLUMNS, GAME_ID_COLUMNS, SAMPLE_COLUMNS, V3_GAME_FEATURES

OUTPUT_COLUMNS = [*GAME_COLUMNS, *SAMPLE_COLUMNS]
OUTPUT_COLUMNS_V3 = [
    *GAME_ID_COLUMNS,
    *[column for column in V3_GAME_FEATURES if column not in GAME_ID_COLUMNS],
    "time_control",
    *SAMPLE_COLUMNS,
    "subset",
]


def checkpoint_dir_for(output: Path) -> Path:
    return output.parent / f"{output.stem}_ckpt"


def meta_path_for(output: Path) -> Path:
    return output.with_name(f"{output.stem}_meta.json")


def is_complete(output: Path) -> bool:
    meta = meta_path_for(output)
    return output.exists() and meta.exists() and bool(json.loads(meta.read_text(encoding="utf-8")).get("completed"))


def build_work_items(manifest: pd.DataFrame) -> list[dict[str, Any]]:
    items = []
    for game_id, group in manifest.groupby("game_id", sort=False):
        offsets = group["offset"].unique()
        if len(offsets) != 1:
            raise ValueError(f"game {game_id} has conflicting offsets")
        keep = {}
        for row in group.itertuples(index=False):
            subset = getattr(row, "subset", 0)
            if subset is None or (isinstance(subset, float) and subset != subset):
                subset = 0
            keep[str(row.color)] = {
                "split": str(row.split),
                "elo_bin": str(row.elo_bin),
                "thin_bin": bool(row.thin_bin),
                "subset": int(subset),
            }
        items.append({"offset": int(offsets[0]), "game_id": str(game_id), "keep": keep})
    items.sort(key=lambda item: item["offset"])
    return items


def _done_ids(ckpt_dir: Path) -> set[str]:
    done: set[str] = set()
    if not ckpt_dir.exists():
        return done
    for path in ckpt_dir.glob("w*_done.json"):
        done.update(str(game_id) for game_id in json.loads(path.read_text(encoding="utf-8")))
    return done


def _next_batch(ckpt_dir: Path, worker: int) -> int:
    nums = []
    for path in ckpt_dir.glob(f"w{worker:02d}_b*.parquet"):
        nums.append(int(path.stem.split("_b")[-1]))
    return max(nums) + 1 if nums else 0


def _write_done(path: Path, game_ids: list[str]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(game_ids), encoding="utf-8")
    tmp.replace(path)


def analyze_chunk(payload: dict[str, Any]) -> dict[str, int]:
    """Worker entry. Seeks forward through one contiguous slice of the PGN."""
    import chess.engine

    ckpt_dir = Path(payload["ckpt"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    worker = int(payload["worker"])
    batch = _next_batch(ckpt_dir, worker)
    done_path = ckpt_dir / f"w{worker:02d}_done.json"
    done_ids = json.loads(done_path.read_text(encoding="utf-8")) if done_path.exists() else []
    rows: list[dict[str, Any]] = []
    moves: list[dict[str, Any]] = []
    feature_version = int(payload.get("feature_version", 2))
    row_columns = OUTPUT_COLUMNS_V3 if feature_version >= 3 else OUTPUT_COLUMNS
    stats = {
        "games": 0,
        "rows": 0,
        "parse_errors": 0,
        "engine_failures": 0,
        "id_mismatches": 0,
        "stockfish_name": None,
    }
    session = None
    if payload.get("stockfish"):
        session = EngineSession(
            payload["stockfish"],
            int(payload["depth"]),
            int(payload["threads"]),
            int(payload["hash_mb"]),
            int(payload["cache_size"]),
        )
    analyse = None if session is None else session.analyse
    if session is not None:
        stats["stockfish_name"] = session.name
    batch_size = int(payload.get("batch_size", 25))

    def flush() -> None:
        nonlocal rows, moves, batch
        wrote = False
        if rows:
            frame = pd.DataFrame(rows, columns=row_columns)
            target = ckpt_dir / f"w{worker:02d}_b{batch:05d}.parquet"
            temporary = target.with_suffix(".parquet.tmp")
            frame.to_parquet(temporary, index=False)
            temporary.replace(target)
            stats["rows"] += len(rows)
            rows = []
            wrote = True
        if feature_version >= 3 and moves:
            frame = pd.DataFrame(moves, columns=MOVE_COLUMNS)
            target = ckpt_dir / f"w{worker:02d}_m{batch:05d}.parquet"
            temporary = target.with_suffix(".parquet.tmp")
            frame.to_parquet(temporary, index=False)
            temporary.replace(target)
            moves = []
            wrote = True
        if wrote:
            batch += 1
        _write_done(done_path, done_ids)

    try:
        with open(payload["path"], "rb") as handle:
            for item in payload["games"]:
                handle.seek(int(item["offset"]))
                parsed = read_game_from_handle(handle)
                game_id = str(item["game_id"])
                move_records: list[dict[str, Any]] = []
                try:
                    if parsed is None:
                        raise ValueError("no headers at offset")
                    header_lines, move_lines = parsed
                    headers = parse_headers(header_lines)
                    if feature_version >= 3:
                        extracted, move_records = extract_game(headers, move_lines, analyse)
                    else:
                        extracted = rows_from_movetext(headers, move_lines)
                        if analyse is not None:
                            extracted = apply_engine_features(headers, move_lines, extracted, analyse)
                except chess.engine.EngineTerminatedError:
                    flush()
                    raise
                except Exception:
                    stats["parse_errors"] += 1
                    done_ids.append(game_id)
                    stats["games"] += 1
                    continue
                keep = item["keep"]
                if feature_version >= 3 and not extracted:
                    stats["engine_failures"] += 1
                    done_ids.append(game_id)
                    stats["games"] += 1
                    continue
                if analyse is not None and any(row.get("cpl_mean") is None for row in extracted if row["color"] in keep):
                    stats["engine_failures"] += 1
                    done_ids.append(game_id)
                    stats["games"] += 1
                    continue
                wrote = False
                for row in extracted:
                    color = str(row["color"])
                    if color not in keep:
                        continue
                    if str(row.get("game_id")) != game_id:
                        stats["id_mismatches"] += 1
                    stamped = dict(row)
                    stamped["game_id"] = game_id
                    stamped["split"] = keep[color]["split"]
                    stamped["elo_bin"] = keep[color]["elo_bin"]
                    stamped["thin_bin"] = keep[color]["thin_bin"]
                    if feature_version >= 3:
                        stamped["subset"] = keep[color]["subset"]
                    rows.append(stamped)
                    wrote = True
                if wrote and feature_version >= 3:
                    moves.extend(move_records)
                if not wrote:
                    stats["parse_errors"] += 1
                done_ids.append(game_id)
                stats["games"] += 1
                if stats["games"] % batch_size == 0:
                    flush()
                    print(
                        f"[analyze] worker {worker} games={stats['games']:,} rows={stats['rows']:,}",
                        flush=True,
                    )
        flush()
        return stats
    finally:
        if session is not None:
            session.close()


def _chunks(items: list[dict[str, Any]], workers: int) -> list[list[dict[str, Any]]]:
    workers = max(1, min(workers, len(items))) if items else 1
    if not items:
        return []
    size = len(items) / workers
    return [items[int(index * size) : int((index + 1) * size)] for index in range(workers) if int(index * size) != int((index + 1) * size)]


def _read_shards(paths: list[Path]) -> list[pd.DataFrame]:
    frames = []
    for path in paths:
        if path.stat().st_size < 8:
            print(f"[analyze] skipping truncated shard {path.name}", flush=True)
            path.unlink()
            continue
        try:
            frames.append(pd.read_parquet(path))
        except Exception as exc:
            print(f"[analyze] skipping unreadable shard {path.name}: {exc}", flush=True)
            path.unlink()
    return frames


def _merge(ckpt_dir: Path, output: Path, columns: list[str] | None = None) -> pd.DataFrame:
    columns = OUTPUT_COLUMNS if columns is None else columns
    frames = _read_shards(sorted(ckpt_dir.glob("w*_b*.parquet")))
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    if len(frame):
        frame = frame.drop_duplicates(["game_id", "color"], keep="last").reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    return frame


def _merge_moves(ckpt_dir: Path, output: Path) -> pd.DataFrame:
    frames = _read_shards(sorted(ckpt_dir.glob("w*_m*.parquet")))
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=MOVE_COLUMNS)
    if len(frame):
        frame = frame.drop_duplicates(["game_id", "ply"], keep="last").reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    return frame


def analyze_games(
    pgn: Path,
    manifest_path: Path,
    output: Path,
    workers: int = 1,
    stockfish: str | None = None,
    depth: int = DEFAULT_DEPTH,
    threads: int = 1,
    hash_mb: int = 128,
    cache_size: int = 200_000,
    fresh: bool = False,
    batch_size: int = 25,
    feature_version: int = 2,
    moves_output: Path | None = None,
) -> dict[str, Any]:
    """Analyze the manifest. ``stockfish=None`` writes behavior features only."""
    ckpt_dir = checkpoint_dir_for(output)
    meta_path = meta_path_for(output)
    if fresh:
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        if output.exists():
            output.unlink()
        if meta_path.exists():
            meta_path.unlink()
    if is_complete(output) and not fresh:
        print(f"[analyze] already complete -> {output}", flush=True)
        return json.loads(meta_path.read_text(encoding="utf-8"))

    manifest = pd.read_parquet(manifest_path)
    items = build_work_items(manifest)
    done = _done_ids(ckpt_dir)
    pending = [item for item in items if item["game_id"] not in done]
    print(
        f"[analyze] games={len(items):,}  pending={len(pending):,}  workers={workers}  "
        f"stockfish={'on' if stockfish else 'off'}",
        flush=True,
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    chunks = _chunks(pending, workers)
    binary = stockfish
    payloads = []
    for index, games in enumerate(chunks):
        payloads.append(
            {
                "path": str(pgn),
                "worker": index,
                "ckpt": str(ckpt_dir),
                "games": games,
                "stockfish": binary,
                "depth": depth,
                "threads": threads,
                "hash_mb": hash_mb,
                "cache_size": cache_size,
                "batch_size": batch_size,
                "feature_version": int(feature_version),
            }
        )
    results: list[dict[str, int]] = []
    if not payloads:
        results = []
    elif len(payloads) == 1:
        results = [analyze_chunk(payloads[0])]
    else:
        import multiprocessing as mp

        mp.freeze_support()
        try:
            with mp.Pool(len(payloads)) as pool:
                results = pool.map(analyze_chunk, payloads)
        except KeyboardInterrupt:
            print("[analyze] interrupted. Re-run the same command to resume.", flush=True)
            raise SystemExit(2) from None
        except Exception:
            print("[analyze] a worker stopped. Re-run the same command to resume.", flush=True)
            raise SystemExit(2)

    columns = OUTPUT_COLUMNS_V3 if int(feature_version) >= 3 else OUTPUT_COLUMNS
    frame = _merge(ckpt_dir, output, columns)
    if int(feature_version) >= 3 and moves_output is not None:
        _merge_moves(ckpt_dir, moves_output)
    from src.sample import assert_split_integrity

    assert_split_integrity(frame)
    names = [item.get("stockfish_name") for item in results if item.get("stockfish_name")]
    stats = {
        "completed": True,
        "rows": int(len(frame)),
        "games": int(frame["game_id"].nunique()) if len(frame) else 0,
        "parse_errors": int(sum(item["parse_errors"] for item in results)),
        "engine_failures": int(sum(item["engine_failures"] for item in results)),
        "id_mismatches": int(sum(item["id_mismatches"] for item in results)),
        "depth": depth if stockfish else None,
        "workers": workers,
        "stockfish": names[0] if int(feature_version) >= 3 else bool(stockfish),
    }
    if int(feature_version) >= 3:
        clock = frame["has_clock"] if "has_clock" in frame.columns else pd.Series(dtype=float)
        stats["feature_version"] = 3
        stats["has_clock_share"] = float((clock == 1).mean()) if len(clock) else 0.0
    meta_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    print(
        f"[analyze] COMPLETE rows={stats['rows']:,} games={stats['games']:,} "
        f"parse_errors={stats['parse_errors']:,} engine_failures={stats['engine_failures']:,} -> {output}",
        flush=True,
    )
    return stats


def default_workers() -> int:
    cpu = os.cpu_count() or 2
    return max(1, min(8, cpu - 1 if cpu > 1 else 1))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stockfish features for a sample manifest.")
    parser.add_argument("--pgn", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash", type=int, default=128, dest="hash_mb")
    parser.add_argument("--stockfish", default=None)
    parser.add_argument("--no-stockfish", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args(argv)
    binary = None
    if not args.no_stockfish:
        binary = ensure_stockfish(args.stockfish or "auto")
    analyze_games(
        args.pgn,
        args.manifest,
        args.output,
        workers=args.workers,
        stockfish=binary,
        depth=args.depth,
        threads=args.threads,
        hash_mb=args.hash_mb,
        fresh=args.fresh,
    )


if __name__ == "__main__":
    main()
