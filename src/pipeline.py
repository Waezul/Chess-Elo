"""Build the site training table without engine-analyzing the whole dump.

1. Index every accepted game by byte offset. No Stockfish.
2. Keep a tail-first sample: 4,000 sides per 200-Elo bin, 4 games per player.
3. Run Stockfish only on those games, in parallel, and resume if you stop it.

Upload ``games_sample.parquet`` and ``sample_meta.json`` to Colab and run
``notebooks/train_site.ipynb``.

Example
-------
python -m src.pipeline --input path/to/lichess.pgn --output-dir data --workers 4
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.analyze_sample import analyze_games, default_workers, is_complete as analyze_complete
from src.engine import ensure_stockfish
from src.index_headers import index_pgn, is_complete as index_complete
from src.sample import write_sample
from src.schema import (
    DEFAULT_DEPTH,
    DEFAULT_SEED,
    ELO_MAX,
    ELO_MIN,
    GAMES_PER_BIN,
    PER_PLAYER_CAP,
    THIN_BIN_PLAYERS,
)


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index a PGN, sample rating bins, then run Stockfish.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pool", default="classical")
    parser.add_argument("--elo-min", type=int, default=ELO_MIN)
    parser.add_argument("--elo-max", type=int, default=ELO_MAX)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--games-per-bin", type=int, default=GAMES_PER_BIN)
    parser.add_argument("--per-player", type=int, default=PER_PLAYER_CAP)
    parser.add_argument("--thin-players", type=int, default=THIN_BIN_PLAYERS)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash", type=int, default=128, dest="hash_mb")
    parser.add_argument("--stockfish", default=None)
    parser.add_argument("--no-stockfish", action="store_true", help="Behavior features only, for tests.")
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="Delete the index, the sample, and the feature table.")
    parser.add_argument("--fresh-sample", action="store_true", help="Redraw the sample and re-analyze. Keep the index.")
    parser.add_argument("--checkpoint-seconds", type=float, default=30.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "header_index.parquet"
    manifest_path = output_dir / "sample_manifest.parquet"
    sample_meta_path = output_dir / "sample_meta.json"
    games_path = output_dir / "games_sample.parquet"

    if args.fresh:
        for path in (
            index_path,
            index_path.with_name("header_index_meta.json"),
            index_path.parent / "header_index_ckpt",
            manifest_path,
            sample_meta_path,
            games_path,
            games_path.with_name("games_sample_meta.json"),
            games_path.parent / "games_sample_ckpt",
        ):
            _remove(path)
    elif args.fresh_sample:
        for path in (
            manifest_path,
            sample_meta_path,
            games_path,
            games_path.with_name("games_sample_meta.json"),
            games_path.parent / "games_sample_ckpt",
        ):
            _remove(path)

    needs_engine = not args.no_stockfish and not args.index_only and not args.sample_only
    if needs_engine and not analyze_complete(games_path):
        ensure_stockfish(args.stockfish or "auto")

    if not index_complete(index_path):
        state = index_pgn(
            args.input,
            index_path,
            pool=args.pool,
            elo_min=args.elo_min,
            elo_max=args.elo_max,
            fresh=False,
            checkpoint_seconds=args.checkpoint_seconds,
        )
        if not state.get("completed"):
            raise SystemExit(2)
    else:
        print(f"[pipeline] index ready -> {index_path}", flush=True)

    if args.index_only:
        print("[pipeline] index only, stopping before the sample", flush=True)
        return

    if not manifest_path.exists() or not sample_meta_path.exists():
        write_sample(
            index_path,
            manifest_path,
            sample_meta_path,
            seed=args.seed,
            games_per_bin=args.games_per_bin,
            per_player=args.per_player,
            thin_players=args.thin_players,
        )
    else:
        print(
            f"[pipeline] using {manifest_path.name}. Pass --fresh-sample to redraw the bins.",
            flush=True,
        )

    if args.sample_only:
        print("[pipeline] sample only, stopping before Stockfish", flush=True)
        return

    if analyze_complete(games_path):
        print(f"[pipeline] features ready -> {games_path}", flush=True)
    else:
        binary = None if args.no_stockfish else ensure_stockfish(args.stockfish or "auto")
        analyze_games(
            args.input,
            manifest_path,
            games_path,
            workers=args.workers,
            stockfish=binary,
            depth=args.depth,
            threads=args.threads,
            hash_mb=args.hash_mb,
            fresh=False,
        )

    print("[pipeline] complete", flush=True)
    print(f"[pipeline] upload {games_path.name} and {sample_meta_path.name} to Drive MyDrive/DATA", flush=True)
    print("[pipeline] train with notebooks/train_site.ipynb", flush=True)


if __name__ == "__main__":
    main()
