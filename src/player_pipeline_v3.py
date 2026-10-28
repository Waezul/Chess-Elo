"""Build the rapid v3 manifest, then the Stockfish tables, from the v2 index.

Does not rescan the PGN for headers. Sample-only stops before Stockfish.
Re-run the same command to resume a stopped engine pass.

Example
-------
python -m src.player_pipeline_v3 --output-dir data/rapid2026 --sample-only --train-subsets 3
python -m src.player_pipeline_v3 --input PATH\\lichess.pgn --output-dir data/rapid2026 --workers 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.analyze_sample import analyze_games, is_complete as analyze_complete
from src.engine import ensure_stockfish
from src.index_headers import is_complete as index_complete
from src.player_sample_v3 import write_v3_sample
from src.schema import DEFAULT_DEPTH, DEFAULT_SEED, GAMES_PER_PLAYER, TRAIN_SUBSETS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample v3 subsets and run Stockfish on them.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--train-subsets", type=int, default=TRAIN_SUBSETS)
    parser.add_argument("--games-per-player", type=int, default=GAMES_PER_PLAYER)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash", dest="hash_mb", type=int, default=128)
    parser.add_argument("--stockfish", default="auto")
    parser.add_argument("--sample-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "header_index.parquet"
    v2_manifest = output_dir / "player_manifest.parquet"
    manifest_path = output_dir / "player_manifest_v3.parquet"
    sample_meta_path = output_dir / "player_sample_v3_meta.json"
    games_path = output_dir / "player_games_v3.parquet"
    moves_path = output_dir / "moves_v3.parquet"

    if not index_complete(index_path):
        raise SystemExit(f"header index is not ready: {index_path}. Leave it in place and do not rescan.")
    if not v2_manifest.exists():
        raise SystemExit(f"v2 manifest is not ready: {v2_manifest}")

    if not manifest_path.exists() or not sample_meta_path.exists():
        write_v3_sample(
            index_path,
            v2_manifest,
            manifest_path,
            sample_meta_path,
            train_subsets=args.train_subsets,
            games_per_player=args.games_per_player,
            seed=args.seed,
        )
    else:
        print(
            f"[v3-pipeline] using {manifest_path.name}. Delete it to redraw the subsets.",
            flush=True,
        )

    if args.sample_only:
        meta = json.loads(sample_meta_path.read_text(encoding="utf-8"))
        print(
            f"[v3-pipeline] sample only. hours_at_5_workers={meta['hours_at_5_workers']:.2f} "
            f"dropped={meta['train_players_dropped']:,}",
            flush=True,
        )
        return

    if args.input is None or not args.input.exists():
        raise SystemExit(f"Connect the drive that holds {args.input} and re-run the same command.")

    if analyze_complete(games_path):
        print(f"[v3-pipeline] features ready -> {games_path}", flush=True)
        return

    binary = ensure_stockfish(args.stockfish or "auto")
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
        feature_version=3,
        moves_output=moves_path,
    )
    print(f"[v3-pipeline] games -> {games_path}", flush=True)
    print(f"[v3-pipeline] moves -> {moves_path}", flush=True)


if __name__ == "__main__":
    main()
