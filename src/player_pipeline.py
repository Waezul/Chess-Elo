"""Build the eight-game player table from the header index already on disk.

Does not read the PGN until Stockfish runs, and does not rebuild the index.
Does not touch ``games_sample.parquet``.

Example
-------
python -m src.player_pipeline --input path/to/lichess.pgn --output-dir data --workers 4
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from src.analyze_sample import analyze_games, default_workers, is_complete as analyze_complete
from src.engine import ensure_stockfish
from src.index_headers import is_complete as index_complete
from src.player_sample import write_player_sample
from src.player_table import average_games
from src.schema import DEFAULT_DEPTH, DEFAULT_SEED, GAMES_PER_PLAYER, PLAYERS_PER_BIN


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample eight games per player and run Stockfish.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--players-per-bin", type=int, default=PLAYERS_PER_BIN)
    parser.add_argument("--games-per-player", type=int, default=GAMES_PER_PLAYER)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash", dest="hash_mb", type=int, default=128)
    parser.add_argument("--stockfish", default="auto")
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument("--fresh-sample", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.index or (output_dir / "header_index.parquet")
    manifest_path = output_dir / "player_manifest.parquet"
    sample_meta_path = output_dir / "player_sample_meta.json"
    games_path = output_dir / "player_games.parquet"
    players_path = output_dir / "players_8.parquet"

    if not index_complete(index_path):
        raise SystemExit(f"header index is not ready: {index_path}. Leave it in place and do not rescan.")

    if args.fresh_sample:
        for path in (
            manifest_path,
            sample_meta_path,
            games_path,
            games_path.with_name("player_games_meta.json"),
            games_path.parent / "player_games_ckpt",
            players_path,
        ):
            _remove(path)

    if not manifest_path.exists() or not sample_meta_path.exists():
        write_player_sample(
            index_path,
            manifest_path,
            sample_meta_path,
            seed=args.seed,
            players_per_bin=args.players_per_bin,
            games_per_player=args.games_per_player,
        )
    else:
        print(
            f"[player-pipeline] using {manifest_path.name}. Pass --fresh-sample to redraw.",
            flush=True,
        )

    if args.sample_only:
        print("[player-pipeline] sample only, stopping before Stockfish", flush=True)
        return

    if analyze_complete(games_path):
        print(f"[player-pipeline] features ready -> {games_path}", flush=True)
    else:
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
        )

    games = pd.read_parquet(games_path)
    players = average_games(games)
    full = int((players["n_games"] == args.games_per_player).sum()) if len(players) else 0
    players = players.loc[players["n_games"] == args.games_per_player].reset_index(drop=True)
    players.to_parquet(players_path, index=False)
    meta = json.loads(sample_meta_path.read_text(encoding="utf-8"))
    meta["player_rows"] = int(len(players))
    meta["player_rows_with_full_games"] = full
    sample_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[player-pipeline] averaged rows={len(players):,} -> {players_path}", flush=True)
    print("[player-pipeline] upload player_games.parquet and player_sample_meta.json to Drive MyDrive/DATA", flush=True)
    print("[player-pipeline] train with notebooks/train_player8.ipynb", flush=True)


if __name__ == "__main__":
    main()
