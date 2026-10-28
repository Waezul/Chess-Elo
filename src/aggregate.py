"""Legacy player rollup. The site model trains on game rows from ``src.sample``.

Assign a player split, then roll features up to one row per player.

The split is by player, before any aggregation, so test players contribute
no games to training rows. The target is the median Elo across that player's
games in the extract. Players with fewer than ``--min-games`` games stay in
the game-level table and drop out of the player table. Game count is stored
for that filter and is not a model feature.

Repertoire concentration is the normalized Shannon entropy of ECO family and
of the subject's first move. Dividing by log(n) keeps the value from growing
just because the player has more games.

Example
-------
python -m src.aggregate --games data/games.parquet --output-players data/players.parquet --output-games data/games_split.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.schema import (
    DEFAULT_MIN_GAMES,
    DEFAULT_SEED,
    DEFAULT_TEST_SIZE,
    DEFAULT_VAL_SIZE,
    ENGINE_FEATURES,
    NUMERIC_FEATURES,
)


def assign_splits(
    player_keys: list[str],
    seed: int,
    test_size: float,
    val_size: float,
) -> dict[str, str]:
    keys = np.array(sorted(set(player_keys)))
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n = len(keys)
    if n == 0:
        return {}
    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))
    if n >= 3:
        n_test = min(max(n_test, 1), n - 2)
        n_val = min(max(n_val, 1), n - n_test - 1)
    elif n == 2:
        n_test, n_val = 1, 0
    else:
        n_test, n_val = 0, 0
    split: dict[str, str] = {}
    for index, key in enumerate(keys):
        if index < n_test:
            split[str(key)] = "test"
        elif index < n_test + n_val:
            split[str(key)] = "val"
        else:
            split[str(key)] = "train"
    return split


def normalized_entropy(values: pd.Series) -> float:
    """Shannon entropy divided by log(n). Zero when the repertoire has one move."""
    cleaned = values.dropna().astype(str)
    cleaned = cleaned[cleaned.str.len() > 0]
    n = int(len(cleaned))
    if n <= 1:
        return 0.0
    probs = cleaned.value_counts(normalize=True).to_numpy(dtype=float)
    entropy = float(-(probs * np.log(probs)).sum())
    return float(entropy / np.log(n))


def aggregate_players(games: pd.DataFrame, min_games: int) -> pd.DataFrame:
    numeric = [column for column in [*NUMERIC_FEATURES, *ENGINE_FEATURES] if column in games.columns]
    rows: list[dict] = []
    for player, group in games.groupby("player_key", sort=False):
        splits = group["split"].unique()
        if len(splits) != 1:
            raise ValueError(f"player {player} appears in multiple splits")
        record: dict = {
            "player_key": player,  # id only; not a feature
            "elo": float(group["elo"].median()),  # label: typical Elo across their games
            "split": str(splits[0]),  # train, val, or test; assigned before this rollup
            "n_games": int(len(group)),  # filter only; not given to the model
            "result_mean": float(group["result_subject"].mean()),  # win rate; ablation only
            "opening_entropy": normalized_entropy(group["eco_family"]),  # 0 = one opening family
            "first_move_entropy": normalized_entropy(group["first_move"]),  # 0 = always the same first move
        }
        for column in numeric:
            present = group[column].notna()
            # Mean is the player's habit. Std is how much that habit varies from game to game.
            record[f"{column}_mean"] = float(group[column].mean()) if present.any() else np.nan
            record[f"{column}_std"] = float(group[column].std(ddof=0)) if present.any() else np.nan
        rows.append(record)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.loc[frame["n_games"] >= min_games].reset_index(drop=True)


def attach_engine(games: pd.DataFrame, engine_path: Path | None) -> pd.DataFrame:
    if engine_path is None:
        return games
    engine = pd.read_parquet(engine_path)
    engine = engine.drop_duplicates(["game_id", "color"])
    merged = games.merge(engine, on=["game_id", "color"], how="left", suffixes=("", "_engine"))
    for column in ENGINE_FEATURES:
        incoming = f"{column}_engine"
        if incoming not in merged.columns:
            continue
        if column in merged.columns:
            merged[column] = merged[column].fillna(merged[incoming])
        else:
            merged[column] = merged[incoming]
        merged = merged.drop(columns=[incoming])
    return merged


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Player split and player-level aggregation.")
    parser.add_argument("--games", required=True, type=Path)
    parser.add_argument("--output-players", required=True, type=Path)
    parser.add_argument("--output-games", required=True, type=Path)
    parser.add_argument("--engine", type=Path, default=None)
    parser.add_argument("--min-games", type=int, default=DEFAULT_MIN_GAMES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--val-size", type=float, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--meta", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    games = pd.read_parquet(args.games)
    games = attach_engine(games, args.engine)
    split_map = assign_splits(games["player_key"].tolist(), args.seed, args.test_size, args.val_size)
    games = games.copy()
    games["split"] = games["player_key"].map(split_map)
    players = aggregate_players(games, args.min_games)
    args.output_players.parent.mkdir(parents=True, exist_ok=True)
    args.output_games.parent.mkdir(parents=True, exist_ok=True)
    # Columnar parquet the notebook reads. One row per player, and one row per side per game.
    players.to_parquet(args.output_players, index=False)
    games.to_parquet(args.output_games, index=False)
    meta = {
        "seed": args.seed,
        "test_size": args.test_size,
        "val_size": args.val_size,
        "min_games": args.min_games,
        "target": "median elo across the player's games",
        "split": "player, assigned before aggregation",
        "n_games_is_feature": False,
        "game_rows": int(len(games)),
        "player_rows": int(len(players)),
        "players_by_split": players["split"].value_counts().to_dict() if len(players) else {},
        "engine_joined": bool(
            (args.engine is not None)
            or ("cpl_mean" in games.columns and games["cpl_mean"].notna().any())
        ),
    }
    meta_path = args.meta or args.output_players.with_name("aggregate_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"wrote {len(players)} players and {len(games)} game rows -> {args.output_players}",
        flush=True,
    )


if __name__ == "__main__":
    main()
