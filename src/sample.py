"""Choose a tail-first, player-capped sample from a header index.

The index is one row per accepted game. This module expands it to sides,
fills 800-1200 and 2000-2400 before the middle, and keeps both sides of a
game in the same player split. Stockfish has not run yet.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.aggregate import assign_splits
from src.metrics import bin_index, is_tail
from src.schema import (
    BIN_LABELS,
    DEFAULT_SEED,
    DEFAULT_TEST_SIZE,
    DEFAULT_VAL_SIZE,
    GAMES_PER_BIN,
    PER_PLAYER_CAP,
    THIN_BIN_PLAYERS,
)

INDEX_COLUMNS = ["offset", "game_id", "white_key", "black_key", "white_elo", "black_elo"]
MANIFEST_COLUMNS = [
    "offset",
    "game_id",
    "color",
    "player_key",
    "elo",
    "elo_bin",
    "bin_index",
    "split",
    "thin_bin",
]
# Rough pace from the depth-10 checkpoint on this project, single core.
GAMES_PER_HOUR = 6100


def assert_split_integrity(frame: pd.DataFrame) -> None:
    """A player and a game each belong to one split."""
    if frame.empty or "split" not in frame.columns:
        return
    if "player_key" in frame.columns:
        player_splits = frame.groupby("player_key")["split"].nunique()
        if bool((player_splits > 1).any()):
            raise ValueError("a player is in more than one split")
    if "game_id" in frame.columns:
        game_splits = frame.groupby("game_id")["split"].nunique()
        if bool((game_splits > 1).any()):
            raise ValueError("a game is in more than one split")


def _sides_from_index(index: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in INDEX_COLUMNS if column not in index.columns]
    if missing:
        raise ValueError(f"header index is missing {missing}")
    frames = []
    for color, key, elo, opponent in (
        ("white", "white_key", "white_elo", "black_key"),
        ("black", "black_key", "black_elo", "white_key"),
    ):
        side = pd.DataFrame(
            {
                "offset": index["offset"].to_numpy(),
                "game_id": index["game_id"].astype(str).to_numpy(),
                "color": color,
                "player_key": index[key].astype(str).to_numpy(),
                "opponent_key": index[opponent].astype(str).to_numpy(),
                "elo": index[elo].to_numpy(),
            }
        )
        frames.append(side)
    sides = pd.concat(frames, ignore_index=True)
    sides["bin_index"] = sides["elo"].map(bin_index)
    sides = sides.dropna(subset=["bin_index", "player_key", "game_id"]).copy()
    sides["bin_index"] = sides["bin_index"].astype(int)
    sides["elo_bin"] = sides["bin_index"].map(lambda value: BIN_LABELS[int(value)])
    return sides


def _cap_bin(
    bin_sides: pd.DataFrame,
    rng: np.random.Generator,
    games_per_bin: int,
    per_player: int,
    keep_all: bool,
) -> pd.DataFrame:
    if bin_sides.empty:
        return bin_sides
    order = rng.permutation(len(bin_sides))
    shuffled = bin_sides.iloc[order].reset_index(drop=True)
    if keep_all:
        return shuffled
    counts: dict[str, int] = {}
    positions: list[int] = []
    players = shuffled["player_key"].to_numpy()
    for position, player in enumerate(players):
        taken = counts.get(str(player), 0)
        if taken >= per_player:
            continue
        counts[str(player)] = taken + 1
        positions.append(position)
        if len(positions) >= games_per_bin:
            break
    return shuffled.iloc[positions]


def _drop_cross_split(ordered: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Walk tail-first. The first side claims the game's split; a later clash is dropped."""
    if ordered.empty:
        return ordered, 0
    game_ids = ordered["game_id"].to_numpy()
    splits = ordered["split"].to_numpy()
    keep = np.zeros(len(ordered), dtype=bool)
    claimed: dict[str, str] = {}
    dropped = 0
    for position, (game_id, split) in enumerate(zip(game_ids, splits)):
        previous = claimed.get(str(game_id))
        if previous is None:
            claimed[str(game_id)] = str(split)
            keep[position] = True
        elif previous == str(split):
            keep[position] = True
        else:
            dropped += 1
    return ordered.loc[keep].copy(), dropped


def select_manifest(
    index: pd.DataFrame,
    seed: int = DEFAULT_SEED,
    games_per_bin: int = GAMES_PER_BIN,
    per_player: int = PER_PLAYER_CAP,
    thin_players: int = THIN_BIN_PLAYERS,
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    split_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Return kept sides and a bin report. ``split_map`` overrides the seeded player split."""
    sides = _sides_from_index(index)
    players = sorted(set(sides["player_key"]).union(sides["opponent_key"]))
    if split_map is None:
        mapping = assign_splits(players, seed, test_size, val_size)
    else:
        missing = [key for key in players if key not in split_map]
        if missing:
            raise KeyError(f"split_map is missing {len(missing)} players")
        mapping = split_map
    sides["split"] = sides["player_key"].map(mapping)
    if sides["split"].isna().any():
        raise ValueError("a selected side has no split")

    index_players = sides.groupby("bin_index")["player_key"].nunique()
    index_sides = sides.groupby("bin_index").size()
    thin_flags = {
        index: bool(is_tail(index) and int(index_players.get(index, 0)) < thin_players)
        for index in range(len(BIN_LABELS))
    }

    def _sort_key(index: int) -> tuple[int, int]:
        # Tails before the middle, and the scarcer bin first inside each group.
        return (0 if is_tail(index) else 1, int(index_sides.get(index, 0)), index)

    pieces: list[pd.DataFrame] = []
    for index in sorted(range(len(BIN_LABELS)), key=_sort_key):
        group = sides.loc[sides["bin_index"] == index]
        if group.empty:
            continue
        rng = np.random.default_rng(seed + index * 10007)
        kept = _cap_bin(group, rng, games_per_bin, per_player, thin_flags[index])
        if kept.empty:
            continue
        kept = kept.copy()
        kept["thin_bin"] = thin_flags[index]
        pieces.append(kept)
    ordered = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=MANIFEST_COLUMNS)
    manifest, dropped = _drop_cross_split(ordered)
    if not manifest.empty:
        manifest = manifest.sort_values(["offset", "color"]).reset_index(drop=True)
        manifest["elo"] = manifest["elo"].astype(int)
        manifest["offset"] = manifest["offset"].astype(np.int64)
        manifest["thin_bin"] = manifest["thin_bin"].astype(bool)
    manifest = manifest.reindex(columns=MANIFEST_COLUMNS)
    assert_split_integrity(manifest)

    bin_rows = []
    for index, label in enumerate(BIN_LABELS):
        chosen = manifest.loc[manifest["bin_index"] == index] if len(manifest) else manifest
        bin_rows.append(
            {
                "bin": label,
                "bin_index": index,
                "tail": is_tail(index),
                "thin": thin_flags[index],
                "keep_all": thin_flags[index],
                "index_players": int(index_players.get(index, 0)),
                "index_sides": int(index_sides.get(index, 0)),
                "selected_sides": int(len(chosen)),
                "selected_players": int(chosen["player_key"].nunique()) if len(chosen) else 0,
            }
        )
    engine_games = int(manifest["game_id"].nunique()) if len(manifest) else 0
    meta = {
        "seed": seed,
        "games_per_bin": games_per_bin,
        "per_player": per_player,
        "thin_players": thin_players,
        "test_size": test_size,
        "val_size": val_size,
        "dropped_cross_split_sides": dropped,
        "sides_selected": int(len(manifest)),
        "engine_games": engine_games,
        "assumed_games_per_hour_single_core": GAMES_PER_HOUR,
        "bins": bin_rows,
        "players_by_split": (
            {
                str(key): int(value)
                for key, value in manifest.groupby("split")["player_key"].nunique().to_dict().items()
            }
            if len(manifest)
            else {}
        ),
        "protocol": (
            "Header index, then a tail-first sample: up to "
            f"{games_per_bin} sides per 200-Elo bin, {per_player} per player, "
            f"tail bins under {thin_players} players kept whole. "
            "Players are split first. Both sides of a game share that split. "
            "Stockfish runs only on the selected games."
        ),
    }
    return manifest, meta


def write_sample(
    index_path: Path,
    manifest_path: Path,
    meta_path: Path,
    **kwargs,
) -> dict:
    index = pd.read_parquet(index_path)
    manifest, meta = select_manifest(index, **kwargs)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(manifest_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    hours = meta["engine_games"] / GAMES_PER_HOUR if GAMES_PER_HOUR else float("nan")
    print(
        f"[sample] sides={meta['sides_selected']:,}  games={meta['engine_games']:,}  "
        f"cross-split dropped={meta['dropped_cross_split_sides']:,}",
        flush=True,
    )
    print(
        f"[sample] about {hours:.1f}h of Stockfish on one core at {GAMES_PER_HOUR:,} games/hour. "
        "Workers divide that. Tails were filled before the middle.",
        flush=True,
    )
    for row in meta["bins"]:
        flag = " thin" if row["thin"] else ""
        print(
            f"[sample] {row['bin']:>11}  index_players={row['index_players']:<6}  "
            f"selected_sides={row['selected_sides']:<6}{flag}",
            flush=True,
        )
    print(f"[sample] wrote {manifest_path}", flush=True)
    return meta


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Stratified sample from a header index.")
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--meta", required=True, type=Path)
    parser.add_argument("--games-per-bin", type=int, default=GAMES_PER_BIN)
    parser.add_argument("--per-player", type=int, default=PER_PLAYER_CAP)
    parser.add_argument("--thin-players", type=int, default=THIN_BIN_PLAYERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    write_sample(
        args.index,
        args.manifest,
        args.meta,
        seed=args.seed,
        games_per_bin=args.games_per_bin,
        per_player=args.per_player,
        thin_players=args.thin_players,
    )


if __name__ == "__main__":
    main()
