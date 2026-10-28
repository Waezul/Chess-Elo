"""v3 sample: keep the v2 validation and test games, and redraw train.

Train players stay in the bin v2 gave them. A game against a validation or
test player is left out, so opponent features cannot see a held-out player.
Up to three disjoint 8-game subsets are taken from whoever still has enough
games. Players who do not are counted and dropped.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.sample import GAMES_PER_HOUR, MANIFEST_COLUMNS, _sides_from_index
from src.schema import TRAIN_SUBSETS

V3_MANIFEST_COLUMNS = [*MANIFEST_COLUMNS, "subset"]


def _bins(v2: pd.DataFrame) -> list[dict]:
    rows = []
    if v2.empty:
        return rows
    for label, group in v2.groupby("elo_bin", sort=True):
        rows.append(
            {
                "bin": str(label),
                "players": int(group["player_key"].nunique()),
                "sides": int(len(group)),
            }
        )
    return rows


def _pools_for_train_players(
    sides: pd.DataFrame,
    train_players: pd.DataFrame,
    held_players: set[str],
) -> dict[tuple[str, str], np.ndarray]:
    """Row positions in ``sides`` for each train player, in file order.

    One pass over the index. The old per-player scan repeated that pass
    thousands of times and printed nothing until it finished.
    """
    player_set = set(train_players["player_key"].astype(str))
    wanted = set(zip(train_players["player_key"].astype(str), train_players["elo_bin"].astype(str)))
    mask = sides["player_key"].isin(player_set) & ~sides["opponent_key"].isin(held_players)
    positions = np.flatnonzero(mask.to_numpy())
    narrowed = sides.loc[mask, ["player_key", "elo_bin"]].copy()
    narrowed["_pos"] = positions
    pools: dict[tuple[str, str], np.ndarray] = {}
    grouped = narrowed.groupby(["player_key", "elo_bin"], sort=False)["_pos"]
    for key, values in grouped:
        label = (str(key[0]), str(key[1]))
        if label not in wanted:
            continue
        pools[label] = values.to_numpy(dtype=np.int64, copy=False)
    return pools


def select_v3_manifest(
    index: pd.DataFrame,
    v2_manifest: pd.DataFrame,
    train_subsets: int = TRAIN_SUBSETS,
    games_per_player: int = 8,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Copy v2 val/test rows and build weighted train subsets from the index."""
    v2 = v2_manifest.copy()
    held_out = v2.loc[v2["split"].astype(str) != "train"].copy()
    held_players = set(held_out["player_key"].astype(str))
    held_out["subset"] = 0

    print(f"[v3-sample] indexing {len(index):,} games", flush=True)
    sides = _sides_from_index(index)
    train_players = (
        v2.loc[v2["split"].astype(str) == "train", ["player_key", "elo_bin", "bin_index", "thin_bin"]]
        .assign(player_key=lambda frame: frame["player_key"].astype(str), elo_bin=lambda frame: frame["elo_bin"].astype(str))
        .drop_duplicates("player_key")
    )
    pools = _pools_for_train_players(sides, train_players, held_players)
    by_bin: dict[int, list] = {}
    for row in train_players.itertuples(index=False):
        by_bin.setdefault(int(row.bin_index), []).append(row)

    kept: list[dict] = []
    dropped = 0
    histogram = {1: 0, 2: 0, 3: 0}
    for bin_index in sorted(by_bin):
        rng = np.random.default_rng(seed + 17 + bin_index * 10007)
        players = sorted(by_bin[bin_index], key=lambda item: str(item.player_key))
        print(f"[v3-sample] bin {bin_index} train_players={len(players):,}", flush=True)
        for row in players:
            player = str(row.player_key)
            pool_at = pools.get((player, str(row.elo_bin)))
            size = 0 if pool_at is None else int(len(pool_at))
            k = min(int(train_subsets), size // int(games_per_player))
            if k <= 0:
                dropped += 1
                continue
            order = rng.permutation(size)
            chosen = pool_at[order][: k * int(games_per_player)]
            if k in histogram:
                histogram[k] += 1
            block = sides.iloc[chosen]
            subsets = np.repeat(np.arange(k), int(games_per_player))
            for subset, side in zip(subsets, block.itertuples(index=False)):
                kept.append(
                    {
                        "offset": side.offset,
                        "game_id": str(side.game_id),
                        "color": str(side.color),
                        "player_key": player,
                        "elo": side.elo,
                        "elo_bin": str(row.elo_bin),
                        "bin_index": int(side.bin_index),
                        "split": "train",
                        "thin_bin": bool(row.thin_bin),
                        "subset": int(subset),
                    }
                )

    train = pd.DataFrame(kept, columns=V3_MANIFEST_COLUMNS)
    held_out = held_out.reindex(columns=V3_MANIFEST_COLUMNS)
    manifest = pd.concat([held_out, train], ignore_index=True) if len(held_out) or len(train) else pd.DataFrame(columns=V3_MANIFEST_COLUMNS)
    engine_games = int(manifest["game_id"].nunique()) if len(manifest) else 0
    train_mask = manifest["split"].astype(str) == "train" if len(manifest) else pd.Series(dtype=bool)
    meta = {
        "train_rows": int(train_mask.sum()) if len(manifest) else 0,
        "train_players": int(manifest.loc[train_mask, "player_key"].nunique()) if len(manifest) else 0,
        "train_players_dropped": int(dropped),
        "subset_histogram": {str(count): int(histogram[count]) for count in (1, 2, 3)},
        "engine_games": engine_games,
        "hours_at_5_workers": float(engine_games) / (GAMES_PER_HOUR * 5),
        "bins": _bins(v2),
        "seed": int(seed),
    }
    return manifest, meta


def write_v3_sample(
    index_path: Path,
    v2_manifest_path: Path,
    manifest_path: Path,
    meta_path: Path,
    **kwargs,
) -> dict:
    print(f"[v3-sample] reading {index_path}", flush=True)
    index = pd.read_parquet(index_path)
    v2 = pd.read_parquet(v2_manifest_path)
    manifest, meta = select_v3_manifest(index, v2, **kwargs)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(manifest_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"[v3-sample] train_players={meta['train_players']:,}  dropped={meta['train_players_dropped']:,}  "
        f"games={meta['engine_games']:,}  hours_at_5_workers={meta['hours_at_5_workers']:.2f}",
        flush=True,
    )
    return meta
