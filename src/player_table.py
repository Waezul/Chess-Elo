"""Average eight analyzed games into one player row.

The label is the mean Elo of those games. The rating bin stays the bin the
player was sampled in, even if the mean sits on a boundary. ``n_games`` is
kept for the filter and is not a model feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.schema import ENGINE_FEATURES, GAME_EXTRA, NUMERIC_FEATURES, V3_CLOCK, V3_GAME_FEATURES, V3_SPREAD_SOURCES

MEAN_FEATURES = [*GAME_EXTRA, *NUMERIC_FEATURES, *ENGINE_FEATURES]


def average_games(games: pd.DataFrame) -> pd.DataFrame:
    """One row per player: the mean of every feature column that is present."""
    frame = games.copy()
    if "player_key" not in frame.columns:
        frame["player_key"] = "player"
    features = [column for column in MEAN_FEATURES if column in frame.columns]
    rows: list[dict] = []
    for player, group in frame.groupby("player_key", sort=False):
        record: dict = {"player_key": str(player), "n_games": int(len(group))}
        if "elo" in group.columns:
            record["elo"] = float(group["elo"].mean())
        if "split" in group.columns:
            splits = group["split"].astype(str).unique()
            if len(splits) != 1:
                raise ValueError(f"player {player} is in more than one split")
            record["split"] = str(splits[0])
        if "elo_bin" in group.columns:
            bins = group["elo_bin"].astype(str).unique()
            if len(bins) != 1:
                raise ValueError(f"player {player} is in more than one rating bin")
            record["elo_bin"] = str(bins[0])
        record["thin_bin"] = False
        for column in features:
            record[column] = float(group[column].mean())
        rows.append(record)
    return pd.DataFrame(rows)


def average_games_v3(games: pd.DataFrame) -> pd.DataFrame:
    """One row per player and subset. Spread is across that subset's games.

    ``weight`` is ``1 / k`` where ``k`` is how many subsets that player has,
    so a player with three chunks does not count three times.
    """
    frame = games.copy()
    if "player_key" not in frame.columns:
        frame["player_key"] = "player"
    if "subset" not in frame.columns:
        frame["subset"] = 0
    subset_count = frame.groupby("player_key")["subset"].nunique()
    rows: list[dict] = []
    for (player, subset), group in frame.groupby(["player_key", "subset"], sort=False):
        record: dict = {
            "player_key": str(player),
            "subset": int(subset),
            "n_games": int(len(group)),
            "weight": 1.0 / float(subset_count.loc[player]),
        }
        if "elo" in group.columns:
            record["elo"] = float(group["elo"].mean())
        if "split" in group.columns:
            splits = group["split"].astype(str).unique()
            if len(splits) != 1:
                raise ValueError(f"player {player} subset {subset} is in more than one split")
            record["split"] = str(splits[0])
        if "elo_bin" in group.columns:
            bins = group["elo_bin"].astype(str).unique()
            if len(bins) != 1:
                raise ValueError(f"player {player} subset {subset} is in more than one rating bin")
            record["elo_bin"] = str(bins[0])
        for column in V3_GAME_FEATURES:
            series = group[column] if column in group.columns else pd.Series(dtype=float)
            numeric = pd.to_numeric(series, errors="coerce")
            record[column] = float(numeric.mean()) if numeric.notna().any() else float("nan")
        for source in V3_SPREAD_SOURCES:
            series = group[source] if source in group.columns else pd.Series(dtype=float)
            clean = pd.to_numeric(series, errors="coerce").dropna()
            if clean.empty:
                record[f"{source}_std"] = float("nan")
                record[f"{source}_min"] = float("nan")
                record[f"{source}_max"] = float("nan")
            else:
                record[f"{source}_std"] = float(clean.std(ddof=0))
                record[f"{source}_min"] = float(clean.min())
                record[f"{source}_max"] = float(clean.max())
        rows.append(record)
    return pd.DataFrame(rows)


def drop_clocks(players: pd.DataFrame, rate: float, seed: int) -> pd.DataFrame:
    """Blank the clock block on a seeded share of rows. The input frame is not changed."""
    out = players.copy()
    count = int(round(float(rate) * len(out)))
    if count <= 0 or out.empty:
        return out
    chosen = np.random.default_rng(seed).choice(len(out), size=min(count, len(out)), replace=False)
    labels = out.index[chosen]
    for column in V3_CLOCK[1:]:
        if column in out.columns:
            out.loc[labels, column] = np.nan
    if "has_clock" in out.columns:
        out.loc[labels, "has_clock"] = 0.0
    return out


def strip_clocks(players: pd.DataFrame) -> pd.DataFrame:
    """Blank every clock feature. Used to score a paste that has no clocks."""
    return drop_clocks(players, 1.0, seed=0)
