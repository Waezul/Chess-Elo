"""Draw 700 players per rating bin, and eight of each player's games.

The header index already exists. This module does not read the PGN. A player
is claimed by one bin only: tails first, and the scarcer tail first. Splits
are assigned inside each bin before games are kept. A game whose two sampled
players fall in different splits is skipped and replaced from that player's
other games.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.aggregate import assign_splits
from src.metrics import is_tail
from src.sample import GAMES_PER_HOUR, MANIFEST_COLUMNS, _sides_from_index, assert_split_integrity
from src.schema import (
    BIN_LABELS,
    DEFAULT_SEED,
    DEFAULT_TEST_SIZE,
    DEFAULT_VAL_SIZE,
    GAMES_PER_PLAYER,
    PLAYERS_PER_BIN,
)


def _eligible(sides: pd.DataFrame, games_per_player: int) -> pd.DataFrame:
    counts = sides.groupby(["bin_index", "player_key"], sort=False).size()
    counts = counts[counts >= games_per_player]
    frame = counts.rename("n_games").reset_index()
    frame["player_key"] = frame["player_key"].astype(str)
    frame["bin_index"] = frame["bin_index"].astype(int)
    return frame


def _bin_order(eligible: pd.DataFrame) -> list[int]:
    totals = eligible.groupby("bin_index").size()

    def sort_key(index: int) -> tuple:
        return (0 if is_tail(index) else 1, int(totals.get(index, 0)), index)

    return sorted(range(len(BIN_LABELS)), key=sort_key)


def _claim_players(
    eligible: pd.DataFrame,
    players_per_bin: int,
    seed: int,
    require_all_bins: bool,
) -> tuple[dict[int, list[str]], dict[int, list[str]]]:
    claimed: set[str] = set()
    chosen: dict[int, list[str]] = {}
    reserves: dict[int, list[str]] = {}
    for index in _bin_order(eligible):
        pool = eligible.loc[eligible["bin_index"] == index, "player_key"]
        pool = sorted({str(player) for player in pool if str(player) not in claimed})
        if len(pool) < players_per_bin:
            if require_all_bins or pool:
                raise RuntimeError(
                    f"{BIN_LABELS[index]} has {len(pool)} players with enough games, "
                    f"and the sample asks for {players_per_bin}."
                )
            continue
        pool_frame = eligible.loc[
            (eligible["bin_index"] == index) & eligible["player_key"].isin(pool),
            ["player_key", "n_games"],
        ].drop_duplicates("player_key")
        rng = np.random.default_rng(seed + index * 10007)
        order = rng.permutation(len(pool_frame))
        shuffled = pool_frame.iloc[order].sort_values("n_games", ascending=False, kind="mergesort")
        ranked = [str(player) for player in shuffled["player_key"]]
        chosen[index] = ranked[:players_per_bin]
        reserves[index] = ranked[players_per_bin:]
        claimed.update(chosen[index])
    return chosen, reserves


def _game_lists(sides: pd.DataFrame, wanted: set[tuple[str, int]]) -> dict[tuple[str, int], list[tuple]]:
    if not wanted:
        return {}
    keys = pd.DataFrame(list(wanted), columns=["player_key", "bin_index"])
    keys["player_key"] = keys["player_key"].astype(str)
    keys["bin_index"] = keys["bin_index"].astype(int)
    sub = sides.merge(keys, on=["player_key", "bin_index"], how="inner")
    grouped: dict[tuple[str, int], list[tuple]] = {}
    columns = ["offset", "game_id", "color", "elo", "opponent_key"]
    for (player, index), group in sub.groupby(["player_key", "bin_index"], sort=False):
        ordered = group.sort_values("game_id")
        grouped[(str(player), int(index))] = list(ordered[columns].itertuples(index=False, name=None))
    return grouped


def _keeps_contested(player: str, opponent: str, n_player: int, n_opponent: int) -> bool:
    """The player with fewer games in-bin keeps the game. A tie follows the player id."""
    if n_player != n_opponent:
        return n_player < n_opponent
    return player < opponent


def _take_games(
    player: str,
    games: list[tuple],
    split: str,
    mapping: dict[str, str],
    game_counts: dict[str, int],
    games_per_player: int,
    rng: np.random.Generator,
) -> list[tuple] | None:
    if len(games) < games_per_player:
        return None
    order = rng.permutation(len(games))
    kept: list[tuple] = []
    n_player = game_counts.get(player, len(games))
    for position in order:
        game = games[int(position)]
        opponent = str(game[4])
        opponent_split = mapping.get(opponent)
        if opponent_split is not None and opponent_split != split:
            if not _keeps_contested(player, opponent, n_player, game_counts.get(opponent, n_player)):
                continue
        kept.append(game)
        if len(kept) == games_per_player:
            return kept
    return None


def select_player_manifest(
    index: pd.DataFrame,
    seed: int = DEFAULT_SEED,
    players_per_bin: int = PLAYERS_PER_BIN,
    games_per_player: int = GAMES_PER_PLAYER,
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    require_all_bins: bool = True,
    split_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Return eight game-rows for each selected player, plus the sample report."""
    sides = _sides_from_index(index)
    eligible = _eligible(sides, games_per_player)
    chosen, reserves = _claim_players(eligible, players_per_bin, seed, require_all_bins)
    mapping: dict[str, str] = {}
    if split_map is None:
        for index_bin, players in chosen.items():
            mapping.update(assign_splits(players, seed + index_bin * 10007, test_size, val_size))
    else:
        missing = [player for players in chosen.values() for player in players if player not in split_map]
        if missing:
            raise KeyError(f"split_map is missing {len(missing)} players")
        mapping = {str(key): str(value) for key, value in split_map.items()}

    wanted: set[tuple[str, int]] = set()
    for index_bin, players in chosen.items():
        for player in players:
            wanted.add((player, index_bin))
        for player in reserves.get(index_bin, []):
            wanted.add((player, index_bin))
    games = _game_lists(sides, wanted)

    rows: list[dict] = []
    replaced = 0
    blocked = {player for players in chosen.values() for player in players}
    game_counts = {player: len(games.get((player, index_bin), [])) for index_bin, players in chosen.items() for player in players}
    for index_bin, players in chosen.items():
        rng = np.random.default_rng(seed + 17 + index_bin * 10007)
        reserve = list(reserves.get(index_bin, []))
        cursor = 0
        slot = 0
        while slot < len(players):
            player = players[slot]
            picked = _take_games(
                player,
                games.get((player, index_bin), []),
                mapping[player],
                mapping,
                game_counts,
                games_per_player,
                rng,
            )
            if picked is None:
                while cursor < len(reserve) and reserve[cursor] in blocked:
                    cursor += 1
                if cursor >= len(reserve):
                    raise RuntimeError(
                        f"{BIN_LABELS[index_bin]} ran out of players after cross-split games were skipped."
                    )
                nxt = reserve[cursor]
                cursor += 1
                blocked.add(nxt)
                mapping[nxt] = mapping.pop(player)
                game_counts[nxt] = len(games.get((nxt, index_bin), []))
                players[slot] = nxt
                replaced += 1
                continue
            for offset, game_id, color, elo, _opponent in picked:
                rows.append(
                    {
                        "offset": int(offset),
                        "game_id": str(game_id),
                        "color": str(color),
                        "player_key": player,
                        "elo": int(elo),
                        "elo_bin": BIN_LABELS[index_bin],
                        "bin_index": int(index_bin),
                        "split": mapping[player],
                        "thin_bin": False,
                    }
                )
            slot += 1
        chosen[index_bin] = players

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    if len(manifest):
        manifest = manifest.sort_values(["offset", "color"]).reset_index(drop=True)
        manifest["thin_bin"] = manifest["thin_bin"].astype(bool)
    manifest = manifest.reindex(columns=MANIFEST_COLUMNS)
    assert_split_integrity(manifest)

    counts = (
        manifest.groupby("bin_index").size()
        if len(manifest)
        else pd.Series(dtype=int)
    )
    players_selected = (
        manifest.groupby("bin_index")["player_key"].nunique()
        if len(manifest)
        else pd.Series(dtype=int)
    )
    eligible_players = eligible.groupby("bin_index")["player_key"].nunique()
    bin_rows = []
    for index_bin, label in enumerate(BIN_LABELS):
        bin_rows.append(
            {
                "bin": label,
                "bin_index": index_bin,
                "tail": is_tail(index_bin),
                "eligible_players": int(eligible_players.get(index_bin, 0)),
                "selected_players": int(players_selected.get(index_bin, 0)),
                "selected_sides": int(counts.get(index_bin, 0)),
            }
        )
    engine_games = int(manifest["game_id"].nunique()) if len(manifest) else 0
    meta = {
        "seed": seed,
        "players_per_bin": players_per_bin,
        "games_per_player": games_per_player,
        "test_size": test_size,
        "val_size": val_size,
        "replaced_players": int(replaced),
        "sides_selected": int(len(manifest)),
        "engine_games": engine_games,
        "players": int(manifest["player_key"].nunique()) if len(manifest) else 0,
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
            "Existing header index. "
            f"{players_per_bin} players in each 200-Elo bin, "
            f"{games_per_player} games each, tails claimed first. "
            "One player belongs to one bin. Splits are assigned inside the bin. "
            "Among players with enough games, the most active are kept. "
            "When two sampled players in different splits played each other, "
            "only the player with fewer games in the bin keeps that game. "
            "Stockfish runs only on the selected games. "
            "Training averages the eight games and predicts mean Elo."
        ),
    }
    return manifest, meta


def write_player_sample(
    index_path: Path,
    manifest_path: Path,
    meta_path: Path,
    **kwargs,
) -> dict:
    index = pd.read_parquet(index_path)
    manifest, meta = select_player_manifest(index, **kwargs)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(manifest_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    hours = meta["engine_games"] / GAMES_PER_HOUR if GAMES_PER_HOUR else float("nan")
    print(
        f"[player-sample] players={meta['players']:,}  sides={meta['sides_selected']:,}  "
        f"games={meta['engine_games']:,}  replaced={meta['replaced_players']:,}",
        flush=True,
    )
    print(
        f"[player-sample] about {hours:.1f}h of Stockfish on one core at {GAMES_PER_HOUR:,} games/hour. "
        "Workers divide that.",
        flush=True,
    )
    for row in meta["bins"]:
        print(
            f"[player-sample] {row['bin']:>11}  eligible={row['eligible_players']:<6}  "
            f"players={row['selected_players']:<6}  sides={row['selected_sides']:<6}",
            flush=True,
        )
    print(f"[player-sample] wrote {manifest_path}", flush=True)
    return meta
