# Training data

This repo does not ship Lichess dumps or derived parquet. Rebuild locally or on Drive, then train from the notebooks under `notebooks/`.

## Lichess dumps

| Pool | Dump file | Elo window |
| --- | --- | --- |
| Classical | `lichess_db_standard_rated_2017-02.pgn` | 800–2400 |
| Rapid | `lichess_db_standard_rated_2026-09.pgn` | 800–2400 |

Download the monthly standard rated databases from [Lichess database](https://database.lichess.org/).

## Pipeline

1. **Index headers** — scan the PGN dump into a header index for the pool and Elo window.
2. **Sample** — draw players / game sides for training, validation, and test.
3. **Analyze** — run Stockfish over the sampled games and write feature tables.
4. **Train** — fit the quantile model in Colab from a notebook in `notebooks/`.

Exact module entry points live in `src/` (`index_headers`, sample / player sample, analyze, train). The notebooks call into that package; they do not redefine the feature lists.

## Sample facts (rapid v3)

Preserved from the local run that produced `rapid8-v3-smoke` / the v3 training tables:

- Sample seed: `42`
- Stockfish depth: `10`
- Feature version: `3`
- Header index rows (classical 2017-02): `2,293,148`
- Header index rows (rapid 2026-09): `12,638,713`
- Rapid v3 engine games analyzed: about `228,038`
- Rapid v3 train rows / players: `201,560` / `8,399`
- Elo bins used 800–1000 … 2200–2400 (1,500 players per bin in the v3 sample layout)

## Colab Drive layout

Notebooks are Colab-first. Before running:

1. Upload the `src` package to `MyDrive/src` so `schema.py` is at `/content/drive/MyDrive/src/schema.py`.
2. Put derived tables under `MyDrive/DATA/` (classical / older player8 paths) or `MyDrive/DATA/rapid2026/` for rapid v3 (`player_games_v3.parquet`, `player_sample_v3_meta.json`, `player_games_v3_meta.json`).
3. Open the matching notebook from this repo and run top to bottom.

Local `data/` on a developer machine is fine for building those tables. Do not commit parquet or large binaries; `.gitignore` ignores the whole `data/` directory.
