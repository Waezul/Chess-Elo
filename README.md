# Chess Elo

HTTP API that estimates a Lichess player's Elo from eight rated games. Paste a PGN, pick the subject name, run Stockfish, get a 10–90% band around a median guess.

JSON only. No API key. `Content-Type: application/json; charset=utf-8`. CORS allows any origin. Body limit 2 MiB (`413` if larger).

## Install and run

```text
pip install -r requirements.txt
```

Stockfish must be on `PATH`, or set `STOCKFISH_PATH`. On Windows, training helpers can download a build into gitignored `tools/stockfish/`.

```text
set HOST=127.0.0.1
set PORT=8000
python -m src.api
```

Env vars: `HOST`, `PORT`, `MODEL_DIR`, `STOCKFISH_PATH`, `WORKERS`, `MAX_WAITING`, `JOB_TTL_SECONDS`.

For a public host (e.g. Render):

```text
HOST=0.0.0.0 PORT=$PORT STOCKFISH_PATH=$(which stockfish) python -m src.api
```

Ship the `model/` folders with the process. Free hosts sleep when idle; Stockfish jobs are slow under demo load.

## API

Base URL locally: `http://127.0.0.1:8000`

Call order:

1. `GET /v1/health` — process up; `engine` is `ready` or `down`
2. `GET /v1/pools` — pool list (`classical`, `rapid`, `blitz`, `bullet`; some may be unavailable)
3. `GET /v1/pools/{pool}` — card for one pool (metrics, revision, games required)
4. `POST /v1/games:check` — validate PGN + subject; returns per-game accept/refuse
5. `POST /v1/estimates` — start a job with the same `pgn` / `subject` and exactly `games_required` accepted indexes
6. `GET /v1/estimates/{id}` — poll until `done` or `failed` (about once per second; drop after ~90s)

Checklist still works when `engine` is `down`; estimates need Stockfish.

### Check

```json
POST /v1/games:check
{
  "pool": "rapid",
  "subject": "YourLichessName",
  "pgn": "[Event \"...\"]\n..."
}
```

### Estimate

```json
POST /v1/estimates
{
  "pool": "rapid",
  "subject": "YourLichessName",
  "pgn": "[Event \"...\"]\n...",
  "game_indexes": [0, 1, 2, 3, 4, 5, 6, 7]
}
```

Response starts as `{ "id": "...", "status": "queued" }` (or `running`). When `done`, `estimate` includes `q10`, `q50`, `q90` (Elo), plus `revision`, `depth`, `n_games`, and related fields. Ratings are never taken from the PGN headers on purpose.

Errors look like:

```json
{ "error": "wrong_count", "message": "Select exactly the number of games this pool requires." }
```

Common codes: `wrong_count`, `pool_unavailable`, `empty_pgn`, `subject_required`, `bad_indexes`, `body_too_large`, `unknown_pool`, `unknown_job`, `busy`, `engine_down`.

### Models in this repo

| Pool | Revision folder | Notes |
| --- | --- | --- |
| classical | `model/classical8-v1` | eight games, Stockfish depth from the card |
| rapid | `model/rapid8-v2` | eight games |
| rapid (experiment) | `model/rapid8-v3-smoke` | smoke / lab bundle from the v3 notebook |

Live availability is whatever `GET /v1/pools` reports from the loaded catalog.

## Training data

This repo does **not** ship Lichess dumps or parquet. Rebuild from [database.lichess.org](https://database.lichess.org/):

| Pool | Dump | Elo window |
| --- | --- | --- |
| Classical | `lichess_db_standard_rated_2017-02.pgn` | 800–2400 |
| Rapid | `lichess_db_standard_rated_2026-09.pgn` | 800–2400 |

Pipeline: index headers → sample players/games → Stockfish analyze → train (see `src/` and `notebooks/`).

Rapid v3 sample (seed `42`, depth `10`): ~12.6M rapid header rows indexed, ~228k engine games, ~201k train rows / ~8.4k train players, bins 800–2400.

Colab-first: put `src` on `MyDrive/src`, tables under `MyDrive/DATA/` (rapid v3: `MyDrive/DATA/rapid2026/`), open a notebook under `notebooks/` and run top to bottom.

## Layout

- `src/` — API, features, sampling, training
- `model/` — shipped quantile bundles
- `notebooks/` — Colab training notebooks
- `tests/` — unit tests (`PYTHONPATH=. pytest`)
