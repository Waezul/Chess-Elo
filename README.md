# Chess Elo

Estimate a Lichess player's Elo band from eight rated games. The API reads a PGN, runs Stockfish, and returns a 10–90% interval around a median guess.

The interactive UI lives in a separate frontend repo (`Chess frontend`). This repository is the **backend**: HTTP API, training code, Colab notebooks, and shipped model cards under `model/`.

## Install

```text
pip install -r requirements.txt
```

You need a Stockfish binary on `PATH`, or set `STOCKFISH_PATH` to it. On Windows, the training/pipeline helpers can download an official build into `tools/stockfish/` (that folder is gitignored and not published).

## Serve locally

```text
set HOST=127.0.0.1
set PORT=8000
python -m src.api
```

Optional env vars: `MODEL_DIR`, `STOCKFISH_PATH`, `WORKERS`, `MAX_WAITING`, `JOB_TTL_SECONDS`.

Health and pools: `GET /v1/health`, `GET /v1/pools`. Checklist: `POST /v1/games:check`. Estimate: `POST /v1/estimates`, then poll `GET /v1/estimates/{id}`.

## Retrain

Derived parquet is not in this repo. Rebuild from Lichess monthly dumps:

- Classical: `lichess_db_standard_rated_2017-02.pgn` (Elo 800–2400)
- Rapid: `lichess_db_standard_rated_2026-09.pgn` (Elo 800–2400)

Colab-first: upload `src/` to `MyDrive/src` and tables under `MyDrive/DATA/` (rapid v3 uses `MyDrive/DATA/rapid2026/`).

Training notebooks live under `notebooks/`:

- `train_player8_v3.ipynb` — rapid, eight games (current experiment)
- `train_player8.ipynb` — classical, eight games
- `train_site.ipynb` — classical, one game
- `train.ipynb` / `train_v1.ipynb` / `train_v2.ipynb` — earlier player/game records

## Free deploy (two hosts)

Default pair: **Render** for the API, **Cloudflare Pages** for the UI.

### API (Render)

1. Push this repo to GitHub and create a Render Web Service from it.
2. Install Python deps (`pip install -r requirements.txt`).
3. Install Stockfish on the Linux host (for example `apt-get install -y stockfish`) and set `STOCKFISH_PATH` to that binary.
4. Start with the service port and a public bind:

```text
HOST=0.0.0.0 PORT=$PORT STOCKFISH_PATH=$(which stockfish) python -m src.api
```

Ship the `model/` folders with the service. Confirm `GET /v1/health` returns `ok` and note whether `engine` is `ready` or `down`.

### UI (Cloudflare Pages)

In the frontend repo, build with Vite and set the API origin:

```text
VITE_API_BASE_URL=https://<your-render-service>
npm run build
```

Publish the `dist/` directory. Without `VITE_API_BASE_URL`, the client defaults to `http://127.0.0.1:8000`.

### Limits

Free tiers sleep when idle. Stockfish jobs are CPU-heavy and can be slow or time out under demo load. Fine for a public trial, not for heavy traffic.
