# Publish trim — design

Date: 2026-10-29  
Repos: `D:\Chess Elo` (backend). Frontend remains `D:\Chess frontend`.

## Intent

Make the Chess Elo backend a public GitHub repo someone can clone to:

1. **Serve** Elo estimates with the shipped models (`python -m src.api`).
2. **Retrain** in Google Colab after fetching the named Lichess dumps.

Deploy for free as **two hosts**: static frontend + Python API. Do not merge the repos.

## Goals

- Strip raw training parquet and other large derived data from the tree (and keep them out of git).
- Document the exact Lichess sources used for classical and rapid.
- Remove spent plans and `*_trained*` notebook copies.
- Polish runnable Colab notebooks so they read like published lab notes (setup first, plain language, less AI-ish gate jargon).
- Add a root README for serve + retrain + deploy pointers.
- Leave unit tests and `src/` intact except where committed dumps were assumed (they already use temp fixtures).

## Non-goals

- No frontend feature work beyond documenting `VITE_API_BASE_URL` for the free static host (frontend already reads that env).
- No monorepo merge.
- No local-first rewrite of notebooks (Colab-first stays).
- No pruning of training modules that runnable notebooks still need.
- No API contract redesign, no Docker unless a host later forces it.
- No paid hosting plans.

## Anti-goals

- Shipping multi-hundred-MB parquet on GitHub.
- Deleting runnable training history (`train_v1`, `train_v2`, `train_site`, `train_player8`, `train_player8_v3`, `train.ipynb`).
- A single combined host that serves UI from FastAPI.

## Approaches considered

| Approach | Summary | Tradeoff |
| --- | --- | --- |
| **Publish trim (chosen)** | Delete data dumps and debris; document sources; polish notebooks; README; two-host free deploy docs | Minimal risk to older runnable notebooks |
| Product reshape | Also prune unused `src`/tests and aggressively rewrite docs/notebooks | Cleaner, easier to break an old notebook import path |
| Archive shelf | Move old work to `archive/` instead of deleting | Safer emotionally, messier for strangers |

**Chosen:** publish trim.

## Constraints

- Backend already binds via `HOST` / `PORT` and loads `MODEL_DIR` / Stockfish via env (`src/api.py`). Free hosts need `0.0.0.0` and `$PORT`.
- CORS currently allows any origin; acceptable for first public cut.
- Frontend (`D:\Chess frontend`) uses `VITE_API_BASE_URL` (default `http://127.0.0.1:8000`).
- Known Lichess dumps from existing metas: classical `lichess_db_standard_rated_2017-02.pgn`; rapid `lichess_db_standard_rated_2026-09.pgn`; Elo window 800–2400.
- Free tiers sleep and are weak for Stockfish; document as demo limits.

## What leaves

- All `*.parquet` under `data/` (and any other large binary sample artifacts in `data/`).
- Notebooks matching `*_trained*.ipynb`.
- Spent planning/progress: `docs/superpowers/plans/`, `.superpowers/` (this design file under `docs/superpowers/specs/` stays).
- After deletion, empty `data/` may remain as a placeholder with a short README pointing at `docs/data.md`, or be omitted entirely if `.gitignore` + docs are enough. **Builder call:** prefer `data/README.md` pointing at `docs/data.md` so Colab instructions that mention `DATA/` stay intuitive.

## What stays

- Runnable notebooks: `train.ipynb`, `train_v1.ipynb`, `train_v2.ipynb`, `train_site.ipynb`, `train_player8.ipynb`, `train_player8_v3.ipynb`.
- All of `src/`, `tests/`, `requirements.txt`.
- `model/` including published cards (`classical8-v1`, `rapid8-v2`) and `rapid8-v3-smoke` as the current experiment record.
- `docs/backend.md`, `docs/frontend.md`.
- Stockfish **source/docs only as needed** — not the binary tree (see Stockfish decisions below).

## Data story

New `docs/data.md` covers:

- Exact dump filenames and pools (classical 2017-02, rapid 2026-09).
- Elo filter 800–2400.
- High-level pipeline: index headers → sample players/games → engine analyze → train.
- Where Colab expects files under Drive (`MyDrive/src`, `MyDrive/DATA/...`) without committing those files.
- Sample-size / seed facts worth preserving from deleted JSON metas (e.g. v3 seed 42, bin layout) — **builder call:** copy the useful fields from current `*_meta.json` into `docs/data.md` before deleting files.

New `.gitignore` at repo root at least blocks:

- `data/**/*.parquet` (prefer ignore all of `data/*` except `data/README.md`)
- `tools/stockfish/` binary tree
- `__pycache__/`, `.pytest_cache/`, `.venv/`, `venv/`, common editor junk

Local parquet may remain on disk for the author’s machine; they must not be committed.

## Notebook polish

Colab-first for every runnable notebook:

1. Short title markdown: what is estimated, which dump/pool, Drive layout, success criteria in plain language.
2. Soften over-formal / AI-ish gate prose; keep real gates and metrics.
3. Setup cells remain Drive mount + pip + path checks; instructions name required uploads.
4. Clear huge noisy outputs when they hurt readability; do not invent run results.
5. Do not relocate training logic out of `src/` into notebooks.

## GitHub / README (backend)

Root `README.md`:

- What the project is (eight-game Elo band estimate).
- Install: `pip install -r requirements.txt`, Stockfish note.
- Serve: `HOST=0.0.0.0 PORT=… python -m src.api` (and env vars already supported).
- Link `docs/backend.md`, `docs/data.md`, notebooks for retrain.
- Note UI lives in a separate repo; free two-host deploy sketch (Render/Fly for API, Pages/Netlify/Vercel for UI with `VITE_API_BASE_URL`).
- Free-tier caveat: cold starts, slow engine jobs.

## Frontend repo (minimal, for publish)

Out of deep scope for code changes in Chess Elo, but required for the two-host story:

- Document in frontend README (or a short deploy note) that production builds set `VITE_API_BASE_URL` to the public API origin.
- No merge with the backend repo.

**Builder call:** if frontend README already covers this, only add a one-line cross-link from the backend README; otherwise add a short deploy subsection on the frontend side when implementing.

## Stockfish decisions (locked)

1. **Free Linux API host:** install Stockfish on the host (apt or an official Linux binary in the deploy image) and set `STOCKFISH_PATH`. Do **not** extend `ensure_stockfish` / Windows zip download for Linux in this trim. Document the env var and a sample install line in the README.
2. **`tools/stockfish` (~110MB Windows tree):** do **not** publish it. Gitignore the binary tree (and/or remove it from tracking). Local Windows keeps working via existing auto-download or a manual binary; clones stay small.

## Hosting

| Piece | Host (free) | Artifact |
| --- | --- | --- |
| Frontend | Cloudflare Pages, Netlify, or Vercel | Vite `dist/` |
| Backend | Render or Fly.io free web service | Python app + host-installed Stockfish + `model/` |

Deploy order: API first (health check with engine ready), then frontend with `VITE_API_BASE_URL`, then smoke `GET /v1/health` and one checklist from the live UI.

**Builder call:** name Render + Cloudflare Pages first in the README as the default free pair; apt/`STOCKFISH_PATH` example for Render.

## Testing / verification

- `pytest` still passes without committed parquet.
- Repo has no tracked `*.parquet` after cleanup.
- Each runnable notebook opens with clear Colab setup prose.
- README commands match `src/api.py` env vars.

## Left to the builder

- Exact `.gitignore` patterns (must cover `data/**/*.parquet` and `tools/stockfish/**` binaries) and whether `data/README.md` is kept (prefer yes).
- How aggressively to trim notebook outputs vs rewrite markdown only.
- Whether tiny JSON metas under `data/` are deleted after their facts are copied into `docs/data.md` (prefer delete after copy so git stays empty of training artifacts).
- Exact Render build/start command lines in the README (must use `HOST=0.0.0.0`, `$PORT`, `STOCKFISH_PATH`).
