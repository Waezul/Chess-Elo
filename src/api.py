"""HTTP API for the eight-game Elo estimate.

Run it with the model folders in place:

    python -m src.api

Then the page calls ``http://127.0.0.1:8000``. ``HOST``, ``PORT``, ``MODEL_DIR``,
``WORKERS``, ``MAX_WAITING``, and ``JOB_TTL_SECONDS`` override the defaults.
Pastes stay in memory only until the job finishes, and they are not logged.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import chess.engine
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.catalog import POOL_ORDER, Catalog, load_catalog
from src.checklist import check_pgn
from src.engine import EngineSession, ensure_stockfish
from src.estimate import stockfish_estimator
from src.jobs import JobRunner

MAX_BODY = 2 * 1024 * 1024

MESSAGES = {
    "wrong_count": "Select exactly the number of games this pool requires.",
    "pool_unavailable": "That time control is not published yet.",
    "empty_pgn": "Paste a PGN first.",
    "subject_required": "Enter the name you played under.",
    "bad_indexes": "Tick only accepted games from the checklist, each once.",
    "bad_request": "That request was malformed. Reload the page and try again.",
    "body_too_large": "That file is over 2 MB.",
    "unknown_pool": "That time control is not published.",
    "unknown_job": "That read expired. Start it again.",
    "busy": "The engine is busy. Wait a moment and try again.",
    "engine_down": "The engine is down. The checklist still works.",
    "unreadable": "One of those games could not be scored. Pick another eight.",
}


class ApiError(Exception):
    def __init__(self, status: int, error: str) -> None:
        self.status = status
        self.error = error
        super().__init__(error)


def _fail(error: str, status: int) -> ApiError:
    return ApiError(status, error)


def _model_dir() -> Path:
    override = os.environ.get("MODEL_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[1] / "model"


def _engine_binary() -> str | None:
    try:
        binary = ensure_stockfish(os.environ.get("STOCKFISH_PATH") or "auto")
        session = EngineSession(binary, 1, 1, 16, 8)
        session.close()
    except (SystemExit, OSError, chess.engine.EngineError):
        return None
    return binary


def create_app(catalog: Catalog, runner: JobRunner) -> FastAPI:
    app = FastAPI(title="Chess Elo")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.state.catalog = catalog
    app.state.runner = runner

    @app.exception_handler(ApiError)
    async def _api_error(_request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content={"error": exc.error, "message": MESSAGES[exc.error]})

    @app.get("/v1/health")
    def health() -> dict:
        return {
            "ok": True,
            "engine": "ready" if runner.ready else "down",
            "pools": catalog.available_ids(),
        }

    @app.get("/v1/pools")
    def pools() -> dict:
        return catalog.list_summaries()

    @app.get("/v1/pools/{pool}")
    def pool_card(pool: str) -> dict:
        name = _known_pool(pool)
        return catalog.card(name)

    @app.post("/v1/games:check")
    async def games_check(request: Request) -> dict:
        payload = await _payload(request)
        pool = _available_pool(catalog, payload.get("pool"))
        subject = _subject(payload.get("subject"))
        pgn = _pgn(payload.get("pgn"))
        return {
            "pool": pool,
            "games_required": catalog.summary(pool)["games_required"],
            "games": check_pgn(pgn, subject, pool),
        }

    @app.post("/v1/estimates")
    async def estimates(request: Request) -> JSONResponse:
        payload = await _payload(request)
        pool = _available_pool(catalog, payload.get("pool"))
        subject = _subject(payload.get("subject"))
        pgn = _pgn(payload.get("pgn"))
        indexes = _indexes(payload.get("game_indexes"), check_pgn(pgn, subject, pool), catalog.summary(pool)["games_required"])
        try:
            job = runner.submit(pool, pgn, subject, indexes)
        except RuntimeError as exc:
            code = str(exc)
            status = 429 if code == "busy" else 503
            raise _fail(code if code in MESSAGES else "engine_down", status) from exc
        return JSONResponse(status_code=202, content=job)

    @app.get("/v1/estimates/{job_id}")
    def estimate_status(job_id: str) -> dict:
        job = runner.get(job_id)
        if job is None:
            raise _fail("unknown_job", 404)
        return job

    return app


def _known_pool(raw: object) -> str:
    if not isinstance(raw, str):
        raise _fail("bad_request", 400)
    pool = raw.strip().lower()
    if pool not in POOL_ORDER:
        raise _fail("unknown_pool", 404)
    return pool


def _available_pool(catalog: Catalog, raw: object) -> str:
    pool = _known_pool(raw)
    if not catalog.is_available(pool):
        raise _fail("pool_unavailable", 400)
    return pool


def _subject(raw: object) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise _fail("subject_required", 400)
    return raw.strip()


def _pgn(raw: object) -> str:
    if not isinstance(raw, str):
        raise _fail("bad_request", 400)
    if not raw.strip():
        raise _fail("empty_pgn", 400)
    if len(raw.encode("utf-8")) > MAX_BODY:
        raise _fail("body_too_large", 413)
    return raw


def _indexes(raw: object, games: list[dict], required: int) -> list[int]:
    if not isinstance(raw, list):
        raise _fail("bad_request", 400)
    if len(raw) != required:
        raise _fail("wrong_count", 400)
    accepted = {game["index"] for game in games if game["accepted"]}
    seen = set()
    for index in raw:
        if isinstance(index, bool) or not isinstance(index, int) or index in seen or index not in accepted:
            raise _fail("bad_indexes", 400)
        seen.add(index)
    return list(raw)


async def _payload(request: Request) -> dict:
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > MAX_BODY:
        raise _fail("body_too_large", 413)
    raw = await request.body()
    if len(raw) > MAX_BODY:
        raise _fail("body_too_large", 413)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        raise _fail("bad_request", 400) from None
    if not isinstance(payload, dict):
        raise _fail("bad_request", 400)
    return payload


def create_production_app() -> FastAPI:
    catalog = load_catalog(_model_dir())
    binary = _engine_binary()
    workers = int(os.environ.get("WORKERS", "2"))
    runner = JobRunner(
        estimate_fn=stockfish_estimator(catalog, binary) if binary else (lambda *_args: None),
        workers=workers,
        max_waiting=int(os.environ.get("MAX_WAITING", "8")),
        ttl_seconds=float(os.environ.get("JOB_TTL_SECONDS", "600")),
        ready=binary is not None,
        messages=MESSAGES,
    )
    runner.start()
    print(
        f"[api] pools={','.join(catalog.available_ids())} engine={'ready' if runner.ready else 'down'} workers={workers}",
        flush=True,
    )
    return create_app(catalog, runner)


def main() -> None:
    import logging

    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(create_production_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
