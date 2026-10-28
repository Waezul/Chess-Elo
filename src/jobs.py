"""In-memory estimate jobs. Pastes are dropped as soon as a job finishes."""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field

from src.estimate import EngineDownError, SelectionError


@dataclass
class Job:
    id: str
    status: str
    pool: str
    pgn: str | None
    subject: str | None
    indexes: list[int]
    created: float
    finished: float | None = None
    estimate: dict | None = None
    error: str | None = None
    message: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def public(self) -> dict:
        with self._lock:
            if self.status == "done":
                return {"id": self.id, "status": "done", "estimate": self.estimate}
            if self.status == "failed":
                return {
                    "id": self.id,
                    "status": "failed",
                    "error": self.error,
                    "message": self.message,
                }
            return {"id": self.id, "status": self.status}

    def take_inputs(self) -> tuple[str, str, str, list[int]] | None:
        with self._lock:
            if self.status != "queued" or self.pgn is None or self.subject is None:
                return None
            self.status = "running"
            return self.pool, self.pgn, self.subject, list(self.indexes)

    def succeed(self, estimate: dict) -> None:
        with self._lock:
            self.status = "done"
            self.estimate = estimate
            self.finished = time.monotonic()
            self.pgn = None
            self.subject = None

    def fail(self, error: str, message: str) -> None:
        with self._lock:
            self.status = "failed"
            self.error = error
            self.message = message
            self.finished = time.monotonic()
            self.pgn = None
            self.subject = None


class JobRunner:
    """A fixed pool of workers and a cap on how many jobs may wait."""

    def __init__(
        self,
        estimate_fn,
        workers: int = 2,
        max_waiting: int = 8,
        ttl_seconds: float = 600,
        ready: bool = True,
        messages: dict[str, str] | None = None,
    ) -> None:
        self.estimate_fn = estimate_fn
        self.workers = workers
        self.max_waiting = max_waiting
        self.ttl_seconds = ttl_seconds
        self.ready = ready
        self.messages = messages or {}
        self._jobs: dict[str, Job] = {}
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        if not self.ready:
            return
        for index in range(self.workers):
            thread = threading.Thread(target=self._loop, name=f"estimate-{index}", daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        for _ in self._threads:
            self._queue.put(None)
        for thread in self._threads:
            thread.join(timeout=2)
        self._threads.clear()

    def submit(self, pool: str, pgn: str, subject: str, indexes: list[int]) -> dict:
        if not self.ready:
            raise RuntimeError("engine_down")
        with self._lock:
            self._purge(time.monotonic())
            waiting = sum(1 for job in self._jobs.values() if job.status == "queued")
            if waiting >= self.max_waiting:
                raise RuntimeError("busy")
            job = Job(
                id="job_" + uuid.uuid4().hex[:12],
                status="queued",
                pool=pool,
                pgn=pgn,
                subject=subject,
                indexes=list(indexes),
                created=time.monotonic(),
            )
            self._jobs[job.id] = job
        self._queue.put(job.id)
        return job.public()

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            self._purge(time.monotonic())
            job = self._jobs.get(job_id)
        if job is None:
            return None
        return job.public()

    def _purge(self, now: float) -> None:
        expired = []
        for job_id, job in self._jobs.items():
            if job.status in {"done", "failed"} and job.finished is not None and now - job.finished > self.ttl_seconds:
                expired.append(job_id)
            elif job.status == "queued" and now - job.created > self.ttl_seconds:
                expired.append(job_id)
        for job_id in expired:
            del self._jobs[job_id]

    def _loop(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None:
                return
            with self._lock:
                job = self._jobs.get(job_id)
            if job is None:
                continue
            inputs = job.take_inputs()
            if inputs is None:
                continue
            pool, pgn, subject, indexes = inputs
            started = time.monotonic()
            try:
                estimate = self.estimate_fn(pool, pgn, subject, indexes)
            except SelectionError as exc:
                job.fail(exc.error, self.messages.get(exc.error, "One of those games could not be scored. Pick another eight."))
                outcome = exc.error
            except EngineDownError:
                self.ready = False
                job.fail("engine_down", self.messages.get("engine_down", "The engine is down. The checklist still works."))
                outcome = "engine_down"
            except Exception:
                logging.exception("[api] estimate failed")
                job.fail("engine_down", self.messages.get("engine_down", "The engine is down. The checklist still works."))
                outcome = "engine_down"
            else:
                job.succeed(estimate)
                outcome = "done"
            print(f"[api] {job_id} {pool} {outcome} {time.monotonic() - started:.1f}s", flush=True)
