"""Stockfish metrics on the subject's moves only.

Used while extracting games, and also as a standalone pass that writes a
parquet keyed by ``game_id`` and ``color``. Centipawn loss uses a white-POV
evaluation clamped to +/- 1000, including mate scores. A mistake is a loss
of at least 100 cp and a blunder is at least 200 cp. Positions are cached
by FEN.

Example
-------
python -m src.engine --games data/games.parquet --pgn games.pgn --output data/engine.parquet
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import shutil
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Callable

import chess
import chess.engine
import chess.pgn
import numpy as np
import pandas as pd

from src.features import game_id_from_site
from src.pgn_stream import iter_records, open_pgn_text, parse_headers
from src.schema import BLUNDER_CP, DEFAULT_DEPTH, ENGINE_FEATURES, MATE_CP, MISTAKE_CP

AnalyseFn = Callable[[chess.Board], tuple[int, str | None]]


class FenCache:
    def __init__(self, maxsize: int) -> None:
        self.maxsize = maxsize
        self._data: OrderedDict[str, tuple[int, str | None]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, fen: str) -> tuple[int, str | None] | None:
        value = self._data.get(fen)
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        self._data.move_to_end(fen)
        return value

    def put(self, fen: str, value: tuple[int, str | None]) -> None:
        self._data[fen] = value
        self._data.move_to_end(fen)
        if self.maxsize and len(self._data) > self.maxsize:
            self._data.popitem(last=False)


def cp_white_from_score(score: chess.engine.PovScore, mate_cp: int = MATE_CP) -> int:
    white = score.white()
    if white.is_mate():
        mate = white.mate()
        cp = mate_cp if mate and mate > 0 else -mate_cp
    else:
        raw = white.score(mate_score=mate_cp)
        cp = 0 if raw is None else int(raw)
    return max(-mate_cp, min(mate_cp, cp))


def summarize_moves(
    losses: list[float],
    evals: list[float],
    best_flags: list[bool],
    mistake_cp: int = MISTAKE_CP,
    blunder_cp: int = BLUNDER_CP,
) -> dict[str, float]:
    loss_arr = np.asarray(losses, dtype=float)
    eval_arr = np.asarray(evals, dtype=float)
    best_arr = np.asarray(best_flags, dtype=float)
    return {
        "cpl_mean": float(loss_arr.mean()),  # average accuracy; similar Elos tend to share this
        "cpl_median": float(np.median(loss_arr)),  # typical move, robust to a single blunder
        "eval_volatility": float(eval_arr.std(ddof=0)),  # how unstable the position was on their moves
        "mistake_rate": float((loss_arr >= mistake_cp).mean()),  # fraction of serious errors
        "blunder_rate": float((loss_arr >= blunder_cp).mean()),  # fraction of large errors
        "best_move_rate": float(best_arr.mean()),  # fraction of engine-best moves
    }


def win_percent(cp_mover: float) -> float:
    """Win percent from the mover's point of view. 0 cp is 50."""
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp_mover)) - 1)


def move_accuracy(wpl: float) -> float:
    """Lichess move accuracy from win-percent loss, clamped to 0–100."""
    raw = 103.1668100711649 * math.exp(-0.04354415386753951 * wpl) - 3.166924740191411
    return max(0.0, min(100.0, raw))


def _san_of_uci(board: chess.Board, uci: str | None) -> str | None:
    if not uci:
        return None
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        return None
    if move not in board.legal_moves:
        return None
    return board.san(move)


def collect_move_records(
    game: chess.pgn.Game,
    analyse_position: AnalyseFn,
) -> list[dict]:
    """One record per mainline move, both colors. Empty on EngineError.

    The Stockfish walk is :func:`src.moves.analyze_moves`. This adds the SAN
    and castling fields the dashboard reads, from that same table.
    """
    from src.moves import analyze_moves

    table = analyze_moves(game, analyse_position, "")
    if not table:
        return []
    board = game.board()
    records = []
    for row, node in zip(table, game.mainline(), strict=True):
        move = node.move
        stm_is_white = board.turn == chess.WHITE
        played_san = board.san(move)
        best_san = _san_of_uci(board, row["best_uci"])
        castled = board.is_castling(move)
        kingside = board.is_kingside_castling(move) if castled else False
        board.push(move)
        cp_before = int(row["cp_before"])
        cp_after = int(row["cp_after"])
        # Loss from the mover's point of view. A gain is stored as 0.
        loss = (cp_before - cp_after) if stm_is_white else (cp_after - cp_before)
        loss = max(0.0, float(loss))
        mover_before = float(cp_before if stm_is_white else -cp_before)
        mover_after = float(cp_after if stm_is_white else -cp_after)
        records.append(
            {
                "color": row["color"],
                "ply": int(row["ply"]),
                "played": played_san,
                "best": best_san,
                "cp_loss": loss,
                "best_flag": row["best_uci"] is not None and row["uci"] == row["best_uci"],
                "eval": mover_before,
                "cp_before_mover": mover_before,
                "cp_after_mover": mover_after,
                "clock_after": row["clock_after"],
                "castled": castled,
                "kingside": kingside,
            }
        )
    return records


def _summaries(records: list[dict], wanted_colors: set[str]) -> list[dict]:
    buckets: dict[str, dict[str, list]] = {
        color: {"loss": [], "eval": [], "best": []} for color in wanted_colors
    }
    for record in records:
        bucket = buckets.get(record["color"])
        if bucket is None:
            continue
        bucket["loss"].append(record["cp_loss"])
        bucket["eval"].append(record["eval"])
        bucket["best"].append(record["best_flag"])
    rows = []
    for color, bucket in buckets.items():
        if not bucket["loss"]:
            continue
        rows.append({"color": color, **summarize_moves(bucket["loss"], bucket["eval"], bucket["best"])})
    return rows


def analyze_game(
    game: chess.pgn.Game,
    wanted_colors: set[str],
    analyse_position: AnalyseFn,
) -> list[dict[str, float]]:
    """Score only the requested side's moves. ``analyse_position`` returns white-POV cp and best UCI."""
    return _summaries(collect_move_records(game, analyse_position), wanted_colors)


def _tools_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "tools" / "stockfish"


def _local_stockfish() -> str | None:
    root = _tools_dir()
    if not root.exists():
        return None
    exes = [path for path in root.rglob("*.exe") if path.is_file()]
    for path in exes:
        if "avx2" in path.name.lower():
            return str(path)
    return str(exes[0]) if exes else None


def resolve_stockfish(explicit: str | None) -> str:
    if explicit and explicit != "auto":
        if not os.path.isfile(explicit):
            raise SystemExit(f"Stockfish binary not found: {explicit}")
        return explicit
    env = os.environ.get("STOCKFISH_PATH")
    if env and os.path.isfile(env):
        return env
    found = shutil.which("stockfish")
    if found:
        return found
    local = _local_stockfish()
    if local:
        return local
    raise SystemExit(
        "Stockfish not found. Pass --stockfish, set STOCKFISH_PATH, "
        "or run python -m src.pipeline so the official Windows build can be downloaded."
    )


_STOCKFISH_ZIPS = (
    "https://github.com/official-stockfish/Stockfish/releases/download/sf_18/stockfish-windows-x86-64-avx2.zip",
    "https://github.com/official-stockfish/Stockfish/releases/download/sf_17.1/stockfish-windows-x86-64-avx2.zip",
)


def download_stockfish() -> str:
    """Download an official Windows Stockfish build into tools/stockfish."""
    import urllib.error
    import urllib.request
    import zipfile

    dest = _tools_dir()
    dest.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for url in _STOCKFISH_ZIPS:
        zip_path = dest / url.rsplit("/", 1)[-1]
        print(f"downloading {url}", flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": "chess-elo-pipeline"})
        try:
            with urllib.request.urlopen(request, timeout=180) as response, zip_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        except urllib.error.HTTPError as exc:
            last_error = exc
            continue
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(dest)
        found = _local_stockfish()
        if found:
            return found
        last_error = SystemExit(f"No executable inside {zip_path.name}")
    raise SystemExit(f"Could not download Stockfish. Last error: {last_error}")


def ensure_stockfish(explicit: str | None = "auto") -> str:
    try:
        return resolve_stockfish(explicit)
    except SystemExit:
        if explicit and explicit != "auto":
            raise
        return download_stockfish()


def read_game(headers: dict[str, str], move_lines: list[str]) -> chess.pgn.Game | None:
    from src.features import _pgn_quote

    body = "\n".join(
        f'[{key} "{_pgn_quote(headers[key])}"]' for key in ("White", "Black", "Result") if key in headers
    )
    return chess.pgn.read_game(io.StringIO(body + "\n\n" + "\n".join(move_lines) + "\n"))


def engine_pass(
    headers: dict[str, str],
    move_lines: list[str],
    rows: list[dict],
    analyse_position: AnalyseFn | None,
) -> tuple[list[dict], list[dict]]:
    """Add engine columns and return every mainline move from that same walk."""
    for row in rows:
        for key in ENGINE_FEATURES:
            row.setdefault(key, None)
    if analyse_position is None or not rows:
        return rows, []
    game = read_game(headers, move_lines)
    if game is None:
        return rows, []
    records = collect_move_records(game, analyse_position)
    by_color = {
        item["color"]: item for item in _summaries(records, {row["color"] for row in rows})
    }
    for row in rows:
        stats = by_color.get(row["color"])
        if not stats:
            continue
        for key in ENGINE_FEATURES:
            row[key] = stats[key]
    return rows, records


def apply_engine_features(
    headers: dict[str, str],
    move_lines: list[str],
    rows: list[dict],
    analyse_position: AnalyseFn | None,
) -> list[dict]:
    """Add subject-only centipawn columns onto feature rows from the same game."""
    annotated, _records = engine_pass(headers, move_lines, rows, analyse_position)
    return annotated


class EngineSession:
    """One local Stockfish process. Each extract worker owns its own session."""

    def __init__(self, binary: str, depth: int, threads: int, hash_mb: int, cache_size: int) -> None:
        self.cache = FenCache(cache_size)
        self.engine = chess.engine.SimpleEngine.popen_uci(binary)
        try:
            self.engine.configure({"Threads": threads, "Hash": hash_mb})
        except chess.engine.EngineError:
            pass
        info = self.engine.id if isinstance(self.engine.id, dict) else {}
        self.name = info.get("name", "stockfish")
        self.analyse = make_stockfish_analyser(self.engine, depth, self.cache)

    def close(self) -> None:
        self.engine.quit()


def make_stockfish_analyser(engine: chess.engine.SimpleEngine, depth: int, cache: FenCache) -> AnalyseFn:
    limit = chess.engine.Limit(depth=depth)

    def analyse(board: chess.Board) -> tuple[int, str | None]:
        fen = board.fen()
        cached = cache.get(fen)
        if cached is not None:
            return cached
        info = engine.analyse(board, limit)
        score = info["score"]
        pv = info.get("pv") or []
        best = pv[0].uci() if pv else None
        value = (cp_white_from_score(score), best)
        cache.put(fen, value)
        return value

    return analyse


def load_targets(games_path: Path) -> dict[str, set[str]]:
    frame = pd.read_parquet(games_path, columns=["game_id", "color"])
    wanted: dict[str, set[str]] = defaultdict(set)
    for game_id, color in zip(frame["game_id"], frame["color"]):
        wanted[str(game_id)].add(str(color))
    return wanted


def iter_engine_rows(pgn_path: Path, wanted: dict[str, set[str]], analyse_position: AnalyseFn, limit_games: int | None):
    rows: list[dict] = []
    found: set[str] = set()
    if not wanted:
        return rows, {"games_targeted": 0, "games_analyzed": 0, "games_missing": 0}
    text, binary = open_pgn_text(str(pgn_path))
    try:
        for header_lines, move_lines in iter_records(text):
            headers = parse_headers(header_lines)
            game_id = game_id_from_site(headers.get("Site", ""))
            if game_id not in wanted:
                continue
            try:
                game = read_game(headers, move_lines)
            except Exception:
                continue
            if game is None:
                continue
            for row in analyze_game(game, wanted[game_id], analyse_position):
                row["game_id"] = game_id
                rows.append(row)
            found.add(game_id)
            if len(found) % 25 == 0:
                print(f"analyzed {len(found)} / {len(wanted)} games", flush=True)
            if found >= set(wanted) or (limit_games is not None and len(found) >= limit_games):
                break
    finally:
        text.close()
        if binary is not None:
            binary.close()
    missing = len(set(wanted) - found)
    return rows, {"games_targeted": len(wanted), "games_analyzed": len(found), "games_missing": missing}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stockfish centipawn features for subject moves.")
    parser.add_argument("--games", required=True, type=Path)
    parser.add_argument("--pgn", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stockfish", default=None)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash", type=int, default=128, dest="hash_mb")
    parser.add_argument("--cache-size", type=int, default=200_000)
    parser.add_argument("--limit-games", type=int, default=None)
    parser.add_argument("--meta", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    binary = ensure_stockfish(args.stockfish or "auto")
    wanted = load_targets(args.games)
    if args.limit_games is not None:
        keep = list(wanted)[: args.limit_games]
        wanted = {key: wanted[key] for key in keep}
    cache = FenCache(args.cache_size)
    engine = chess.engine.SimpleEngine.popen_uci(binary)
    try:
        try:
            engine.configure({"Threads": args.threads, "Hash": args.hash_mb})
        except chess.engine.EngineError:
            pass
        name = engine.id.get("name", "stockfish") if isinstance(engine.id, dict) else "stockfish"
        analyse = make_stockfish_analyser(engine, args.depth, cache)
        rows, stats = iter_engine_rows(args.pgn, wanted, analyse, limit_games=None)
    finally:
        engine.quit()
    columns = ["game_id", "color", *ENGINE_FEATURES]
    frame = pd.DataFrame(rows, columns=columns)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    meta = {
        "stockfish": name,
        "binary": binary,
        "depth": args.depth,
        "mate_cp": MATE_CP,
        "mistake_cp": MISTAKE_CP,
        "blunder_cp": BLUNDER_CP,
        "threads": args.threads,
        "hash_mb": args.hash_mb,
        "cache_hits": cache.hits,
        "cache_misses": cache.misses,
        "subject_moves_only": True,
        **stats,
        "rows": int(len(frame)),
    }
    meta_path = args.meta or args.output.with_name("engine_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {len(frame)} engine rows -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
