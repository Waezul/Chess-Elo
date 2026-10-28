"""Checklist, model cards, and the /v1 routes. Stockfish is not started here."""

from __future__ import annotations

import json
import threading
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from src.api import MAX_BODY, create_app
from src.catalog import load_catalog
from src.checklist import check_pgn, estimated_seconds, pool_from_clock, pool_from_event
from src.estimate import score_selection
from src.jobs import JobRunner

ROOT = Path(__file__).resolve().parents[1] / "model"
LONG = (
    "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 "
    "7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7"
)
SHORT = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6"


def game(
    event,
    white="Ada",
    black="Bea",
    moves=LONG,
    site="https://lichess.org/abcdefgh",
    time_control="600+0",
    termination="Normal",
    result="1-0",
    variant=None,
    opening="Ruy Lopez",
    date="2026.09.14",
):
    variant_line = f'[Variant "{variant}"]\n' if variant else ""
    return (
        f'[Event "{event}"]\n'
        f'[Site "{site}"]\n'
        f'[Date "{date}"]\n'
        f'[White "{white}"]\n'
        f'[Black "{black}"]\n'
        f'[Result "{result}"]\n'
        '[WhiteElo "1500"]\n'
        '[BlackElo "1600"]\n'
        f'[Opening "{opening}"]\n'
        f'[TimeControl "{time_control}"]\n'
        f"{variant_line}"
        f'[Termination "{termination}"]\n'
        f"\n{moves} {result}\n"
    )


def scripted(_board):
    return 12, None


class ChecklistTests(unittest.TestCase):
    def test_clock_bands_follow_lichess(self):
        self.assertEqual(pool_from_clock("60+0"), "bullet")
        self.assertEqual(pool_from_clock("180+0"), "blitz")
        self.assertEqual(pool_from_clock("600+0"), "rapid")
        self.assertEqual(pool_from_clock("1800+0"), "classical")
        self.assertEqual(estimated_seconds("600+5"), 800)
        self.assertIsNone(pool_from_clock("-"))
        self.assertEqual(pool_from_event("Rated UltraBullet game"), "ultrabullet")
        self.assertEqual(pool_from_event("Rated Bullet game"), "bullet")

    def test_each_refusal_and_an_accepted_rapid_game(self):
        paste = "\n".join(
            [
                game("Rated Rapid game"),
                game("Rated Blitz game", time_control="180+0"),
                game("Rated Correspondence game", time_control="-"),
                game("Rated Rapid game", variant="Chess960"),
                game("Rated Rapid game", white="Cara"),
                game("Rated Rapid game", termination="Time forfeit"),
                game("Rated Rapid game", moves=SHORT),
                game("Live Chess", site="https://www.chess.com/game/live/1"),
                game("Rated game", time_control="600+0"),
                game("Casual Rapid game"),
            ]
        )
        games = check_pgn(paste, "ada", "rapid")
        reasons = [row["reason"] for row in games]
        self.assertEqual(
            reasons,
            [
                "ok",
                "wrong_pool",
                "unrated",
                "variant",
                "name_missing",
                "unfinished",
                "too_short",
                "wrong_pool",
                "ok",
                "unrated",
            ],
        )
        accepted = games[0]
        self.assertTrue(accepted["accepted"])
        self.assertEqual(accepted["color"], "white")
        self.assertEqual(accepted["date"], "2026.09.14")
        self.assertEqual(accepted["opening"], "Ruy Lopez")
        self.assertEqual(accepted["result"], "1-0")
        self.assertGreaterEqual(accepted["moves"], 10)
        self.assertFalse(games[1]["accepted"])
        self.assertIsNone(games[1]["color"])
        dumped = json.dumps(games)
        self.assertNotIn("1500", dumped)
        self.assertNotIn("1600", dumped)
        self.assertNotIn("Elo", dumped)

    def test_event_pool_wins_over_the_clock_and_black_matches(self):
        rapid_clock_classical_event = check_pgn(
            game("Rated Classical game", time_control="600+0"),
            "Ada",
            "rapid",
        )
        self.assertEqual(rapid_clock_classical_event[0]["reason"], "wrong_pool")
        black = check_pgn(game("Rated Rapid game", black="Ada", white="Bea"), "Ada", "rapid")
        self.assertEqual(black[0]["color"], "black")
        self.assertTrue(black[0]["accepted"])


class CatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog(ROOT)

    def test_cards_match_the_scored_notebooks(self):
        classical = self.catalog.card("classical")
        self.assertEqual(classical["revision"], "classical8-v1")
        self.assertEqual(classical["test_players"], 1119)
        self.assertFalse(classical["ship_ready"])
        self.assertEqual(
            classical["summary"],
            [
                "This is a lab estimate. The ship bar was a typical miss of 120 Elo or better, and the test missed by 124.",
                "68% of 1,119 players the model had not seen landed within 150 rating points. The hidden range contained the real rating for 80% of them.",
                "Players from 800 to 1000 were guessed about 121 Elo too high. Players from 2000 to 2200 were guessed about 90 Elo too low.",
            ],
        )
        self.assertEqual(classical["bins"][0]["bin"], "800\u20131000")
        self.assertEqual(classical["bins"][0]["mae"], 145.8)
        self.assertEqual(classical["bins"][3]["bias"], 17.6)
        self.assertEqual(classical["bins"][4]["bias"], 4.7)
        self.assertEqual(classical["bins"][-1]["median_width"], 250)

        rapid = self.catalog.card("rapid")
        self.assertEqual(rapid["revision"], "rapid8-v2")
        self.assertEqual(rapid["test_players"], 2398)
        self.assertEqual(rapid["source_label"], "Lichess rapid, September 2026")
        self.assertEqual(
            rapid["summary"][2],
            "Players from 800 to 1000 were guessed about 110 Elo too high, and players from 1000 to 1200 about 102 too high. Players from 2000 to 2200 were guessed about 80 Elo too low.",
        )
        self.assertEqual(rapid["bins"][2]["median_width"], 580)
        self.assertEqual(rapid["bins"][6]["bias"], -80.1)

    def test_driver_sentences_do_not_say_the_center(self):
        from src.catalog import DRIVER_COPY

        texts = [text for pair in DRIVER_COPY.values() for text in pair.values()]
        self.assertIn("Fewer serious mistakes than usual raised the guess.", texts)
        self.assertIn("Giving away less on a typical move raised the guess.", texts)
        self.assertFalse(any("center" in text.casefold() for text in texts))

    def test_tail_note_follows_the_center(self):
        low = self.catalog.limits("rapid", 980)
        high = self.catalog.limits("classical", 2100)
        middle = self.catalog.limits("classical", 1500)
        self.assertEqual(low["typical_error"], 152)
        self.assertEqual(low["tail_note"], "Players this far below the middle were often guessed high.")
        self.assertEqual(high["tail_note"], "Players this far above the middle were often guessed low.")
        self.assertIsNone(middle["tail_note"])
        self.assertEqual(high["typical_error"], 124)

    def test_eight_games_score_without_stockfish(self):
        paste = "\n".join(game("Rated Rapid game", site=f"https://lichess.org/game{i}") for i in range(8))
        estimate = score_selection(self.catalog, paste, "Ada", "rapid", list(range(8)), scripted)
        self.assertEqual(estimate["revision"], "rapid8-v2")
        self.assertTrue(estimate["lab"])
        self.assertFalse(estimate["ship_ready"])
        self.assertEqual(estimate["n_games"], 8)
        self.assertEqual(estimate["depth"], 10)
        self.assertLessEqual(estimate["q10"], estimate["q50"])
        self.assertLessEqual(estimate["q50"], estimate["q90"])
        self.assertLessEqual(len(estimate["drivers"]), 3)
        self.assertTrue(all(item["text"] and item["direction"] in {"higher", "lower"} for item in estimate["drivers"]))
        self.assertNotIn("1500", json.dumps(estimate))


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog(ROOT)

    def _client(self, runner: JobRunner) -> TestClient:
        return TestClient(create_app(self.catalog, runner))

    def test_health_pools_and_method_card(self):
        runner = JobRunner(lambda *_args: None, workers=0, ready=True)
        with self._client(runner) as client:
            health = client.get("/v1/health")
            pools = client.get("/v1/pools")
            card = client.get("/v1/pools/rapid")
            missing = client.get("/v1/pools/hyperbullet")
            blitz = client.get("/v1/pools/blitz")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["pools"], ["classical", "rapid"])
        self.assertEqual(health.json()["engine"], "ready")
        ids = [row["id"] for row in pools.json()["pools"]]
        self.assertEqual(ids, ["classical", "rapid", "blitz", "bullet"])
        self.assertFalse(pools.json()["pools"][2]["available"])
        self.assertEqual(pools.json()["pools"][2]["unavailable_reason"], "No blitz model is trained yet.")
        self.assertEqual(card.status_code, 200)
        self.assertEqual(card.json()["revision"], "rapid8-v2")
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.json()["error"], "unknown_pool")
        self.assertEqual(blitz.status_code, 200)
        self.assertFalse(blitz.json()["available"])

    def test_check_and_estimate_errors(self):
        runner = JobRunner(lambda *_args: None, workers=0, ready=False)
        paste = game("Rated Rapid game")
        with self._client(runner) as client:
            down = client.get("/v1/health")
            checked = client.post("/v1/games:check", json={"pool": "rapid", "subject": "Ada", "pgn": paste})
            blitz = client.post("/v1/games:check", json={"pool": "blitz", "subject": "Ada", "pgn": paste})
            empty = client.post("/v1/games:check", json={"pool": "rapid", "subject": "Ada", "pgn": "  "})
            nameless = client.post("/v1/games:check", json={"pool": "rapid", "pgn": paste})
            broken = client.post("/v1/games:check", content=b"{", headers={"content-type": "application/json"})
            estimate = client.post(
                "/v1/estimates",
                json={"pool": "rapid", "subject": "Ada", "pgn": paste, "game_indexes": [0]},
            )
        self.assertEqual(down.json()["engine"], "down")
        self.assertEqual(checked.status_code, 200)
        self.assertTrue(checked.json()["games"][0]["accepted"])
        self.assertEqual(blitz.status_code, 400)
        self.assertEqual(blitz.json()["error"], "pool_unavailable")
        self.assertEqual(empty.json()["error"], "empty_pgn")
        self.assertEqual(nameless.json()["error"], "subject_required")
        self.assertEqual(broken.status_code, 400)
        self.assertEqual(broken.json()["error"], "bad_request")
        self.assertEqual(estimate.status_code, 400)
        self.assertEqual(estimate.json()["error"], "wrong_count")

    def test_estimate_job_round_trip_busy_and_expiry(self):
        started = threading.Event()
        release = threading.Event()

        def blocking(pool, _pgn, _subject, _indexes):
            started.set()
            release.wait(2)
            return {
                "pool": pool,
                "revision": "rapid8-v2",
                "ship_ready": False,
                "lab": True,
                "n_games": 8,
                "depth": 10,
                "q10": 1483.2,
                "q50": 1706.4,
                "q90": 1958.7,
                "drivers": [],
                "limits": {"typical_error": 152, "within_150": 0.58, "coverage": 0.80, "tail_note": None},
            }

        runner = JobRunner(blocking, workers=1, max_waiting=1, ttl_seconds=2, ready=True)
        runner.start()
        paste = "\n".join(game("Rated Rapid game") for _ in range(8))
        body = {"pool": "rapid", "subject": "Ada", "pgn": paste, "game_indexes": list(range(8))}
        try:
            with self._client(runner) as client:
                first = client.post("/v1/estimates", json=body)
                self.assertEqual(first.status_code, 202)
                job_id = first.json()["id"]
                for _ in range(50):
                    if client.get(f"/v1/estimates/{job_id}").json()["status"] == "running":
                        break
                    time.sleep(0.02)
                else:
                    self.fail("job never started")
                second = client.post("/v1/estimates", json=body)
                third = client.post("/v1/estimates", json=body)
                self.assertEqual(second.status_code, 202)
                self.assertEqual(third.status_code, 429)
                self.assertEqual(third.json()["error"], "busy")
                duplicate = dict(body)
                duplicate["game_indexes"] = [0, 0, 1, 2, 3, 4, 5, 6]
                rejected = client.post("/v1/estimates", json=duplicate)
                self.assertEqual(rejected.json()["error"], "bad_indexes")
                release.set()
                for _ in range(50):
                    done = client.get(f"/v1/estimates/{job_id}").json()
                    if done["status"] == "done":
                        break
                    time.sleep(0.02)
                else:
                    self.fail(done)
                self.assertEqual(done["estimate"]["q50"], 1706.4)
                self.assertNotIn("pgn", done)
                time.sleep(2.2)
                expired = client.get(f"/v1/estimates/{job_id}")
                self.assertEqual(expired.status_code, 404)
                self.assertEqual(expired.json()["error"], "unknown_job")
        finally:
            release.set()
            runner.stop()

    def test_body_over_2mb_is_rejected(self):
        runner = JobRunner(lambda *_args: None, workers=0, ready=True)
        payload = {"pool": "rapid", "subject": "Ada", "pgn": "x" * (MAX_BODY + 1)}
        with self._client(runner) as client:
            response = client.post("/v1/games:check", json=payload)
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["error"], "body_too_large")


if __name__ == "__main__":
    unittest.main()
