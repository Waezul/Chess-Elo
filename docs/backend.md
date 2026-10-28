# Backend API

The live server for the page in `docs/frontend.md`. This file is the HTTP contract. Screen copy and rounding live in `docs/frontend.md`. Use the `message` strings from this file. They are what the server sends.

Base URL while developing: `http://127.0.0.1:8000`

Start it from the repo root:

```text
python -m src.api
```

`HOST` defaults to `127.0.0.1`, `PORT` to `8000`. JSON only. No API key. Send `Content-Type: application/json; charset=utf-8`. CORS allows any origin. There is no cancel route and no websocket.

The body limit is 2,097,152 bytes (2 MiB), measured on the raw request and again on the `pgn` string. A larger body is `413`.

## Call order

1. `GET /v1/health` and `GET /v1/pools` on load.
2. `GET /v1/pools/{pool}` when the picker changes, and for `/method`.
3. `POST /v1/games:check` 400 ms after the paste or the subject name changes. Skip an empty paste. Keep a request id and ignore a response that is older than the latest edit. Changing pool clears the checklist and the result, then checks again when both fields are non-empty.
4. `POST /v1/estimates` when exactly `games_required` accepted games are ticked. The body is the same `pgn` and `subject` that were checked, plus those indexes.
5. `GET /v1/estimates/{id}` every 1 second. Stop on `done` or `failed`. Stop after 90 seconds, drop the id, and show `The read took too long. Try again.` Cancel drops the id and stops polling. The server keeps working. The next poll of a dropped id is unnecessary.

`engine: "down"` on health disables the estimate button and shows the `engine_down` sentence. The checklist still runs.

## Errors

Every 4xx and 5xx body is:

```json
{ "error": "wrong_count", "message": "Select exactly the number of games this pool requires." }
```

Show `message`.

| HTTP | `error` | `message` |
| --- | --- | --- |
| 400 | `wrong_count` | Select exactly the number of games this pool requires. |
| 400 | `pool_unavailable` | That time control is not published yet. |
| 400 | `empty_pgn` | Paste a PGN first. |
| 400 | `subject_required` | Enter the name you played under. |
| 400 | `bad_indexes` | Tick only accepted games from the checklist, each once. |
| 400 | `bad_request` | That request was malformed. Reload the page and try again. |
| 413 | `body_too_large` | That file is over 2 MB. |
| 404 | `unknown_pool` | That time control is not published. |
| 404 | `unknown_job` | That read expired. Start it again. |
| 429 | `busy` | The engine is busy. Wait a moment and try again. |
| 503 | `engine_down` | The engine is down. The checklist still works. |

A failed job is HTTP 200 with `"status": "failed"` and the same `error` / `message` pair. `unknown_job` is the 404, not a failed job.

Pool ids are `classical`, `rapid`, `blitz`, and `bullet`. The server lowercases them. Any other string is `unknown_pool`. A known pool with `available: false` is `pool_unavailable` on `games:check` and `estimates`. `GET /v1/pools/{pool}` still returns 200 for those, so the method page can explain why.

Validation order on the two POST routes: pool, then subject, then pgn, then (estimates only) `game_indexes`.

## Types

```ts
type PoolId = "classical" | "rapid" | "blitz" | "bullet";

type Health = {
  ok: boolean;
  engine: "ready" | "down";
  pools: PoolId[];
};

type PoolSummary = {
  id: PoolId;
  label: string;
  available: boolean;
  unavailable_reason: string | null;
  games_required: number;
  elo_min: number;
  elo_max: number;
  depth: number;
  revision: string | null;
  ship_ready: boolean;
};

type Reason =
  | "ok"
  | "wrong_pool"
  | "unrated"
  | "unfinished"
  | "variant"
  | "name_missing"
  | "too_short";

type CheckedGame = {
  index: number;
  accepted: boolean;
  reason: Reason;
  color: "white" | "black" | null;
  white: string;
  black: string;
  date: string | null;
  opening: string | null;
  result: string | null;
  moves: number | null;
};

type Driver = {
  feature: string;
  direction: "higher" | "lower";
  text: string;
};

type Moment = {
  game_index: number;
  ply: number;
  played: string;
  best: string;
  cp_loss: number;
  blunder: boolean;
};

type Stats = {
  acc: number;
  blunder_rate: number;
  mistake_rate: number;
  best_move_rate: number;
  cpl_opening: number | null;
  cpl_late: number | null;
  n_games: number;
  castle_kingside: number;
  castle_queenside: number;
  uncastled: number;
  castle_ply_mean: number | null;
  opp_acc: number;
  clk_low_rate: number | null;
  blunder_in_time_trouble_rate: number | null;
};

type Estimate = {
  pool: PoolId;
  revision: string;
  ship_ready: boolean;
  lab: boolean;
  n_games: number;
  depth: number;
  q10: number;
  q50: number;
  q90: number;
  drivers: Driver[];
  stats: Stats;
  moments: Moment[];
  limits: {
    typical_error: number;
    within_150: number;
    coverage: number;
    tail_note: string | null;
  };
};

type Job =
  | { id: string; status: "queued" | "running" }
  | { id: string; status: "done"; estimate: Estimate }
  | { id: string; status: "failed"; error: string; message: string };

type BinRow = {
  bin: string;
  n: number;
  mae: number;
  bias: number;
  coverage: number;
  median_width: number;
};

type PoolCard = PoolSummary & {
  source_label: string;
  test_players: number;
  bins: BinRow[];
  summary: string[];
};
```

`lab` is true whenever `ship_ready` is false. Trust the flag.

`pools` on health lists only available pools. The picker uses `GET /v1/pools`, which also lists the disabled ones.

`depth` is the Stockfish search depth. `revision` names the model. Current values: `classical8-v1`, `rapid8-v3`. Disabled pools send `revision: null`.

`within_150` and `coverage` are fractions from 0 to 1. Format `within_150` as a percent with no decimal (`0.658` → `66%`). `typical_error`, `mae`, `bias`, and `median_width` are rating points. Format as integers. Bias gets a sign (`+121`, `−90`). `bin` already contains an en dash (`800–1000`).

The headline is `q50` rounded to the nearest 10. A payload of 1706 reads **1710**. `q10` and `q90` stay in the JSON for the report track; the page rounds them the same way. They are not a confidence score.

### Published eight-game models

| Pool | Revision | Bundle | Features | Interval | Ship bar |
| --- | --- | --- | --- | --- | --- |
| classical | `classical8-v1` | `model/classical8-v1` | 25 behavior/engine means | one global `interval_scale` | macro MAE ≤120, tail bias ±80, width ≤350, bin coverage ≥70% |
| rapid | `rapid8-v3` | `model/rapid8-v3` | 57 (`feature_version: 3`) | one scale per predicted rating bin (`interval_scales`) | macro MAE ≤120 and tail bias ±80 only; width and coverage are reported |

Rapid v3 passed gates G2 and G3 in `notebooks/train_player8_v3_trained.ipynb` (test macro MAE 130.2 with clocks, 133.8 without, vs v2's 152.2). `ship_ready` is still false, so every rapid estimate carries `lab: true`. Serving v3 requires the full Colab bundle under `model/rapid8-v3` (not the smoke folder), `PUBLISHED["rapid"]` pointed at that revision, and the live Stockfish pass to emit the v3 columns.

Rapid scoring path:

1. For each selected game, call the same `extract_game` walk training used (`src/game_features.py`). Missing clocks leave clock features blank (`NaN`) and set `has_clock` to 0. Do not fail the job.
2. Average the eight game rows with `average_games_v3` (means plus the spread columns). Score that one row. Do not average eight separate guesses.
3. Apply `interval_scales` from the card (per predicted bin). Fall back to a single `interval_scale` only for classical / v2 cards.
4. Fill `stats` and `moments` from that same pass. Clock dashboard fields stay null unless every subject move has a clock.

`stats.acc` and `stats.opp_acc` are 0–100. The rate fields are fractions. `cpl_opening` is the mean subject loss on plies 1–20, and `cpl_late` is the mean on plies after 60. Either is null when that phase has no subject moves. A mistake is a loss of at least 100. A blunder is a loss of at least 200. `clk_low_rate` and `blunder_in_time_trouble_rate` are null unless every subject move has a clock. `moments` is the subject's three largest losses across the eight games, even when none reaches 200, ordered by `cp_loss` descending. `played` and `best` are standard algebraic notation. `game_index` is the checklist index.

Ratings are absent from every payload on purpose. Do not parse them out of the PGN to show them.

## GET /v1/health

`200`

```json
{ "ok": true, "engine": "ready", "pools": ["classical", "rapid"] }
```

`ok` means the process is up. `engine` is `ready` or `down`.

## GET /v1/pools

`200`

```json
{
  "pools": [
    {
      "id": "classical",
      "label": "Lichess classical",
      "available": true,
      "unavailable_reason": null,
      "games_required": 8,
      "elo_min": 800,
      "elo_max": 2400,
      "depth": 10,
      "revision": "classical8-v1",
      "ship_ready": false
    },
    {
      "id": "rapid",
      "label": "Lichess rapid",
      "available": true,
      "unavailable_reason": null,
      "games_required": 8,
      "elo_min": 800,
      "elo_max": 2400,
      "depth": 10,
      "revision": "rapid8-v3",
      "ship_ready": false
    },
    {
      "id": "blitz",
      "label": "Lichess blitz",
      "available": false,
      "unavailable_reason": "No blitz model is trained yet.",
      "games_required": 8,
      "elo_min": 800,
      "elo_max": 2400,
      "depth": 10,
      "revision": null,
      "ship_ready": false
    },
    {
      "id": "bullet",
      "label": "Lichess bullet",
      "available": false,
      "unavailable_reason": "No bullet model is trained yet.",
      "games_required": 8,
      "elo_min": 800,
      "elo_max": 2400,
      "depth": 10,
      "revision": null,
      "ship_ready": false
    }
  ]
}
```

Render one radio per row, in this order. `available: false` is visible and disabled, with `unavailable_reason` beside it. Both live pools use the track 800 to 2400.

## GET /v1/pools/{pool}

`200` with a `PoolCard` when `{pool}` is one of the four ids. `404 unknown_pool` otherwise.

`/method` renders only this object. Title is `label`. Print each `summary` string as its own paragraph. Table column titles: rating range (`bin`), players (`n`), typical miss (`mae`), lean high or low (`bias`), how often the hidden range contained the real rating (`coverage`), how wide that range usually was (`median_width`). Footer line: `{source_label}. Stockfish depth {depth}. {games_required} games. Revision {revision}.`

An available card looks like this. `bins` has eight rows. The numbers below are the live classical card, rounded the way the server sends them.

```json
{
  "id": "classical",
  "label": "Lichess classical",
  "available": true,
  "unavailable_reason": null,
  "games_required": 8,
  "elo_min": 800,
  "elo_max": 2400,
  "depth": 10,
  "revision": "classical8-v1",
  "ship_ready": false,
  "source_label": "Lichess classical, February 2017",
  "test_players": 1119,
  "summary": [
    "This is a lab estimate. The ship bar was a typical miss of 120 Elo or better, and the test missed by 124.",
    "68% of 1,119 players the model had not seen landed within 150 rating points. The hidden range contained the real rating for 80% of them.",
    "Players from 800 to 1000 were guessed about 121 Elo too high. Players from 2000 to 2200 were guessed about 90 Elo too low."
  ],
  "bins": [
    { "bin": "800–1000", "n": 140, "mae": 145.8, "bias": 120.5, "coverage": 0.68, "median_width": 321 }
  ]
}
```

An available rapid card looks like this. Numbers come from the scored eight-game test in `notebooks/train_player8_v3_trained.ipynb`. Read live responses from the server; do not hardcode the metrics in the client.

```json
{
  "id": "rapid",
  "label": "Lichess rapid",
  "available": true,
  "unavailable_reason": null,
  "games_required": 8,
  "elo_min": 800,
  "elo_max": 2400,
  "depth": 10,
  "revision": "rapid8-v3",
  "ship_ready": false,
  "source_label": "Lichess rapid, September 2026",
  "test_players": 2398,
  "summary": [
    "This is a lab estimate. The ship bar was a typical miss of 120 Elo or better, and the test missed by 130.",
    "66% of 2,398 players the model had not seen landed within 150 rating points. The hidden range contained the real rating for 81% of them.",
    "Players from 800 to 1000 were guessed about 94 Elo too high."
  ],
  "bins": [
    { "bin": "800–1000", "n": 300, "mae": 112.2, "bias": 93.7, "coverage": 0.75, "median_width": 347 },
    { "bin": "1000–1200", "n": 300, "mae": 134.0, "bias": 69.0, "coverage": 0.85, "median_width": 465 },
    { "bin": "1200–1400", "n": 300, "mae": 139.8, "bias": 49.7, "coverage": 0.85, "median_width": 523 },
    { "bin": "1400–1600", "n": 300, "mae": 155.1, "bias": 1.5, "coverage": 0.82, "median_width": 531 },
    { "bin": "1600–1800", "n": 300, "mae": 136.8, "bias": -11.0, "coverage": 0.84, "median_width": 521 },
    { "bin": "1800–2000", "n": 300, "mae": 135.4, "bias": -50.9, "coverage": 0.82, "median_width": 476 },
    { "bin": "2000–2200", "n": 299, "mae": 125.3, "bias": -63.3, "coverage": 0.81, "median_width": 398 },
    { "bin": "2200–2400", "n": 299, "mae": 103.0, "bias": -79.1, "coverage": 0.7, "median_width": 233 }
  ]
}
```

A disabled pool returns `200` with `available: false`, `bins: []`, `test_players: 0`, and `summary` set to a one-item list holding `unavailable_reason`.

## POST /v1/games:check

Does not run Stockfish. Safe to call while `engine` is `down`.

```json
{
  "pool": "rapid",
  "pgn": "[Event \"Rated Rapid game\"]\n[White \"Ada\"]\n[Black \"Bea\"]\n[Result \"1-0\"]\n\n1. e4 e5 1-0\n",
  "subject": "Ada"
}
```

`200`

```json
{
  "pool": "rapid",
  "games_required": 8,
  "games": [
    {
      "index": 0,
      "accepted": true,
      "reason": "ok",
      "color": "white",
      "white": "Ada",
      "black": "Bea",
      "date": "2026.09.14",
      "opening": "Italian Game",
      "result": "1-0",
      "moves": 42
    },
    {
      "index": 1,
      "accepted": false,
      "reason": "wrong_pool",
      "color": null,
      "white": "Ada",
      "black": "Bea",
      "date": "2026.09.14",
      "opening": null,
      "result": "0-1",
      "moves": 31
    }
  ]
}
```

`index` is the game's position in the paste, starting at 0, including refused games. `accepted` is true only for `reason: "ok"`. `color` is `white` or `black` only when accepted, otherwise `null`. The subject match ignores case and surrounding space. If the name is on both sides, the color is white.

`moves` is the length of the game in full moves. It can be present on a refused row. `date` is the PGN `Date` header. `opening` is the `Opening` header.

Checklist copy. `{id}` is the selected pool's id (`rapid`). `{label}` is its label (`Lichess rapid`).

| `reason` | Copy |
| --- | --- |
| `ok` | Rated {id}, finished, standard chess. |
| `wrong_pool` | This game is another time control. The loaded model is {label}. |
| `unrated` | Unrated and correspondence games are refused. |
| `unfinished` | The game did not end normally, so the move record is cut short. |
| `variant` | Only standard chess. |
| `name_missing` | Neither header matches the subject. |
| `too_short` | The subject has fewer than 10 moves. |

For `ok`, prefix the color: `White · 2026.09.14 · Italian Game · 1-0 · 42 moves`. Names, date, opening, result, and move count may sit on any row. Elo must not.

The first matching reason wins:

1. `variant` when `Variant` is present and is not Standard chess.
2. `wrong_pool` when `Site` contains `chess.com`.
3. `unrated` for correspondence, a casual or unrated event, an event that does not say rated, or a missing `TimeControl` (`-` or blank).
4. `wrong_pool` when the game's speed is not the selected pool. The speed is the Lichess `Event` name (`classical`, `rapid`, `blitz`, `bullet`, `ultrabullet`). If the event names none, the clock is used: base seconds plus 40 times the increment. Under 3 minutes is bullet, under 8 is blitz, under 25 is rapid, and the rest is classical.
5. `name_missing` when neither player header matches the subject.
6. `unfinished` when `Termination` is not `Normal`, or the result is not `1-0`, `0-1`, or `1/2-1/2`.
7. `too_short` when the subject played fewer than 10 moves.
8. `ok`.

A blitz game in a rapid paste is `wrong_pool` and its checkbox stays off. Switching the picker to blitz does not make it acceptable, because blitz is `pool_unavailable` and the check returns 400.

## POST /v1/estimates

```json
{
  "pool": "rapid",
  "pgn": "[Event \"Rated Rapid game\"]\n...",
  "subject": "Ada",
  "game_indexes": [0, 2, 3, 4, 5, 6, 7, 9]
}
```

`game_indexes` must have length `games_required`. Each index must be an integer that was `accepted` on this same paste and subject, with no repeats. A wrong length is `wrong_count`. A repeat, a boolean, a non-integer, an unknown index, or a refused index is `bad_indexes`.

`202` when the job is accepted. The body is a job object, usually `queued`:

```json
{ "id": "job_15f52d0a9e52", "status": "queued" }
```

The id is `job_` plus 12 hex characters. Poll that id. Do not assume the 202 body is the estimate.

`503` when the engine is down. `429` when eight jobs are already waiting. Both leave the checklist as it was.

The server reads the eight games with Stockfish at the pool's `depth`, averages the features, and scores that one row. It does not average eight separate guesses. Classical uses the v1 feature table. Rapid uses the v3 table (`extract_game` → `average_games_v3` → per-bin `interval_scales`). The same pass fills `stats` and `moments`. A typical read takes a few seconds. Eight long games take longer. The wait copy is `Reading the games with Stockfish, depth {depth}.`

## GET /v1/estimates/{id}

`200` while the job exists. `404 unknown_job` when the id is unknown or the finished job has aged out. Finished jobs are kept for 10 minutes.

Queued or running:

```json
{ "id": "job_15f52d0a9e52", "status": "running" }
```

Done:

```json
{
  "id": "job_15f52d0a9e52",
  "status": "done",
  "estimate": {
    "pool": "rapid",
    "revision": "rapid8-v3",
    "ship_ready": false,
    "lab": true,
    "n_games": 8,
    "depth": 10,
    "q10": 1677.2,
    "q50": 2096.4,
    "q90": 2217.1,
    "drivers": [
      {
        "feature": "cpl_mean",
        "direction": "lower",
        "text": "Giving away less on a typical move raised the guess."
      }
    ],
    "stats": {
      "acc": 81,
      "blunder_rate": 0.02,
      "mistake_rate": 0.08,
      "best_move_rate": 0.44,
      "cpl_opening": 28,
      "cpl_late": null,
      "n_games": 8,
      "castle_kingside": 8,
      "castle_queenside": 0,
      "uncastled": 0,
      "castle_ply_mean": 11,
      "opp_acc": 74,
      "clk_low_rate": null,
      "blunder_in_time_trouble_rate": null
    },
    "moments": [
      {
        "game_index": 2,
        "ply": 47,
        "played": "Bxf7",
        "best": "Qh5",
        "cp_loss": 310,
        "blunder": true
      }
    ],
    "limits": {
      "typical_error": 130,
      "within_150": 0.658,
      "coverage": 0.805,
      "tail_note": "Players this far above the middle were often guessed low."
    }
  }
}
```

That payload's headline is **2100**. `limits.typical_error` and `within_150` are the rapid v3 held-out figures (macro MAE ~130, within-150 ~66%). `drivers` has at most three items. Render `text` in order. Use `feature` only as a React key. Do not write sentences from feature names. `direction` is `higher` or `lower` and is not shown. Driver copy still covers the five monotone v1 features; v3-only columns are not turned into driver sentences.

When `lab` is true, the banner above the headline is:

> This is a lab guess. For players the model had not seen, about {within_150 percent} of guesses landed within 150 rating points. Guesses at the very bottom and the very top tend to drift toward the middle.

When `lab` is true, do not also print the miss line. When `lab` is false, omit the banner and print: `Guesses like this are usually off by about {typical_error} rating points.`

Then `{n_games} rated {pool label} games.` The result does not print `tail_note`, `q10`, or `q90`.

Driver sentences, one pair per feature. The server sends the matching half:

- Fewer serious mistakes than usual raised the guess. / More serious mistakes than usual lowered the guess.
- Fewer big giveaways raised the guess. / More big giveaways lowered the guess.
- Giving away less on a typical move raised the guess. / Giving away more on a typical move lowered the guess.
- A steadier typical move raised the guess. / A sloppier typical move lowered the guess.
- Matching the engine less often lowered the guess. / Matching the engine more often raised the guess.

Dashboard lines, from `docs/frontend.md`, one per stat that is present:

- `Move quality: {acc} out of 100. 100 would mean every move matched a strong engine.`
- `Big giveaways: {blunder percent} of moves.`
- `Serious slips: {mistake percent} of moves.`
- `Matched the engine's choice: {best percent} of moves.`
- Opening versus late, only when both losses exist: `The opening was cleaner than the ending.` or `The ending was cleaner than the opening.` or `The opening and the ending were about equally clean.`
- `Castled kingside in {k} of {n} games.` and the same shape for queenside. If any game never castled: `Left the king in the center in {u} of {n} games.`
- `Opponents' move quality: {opp} out of 100. That is how cleanly they played, not their rating.`
- `About {low percent} of moves were made short of time.`
- `Of the big giveaways, about {trouble percent} came when short of time.`

Moment list title: `Three moves that cost the most.` Each row: `Game {n}, move {move}. Played {played}. A strong engine preferred {best}.` A blunder ends with `This one gave away a lot.` Any other row ends with `This one was a costly slip.`

`moments` in the example above is one row for brevity; a real response has the three largest subject losses. This example has no clocks, so both clock stats are null. Skip the queenside and uncastled lines because those counts are zero. `cpl_late` is null, so the opening-versus-late line is omitted.

Footer, and it must match the card: `{pool label} · Stockfish depth {depth} · {revision}`. Link the revision text to `/method`.

Leave out a confidence percent and a rating read out of the PGN. The headline is the guess.

Failed, still HTTP 200:

```json
{
  "id": "job_15f52d0a9e52",
  "status": "failed",
  "error": "unreadable",
  "message": "One of those games could not be scored. Pick another eight."
}
```

`unreadable` is the usual job failure. `engine_down` is the other. Show `message` under the button and keep the checklist.
