# Frontend reference

The page turns eight rated games into a 10–90% Elo band. The client calls `/v1` and renders the JSON. It does not ship a model, run Stockfish, or parse ratings out of the PGN to display them.

Two cards are live, both eight games at Stockfish depth 10:

- Lichess classical, February 2017, revision `classical8-v1`.
- Lichess rapid, September 2026, revision `rapid8-v2`.

`ship_ready` is false on both. The estimate payload still carries `lab: true`. The report screen does not print a lab banner. Blitz and bullet are listed by `GET /v1/pools` as unavailable.

`depth` is always the Stockfish search depth. The revision string names the model. It is not a depth.

Client routes:

- `/` is the landing and the upload tool.
- `/report/{id}` is the finished read. `{id}` is the job id from `POST /v1/estimates`.
- `/replay` plays the uploaded PGN on the board.
- `/method` is the model card. Render it from `GET /v1/pools/{pool}`. Do not hardcode the metrics.

## Screen

Build `/` as the landing: queen portrait and headline first, then one upload column under it, in this order.

1. Pool picker. One radio per pool from `GET /v1/pools`, in that order. `available: false` is visible and disabled, with `unavailable_reason` beside it.
2. Subject. Label: `Username you played under`. Sent to the server. On the report, show that name again only when this browser still has it for the job id.
3. PGN. A drop target (drop or browse). Read the file as UTF-8 text and store it in the same field used for checks and estimates. Cap the client at 2,097,152 bytes before posting. Hint under the control: `A PGN is the text record of the games.` There is no paste textarea on this iteration.
4. Checklist. Empty until a check succeeds. Title: `Select games`. While checking, the count line says `Reading headers.` Each accepted game has a checkbox. Refused games are visible, unchecked, and not checkable. Show a live count `{ticked} of {games_required} selected`. Offer `Select the first {games_required}` when the sheet is visible.
5. Watch (optional). When at least one watched accepted game is available, a text control `Watch the game` / `Watch the games` navigates to `/replay`. It does not start an estimate.
6. Button. Label: `Estimate from these {games_required}` (on the live pools that reads `Estimate from these 8`). Disabled until the number of ticked games equals `games_required` on the selected pool.
7. Wait. Replaces the button while a job is in flight. Copy: `Reading the games with Stockfish, depth {depth}.` A cancel control stops polling and leaves the checklist as it was.
8. Failure. One sentence under the button. The checklist stays.
9. Done. Do not render the band on `/`. On `done`, navigate to `/report/{id}`.

Headline on `/`: `Get your Elo guessed.` Support line on this iteration: `Upload your games. Discover your estimated playing strength. The 10–90% band is the low and high of that guess, not how sure the model is.`

The name row can load that player's latest rated games from Lichess and write them into the PGN field. The subject becomes the username. [plan.md](plan.md) specifies the request, the 30-game cap, and the optional Lichess login. The check and the estimate still post the same `pgn` and `subject`. This client calls Lichess. The Queen's Range API does not. Connect with Lichess is specified in plan.md and is not required for this document’s Done when until that control ships.

`/method` is a separate page for the selected pool, linked from the header (`How it works`). `/replay` is linked from the landing watch control and from the report.

## What the result shows

The band is the claim. The median sits inside it. The finished read is `/report/{id}`, not the landing.

Display `q10`, `q50`, and `q90` rounded to the nearest 10. Keep the unrounded values off the page. A payload of 1483 / 1706 / 1958 reads **1480–1960**, center **1710**.

The largest type is the rounded center. The range `{q10}–{q90}` prints smaller, on the same screen as the track.

`/report/{id}` loads `GET /v1/estimates/{id}`.

- `queued` or `running`: show `Reading the games with Stockfish, depth {depth}.` when this page already knows `depth` from the posted read; otherwise `The read is still running.`
- `failed`: show `message` (prefer the server string; see error rules below).
- `404` / unknown job: `That read expired.` with a link back to `/` labeled `Start it again.`
- `done`: one centered screen.

Centered done screen, in order:

1. The subject name when this browser still has it for that job id.
2. `We are centering your guess at`
3. The center figure (rounded `q50`)
4. `based on these {n_games} games.` (singular: `based on this game.` when `n_games` is 1)
5. The track, with a brass tick at `q50` and no ivory span
6. The range under the track
7. `Watch replays` linking to `/replay`

Draw the track from `elo_min` to `elo_max` on the pool (800 to 2400 for both live pools). A brass tick marks `q50`. If a quantile falls outside the track, keep the printed numbers and clip the mark to the track edge.

The estimate payload may still include `lab`, `drivers`, `limits.tail_note`, and `revision`. This iteration’s report UI does not print the lab line, driver sentences, tail note, posted game list, or revision footer. `feature` is only a React key when drivers are handled in code. `direction` is not shown.

A reload of `/report/{id}` that only has the job id still shows the band when the job is `done`. It does not invent a game list.

## What the page leaves out

- A cheat score, a confidence percent, or a “you are 1710” badge.
- The subject’s rating, the opponent’s rating, or who won, as inputs or as outputs on the landing, checklist, board, or report.
- Blitz, bullet, Chess.com, or an unrated game scored on the classical card. Those fail in the checklist. Blitz and bullet stay visible and disabled in the picker.
- Any averaging of one-game guesses. The button sends `games_required` indexes. The server averages features, then scores once.
- Lab banner, driver sentences, tail note, and revision footer on the report screen (payload fields may still arrive).
- Hardcoded model metrics on `/method` (test-player counts, bins, depth, revision). Those render from `GET /v1/pools/{pool}`.

## Form states

| State | What is on screen |
| --- | --- |
| `idle` | Picker, empty PGN, button disabled. |
| `checking` | Name and PGN stay editable. Checklist slot says `Reading headers.` A newer edit supersedes the check in flight. |
| `ready` | Checklist filled. Button enabled only at `games_required` ticks. |
| `queued` / `running` | Button hidden. Wait copy. Cancel returns to `ready`. |
| `done` | Navigate to `/report/{id}`. Landing does not keep a result block. |
| `failed` | `ready`, plus one error sentence under the button. |

Debounce `games:check` by 400 ms after the PGN or the subject name changes. Do not check an empty PGN. A new check increments a request id and ignores stale responses. Changing the pool clears the checklist, the job, and any in-flight read, then checks again at once if the PGN and the name are both non-empty. Acceptance depends on the pool.

Poll `GET /v1/estimates/{id}` every 1 second. Stop on `done` or `failed`. Stop after 90 seconds, drop the id, and show `The read took too long. Try again.` Cancel also drops the id. There is no cancel route. On `done`, the landing navigates to `/report/{id}`; that page loads the same id again for the estimate body.

## HTTP

Base URL comes from the environment (`VITE_API_BASE_URL`, default `http://127.0.0.1:8000`). JSON only. No API key. Send `Content-Type: application/json`.

| Call | When | Success |
| --- | --- | --- |
| `GET /v1/health` | App load | `200` |
| `GET /v1/pools` | App load | `200` |
| `GET /v1/pools/{pool}` | `/method`, and when the picker changes | `200` |
| `POST /v1/games:check` | PGN or subject changed | `200` |
| `POST /v1/estimates` | Button | `202` |
| `GET /v1/estimates/{id}` | Every 1 s until done on `/`, and again on `/report/{id}` | `200` |

Error body, for every 4xx and 5xx:

```json
{ "error": "wrong_count", "message": "This model scores 8 games at a time." }
```

Show `message` when it is a non-empty string. Otherwise use the sentence in the error table below.

| HTTP | `error` | Sentence |
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

A failed job is HTTP 200 with `"status": "failed"` and the same `error` / `message` pair. `unknown_job` is the 404.

Health `engine: "down"` disables the estimate button and shows the `engine_down` sentence. The checklist still runs.

## Types

Client TypeScript (from `src/api/types.ts`). Wire JSON still matches [backend.md](backend.md); the client splits poll parsing from the full done payload.

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

type GameCheck = {
  pool: PoolId;
  games_required: number;
  games: CheckedGame[];
};

type EstimateRequest = {
  pool: PoolId;
  pgn: string;
  subject: string;
  game_indexes: number[];
};

type Driver = {
  feature: string;
  direction: "higher" | "lower";
  text: string;
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
  limits: {
    typical_error: number;
    within_150: number;
    coverage: number;
    tail_note: string | null;
  };
};

type StartedJob = {
  id: string;
  status: "queued" | "running";
};

/** Poll outcome used while waiting on `/`. Done here has no estimate body. */
type PolledJob =
  | { id: string; status: "queued" | "running" }
  | { id: string; status: "done" }
  | { id: string; status: "failed"; message: string };

/** Full GET /v1/estimates/{id} parse used by the report (and by the poller before discarding estimate). */
type EstimateJob =
  | { id: string; status: "queued" | "running" }
  | { id: string; status: "done"; estimate: Estimate }
  | { id: string; status: "failed"; message: string };

type ApiFailure = {
  error: string;
  message: string;
};

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

On the wire, a failed job is still `{ status: "failed", error, message }` and a done job still includes `estimate`. The client maps failures to a `message` string for display (`sentenceFor`), and the landing poll treats `done` as an id to open `/report/{id}`.

`lab` is true whenever `ship_ready` is false. Trust the flag on the payload even when the report UI does not print a lab line.

`within_150` and `coverage` are fractions from 0 to 1. Format as percents with no decimal (`0.68` → `68%`) when those numbers are shown (method page / future UI). `typical_error`, `mae`, `bias`, and `median_width` are Elo. Format as integers. Bias gets a sign (`+121`, `−90`).

## Checklist copy

Render refused `reason` values with these sentences. Names, date, opening, result, and move count can sit on the row. Elo must not, and it is absent from the payload on purpose.

`{label}` is the selected pool's `label`.

| `reason` | Copy |
| --- | --- |
| `ok` | (no refusal sentence; accepted rows are checkable) |
| `wrong_pool` | This game is another time control. The loaded model is {label}. |
| `unrated` | Unrated and correspondence games are refused. |
| `unfinished` | The game did not end normally, so the move record is cut short. |
| `variant` | Only standard chess. |
| `name_missing` | Neither header matches the subject. |
| `too_short` | The subject has fewer than 10 moves. |

Accepted rows show game facts (opening, date, result, moves, color), not the old `Rated {id}, finished, standard chess.` sentence. A compact line used when a game list is built elsewhere is still `White · 2017.02.14 · Italian Game · 1-0 · 42 moves` (`gameLine`).

## Method page

Render only fields on `PoolCard`.

- Title: `{label}`.
- Print each string in `summary` as its own paragraph. The server writes those sentences.
- A table of `bins`: bin, players (`n`), typical miss (`mae`), bias, coverage, band width (`median_width`).
- Footer / caption: `{source_label}. Stockfish depth {depth}. {games_required} games. Revision {revision}.`
- Disabled pools still return 200 with empty `bins` and a thin card; render that payload, do not invent metrics.
- No board on `/method`.

## Fixtures

Use these while the API is absent. Swap them for the network without changing components. The numbers are the scored eight-game tests: classical from `notebooks/train_player8_trained.ipynb`, rapid from `notebooks/train_player8_trainedv2.ipynb`. Client stand-ins live in `src/api/fixtures.ts` and must stay in sync with this section.

`GET /v1/health`

```json
{ "ok": true, "engine": "ready", "pools": ["classical", "rapid"] }
```

`GET /v1/pools`

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
      "revision": "rapid8-v2",
      "ship_ready": false
    },
    {
      "id": "blitz",
      "label": "Lichess blitz",
      "available": false,
      "unavailable_reason": "Coming Soon.",
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
      "unavailable_reason": "Coming Soon.",
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

`GET /v1/pools/classical`

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
    "68% of 1,119 held-out players landed within 150 Elo. The band caught 80% of them.",
    "Players from 800 to 1000 were guessed about 121 Elo too high. Players from 2000 to 2200 were guessed about 90 Elo too low."
  ],
  "bins": [
    { "bin": "800–1000", "n": 140, "mae": 145.8, "bias": 120.5, "coverage": 0.68, "median_width": 321 },
    { "bin": "1000–1200", "n": 140, "mae": 127.0, "bias": 64.3, "coverage": 0.89, "median_width": 395 },
    { "bin": "1200–1400", "n": 140, "mae": 129.9, "bias": 38.3, "coverage": 0.81, "median_width": 450 },
    { "bin": "1400–1600", "n": 140, "mae": 106.7, "bias": 17.6, "coverage": 0.86, "median_width": 470 },
    { "bin": "1600–1800", "n": 140, "mae": 115.6, "bias": 4.7, "coverage": 0.84, "median_width": 456 },
    { "bin": "1800–2000", "n": 140, "mae": 125.0, "bias": -24.4, "coverage": 0.84, "median_width": 411 },
    { "bin": "2000–2200", "n": 140, "mae": 133.6, "bias": -89.6, "coverage": 0.79, "median_width": 374 },
    { "bin": "2200–2400", "n": 139, "mae": 104.4, "bias": -79.2, "coverage": 0.71, "median_width": 250 }
  ]
}
```

`GET /v1/pools/rapid`

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
  "revision": "rapid8-v2",
  "ship_ready": false,
  "source_label": "Lichess rapid, September 2026",
  "test_players": 2398,
  "summary": [
    "This is a lab estimate. The ship bar was a typical miss of 120 Elo or better, and the test missed by 152.",
    "58% of 2,398 held-out players landed within 150 Elo. The band caught 80% of them.",
    "Players from 800 to 1000 were guessed about 110 Elo too high, and players from 1000 to 1200 about 102 too high. Players from 2000 to 2200 were guessed about 80 Elo too low."
  ],
  "bins": [
    { "bin": "800–1000", "n": 300, "mae": 139.7, "bias": 110.4, "coverage": 0.67, "median_width": 331 },
    { "bin": "1000–1200", "n": 300, "mae": 168.3, "bias": 102.0, "coverage": 0.84, "median_width": 484 },
    { "bin": "1200–1400", "n": 300, "mae": 162.7, "bias": 76.0, "coverage": 0.86, "median_width": 580 },
    { "bin": "1400–1600", "n": 300, "mae": 172.9, "bias": 12.6, "coverage": 0.81, "median_width": 584 },
    { "bin": "1600–1800", "n": 300, "mae": 155.3, "bias": -14.3, "coverage": 0.84, "median_width": 588 },
    { "bin": "1800–2000", "n": 300, "mae": 152.3, "bias": -71.8, "coverage": 0.85, "median_width": 560 },
    { "bin": "2000–2200", "n": 299, "mae": 143.5, "bias": -80.1, "coverage": 0.84, "median_width": 476 },
    { "bin": "2200–2400", "n": 299, "mae": 122.8, "bias": -79.8, "coverage": 0.72, "median_width": 302 }
  ]
}
```

`POST /v1/games:check`

```json
{
  "pool": "classical",
  "pgn": "[Event \"Rated Classical game\"]\n...",
  "subject": "SomeName"
}
```

```json
{
  "pool": "classical",
  "games_required": 8,
  "games": [
    {
      "index": 0,
      "accepted": true,
      "reason": "ok",
      "color": "white",
      "white": "SomeName",
      "black": "Opponent",
      "date": "2017.02.14",
      "opening": "Italian Game",
      "result": "1-0",
      "moves": 42
    },
    {
      "index": 1,
      "accepted": false,
      "reason": "wrong_pool",
      "color": null,
      "white": "SomeName",
      "black": "Opponent",
      "date": "2017.02.14",
      "opening": null,
      "result": "0-1",
      "moves": 31
    }
  ]
}
```

`POST /v1/estimates` returns `202`:

```json
{ "id": "job_01", "status": "queued" }
```

Body:

```json
{
  "pool": "classical",
  "pgn": "[Event \"Rated Classical game\"]\n...",
  "subject": "SomeName",
  "game_indexes": [0, 2, 3, 4, 5, 6, 7, 9]
}
```

`game_indexes` has length `games_required`, each index was `accepted`, and the `pgn` and `subject` are the same strings that were checked. The server checks the paste again. A wrong length is `wrong_count`. A repeated, out-of-range, or refused index is `bad_indexes`.

`GET /v1/estimates/job_01` while running:

```json
{ "id": "job_01", "status": "running" }
```

Done:

```json
{
  "id": "job_01",
  "status": "done",
  "estimate": {
    "pool": "classical",
    "revision": "classical8-v1",
    "ship_ready": false,
    "lab": true,
    "n_games": 8,
    "depth": 10,
    "q10": 1483,
    "q50": 1706,
    "q90": 1958,
    "drivers": [
      {
        "feature": "mistake_rate",
        "direction": "lower",
        "text": "Fewer serious mistakes than the typical game in this sample pulled the center up."
      },
      {
        "feature": "blunder_rate",
        "direction": "lower",
        "text": "Fewer large errors pulled the center up."
      },
      {
        "feature": "cpl_mean",
        "direction": "higher",
        "text": "Giving away more on a typical move pulled the center down."
      }
    ],
    "limits": {
      "typical_error": 124,
      "within_150": 0.68,
      "coverage": 0.80,
      "tail_note": null
    }
  }
}
```

That done payload displays on `/report/{id}` as **1480–1960**, center **1710**. The payload still has `lab: true` and drivers; the report screen does not print them. For a tail fixture, set `q50` to `980` and `tail_note` to `Players this far below the middle were often guessed high.` (`tail_note` remains off the report UI in this iteration.)

## Done when

- Pools, including disabled blitz and bullet rows, render from `GET /v1/pools`.
- The button stays off until `games_required` accepted games are ticked, and it posts those indexes. Label uses `Estimate from these {games_required}`.
- A blitz game in a classical export shows `wrong_pool` and cannot be ticked.
- On `done`, `/` navigates to `/report/{id}`. The report prints the rounded band (center + range + q50 track) and `Watch replays`. It does not print lab banner, drivers, or revision footer.
- Reloading `/report/{id}` after the checklist is gone still shows the band when the job is `done`.
- `/method` matches both fixture cards, including the bias signs, and mounts no board.
- Switching from classical to rapid re-checks the same PGN, and a rapid game moves from `wrong_pool` to `ok`.
- Engine down disables the estimate and leaves the checklist usable.
- A 90 second poll and a cancel both return to the checklist with no band on `/`.
- `/replay` plays from the landing watch control and from the report link.
