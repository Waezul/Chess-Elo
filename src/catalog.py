"""Published model cards. The method page and the estimate limits both come from here."""

from __future__ import annotations

import threading
from pathlib import Path

import pandas as pd

from src.metrics import bin_index, bin_label
from src.schema import ELO_MAX, ELO_MIN, MONOTONE, PLAYER_TAIL_BIAS_MAX
from src.score import design_matrix, load_bundle, predict_rows, top_contributions

# Folder name is the revision the page prints. The card's pool has to match.
PUBLISHED = {
    "classical": {
        "directory": "classical8-v1",
        "label": "Lichess classical",
        "source_label": "Lichess classical, February 2017",
        "revision": "classical8-v1",
    },
    "rapid": {
        "directory": "rapid8-v2",
        "label": "Lichess rapid",
        "source_label": "Lichess rapid, September 2026",
        "revision": "rapid8-v2",
    },
}

UNAVAILABLE = (
    {"id": "blitz", "label": "Lichess blitz", "reason": "No blitz model is trained yet."},
    {"id": "bullet", "label": "Lichess bullet", "reason": "No bullet model is trained yet."},
)

POOL_ORDER = ("classical", "rapid", "blitz", "bullet")

# Sentences for features whose direction is fixed by the monotone constraint.
# "lower" means the feature sat below the level that holds the center still.
DRIVER_COPY = {
    "mistake_rate": {
        "lower": "Fewer serious mistakes than usual raised the guess.",
        "higher": "More serious mistakes than usual lowered the guess.",
    },
    "blunder_rate": {
        "lower": "Fewer big giveaways raised the guess.",
        "higher": "More big giveaways lowered the guess.",
    },
    "cpl_mean": {
        "lower": "Giving away less on a typical move raised the guess.",
        "higher": "Giving away more on a typical move lowered the guess.",
    },
    "cpl_median": {
        "lower": "A steadier typical move raised the guess.",
        "higher": "A sloppier typical move lowered the guess.",
    },
    "best_move_rate": {
        "lower": "Matching the engine less often lowered the guess.",
        "higher": "Matching the engine more often raised the guess.",
    },
}

_EN_DASH = "\u2013"


def direction_for(feature: str, contribution: float) -> str:
    """Map a median-model contribution to whether the feature was high or low.

    A positive contribution raises the center. For a feature constrained to
    push the center down as it grows, that means the value was low.
    """
    sign = MONOTONE[feature]
    lowered = (sign < 0 and contribution > 0) or (sign > 0 and contribution < 0)
    return "lower" if lowered else "higher"


def _display_bin(label: str) -> str:
    return label.replace("-", _EN_DASH)


def _round_1(value: float) -> float:
    return round(float(value), 1)


def format_bins(rows: list[dict]) -> list[dict]:
    formatted = []
    for row in rows:
        formatted.append(
            {
                "bin": _display_bin(str(row["bin"])),
                "n": int(row["n"]),
                "mae": _round_1(row["mae"]),
                "bias": _round_1(row["bias"]),
                "coverage": round(float(row["coverage"]), 2),
                "median_width": int(round(float(row["median_width"]))),
            }
        )
    return formatted


def _bin_ends(label: str) -> tuple[str, str]:
    raw = label.replace(_EN_DASH, "-")
    low, high = raw.split("-")
    return low, high


def _side_sentence(rows: list[dict], word: str) -> str:
    first, *rest = rows
    low, high = _bin_ends(str(first["bin"]))
    amount = int(round(abs(float(first["bias"]))))
    text = f"Players from {low} to {high} were guessed about {amount} Elo too {word}"
    for row in rest:
        low, high = _bin_ends(str(row["bin"]))
        amount = int(round(abs(float(row["bias"]))))
        text += f", and players from {low} to {high} about {amount} too {word}"
    return text + "."


def summary_sentences(card: dict) -> list[str]:
    """The paragraphs on ``/method``. Built from the scored test, not written by the client."""
    bar = int(round(float(card["ship_bar"]["macro_mae_max"])))
    missed = int(round(float(card["test_mae"])))
    if card.get("ship_ready"):
        opening = (
            "This estimate cleared the ship bar. Players the model had not seen were missed by about "
            f"{missed} rating points."
        )
    else:
        opening = (
            "This is a lab estimate. The ship bar was a typical miss of "
            f"{bar} Elo or better, and the test missed by {missed}."
        )
    players = sum(int(row["n"]) for row in card["test_bins"])
    within = int(round(100 * float(card["test_within_150"])))
    covered = int(round(100 * float(card["test_coverage"])))
    middle = (
        f"{within}% of {players:,} players the model had not seen landed within 150 rating points. "
        f"The hidden range contained the real rating for {covered}% of them."
    )
    sentences = [opening, middle]
    biased = [row for row in card["test_bins"] if abs(float(row["bias"])) > PLAYER_TAIL_BIAS_MAX]
    highs = [row for row in biased if float(row["bias"]) > 0]
    lows = [row for row in biased if float(row["bias"]) < 0]
    parts = []
    if highs:
        parts.append(_side_sentence(highs, "high"))
    if lows:
        parts.append(_side_sentence(lows, "low"))
    if parts:
        sentences.append(" ".join(parts))
    return sentences


def tail_note(q50: float, bins: list[dict]) -> str | None:
    """Warn only when the center lands in a test bin whose bias cleared 80 Elo."""
    clipped = min(ELO_MAX, max(ELO_MIN, float(q50)))
    label = bin_label(clipped)
    if label is None:
        return None
    row = next((item for item in bins if str(item["bin"]) == label), None)
    if row is None or abs(float(row["bias"])) <= PLAYER_TAIL_BIAS_MAX:
        return None
    index = bin_index(clipped)
    high = float(row["bias"]) > 0
    if index is not None and index <= 1 and high:
        return "Players this far below the middle were often guessed high."
    if index is not None and index >= 6 and not high:
        return "Players this far above the middle were often guessed low."
    if high:
        return "Players in this part of the ladder were often guessed high."
    return "Players in this part of the ladder were often guessed low."


def drivers_from(contributions: list[dict]) -> list[dict]:
    """Up to three sentences, largest absolute contribution first. Zeros are skipped."""
    ranked = []
    for item in contributions:
        feature = str(item["feature"])
        contribution = float(item["contribution"])
        if feature not in DRIVER_COPY or contribution == 0:
            continue
        direction = direction_for(feature, contribution)
        ranked.append((abs(contribution), feature, direction, DRIVER_COPY[feature][direction]))
    ranked.sort(key=lambda row: row[0], reverse=True)
    return [
        {"feature": feature, "direction": direction, "text": text}
        for _, feature, direction, text in ranked[:3]
    ]


class Catalog:
    """Loaded bundles plus the unavailable rows the picker still shows."""

    def __init__(self, root: Path, bundles: dict[str, dict]):
        self.root = root
        self.bundles = bundles
        self._lock = threading.Lock()

    def available_ids(self) -> list[str]:
        return [pool for pool in POOL_ORDER if pool in self.bundles]

    def is_known(self, pool: str) -> bool:
        return pool in POOL_ORDER

    def is_available(self, pool: str) -> bool:
        return pool in self.bundles

    def summary(self, pool: str) -> dict:
        if pool in self.bundles:
            card = self.bundles[pool]["card"]
            meta = PUBLISHED[pool]
            return {
                "id": pool,
                "label": meta["label"],
                "available": True,
                "unavailable_reason": None,
                "games_required": int(card["games_per_player"]),
                "elo_min": ELO_MIN,
                "elo_max": ELO_MAX,
                "depth": int(card["depth"]),
                "revision": meta["revision"],
                "ship_ready": bool(card["ship_ready"]),
            }
        blank = next(item for item in UNAVAILABLE if item["id"] == pool)
        return {
            "id": pool,
            "label": blank["label"],
            "available": False,
            "unavailable_reason": blank["reason"],
            "games_required": 8,
            "elo_min": ELO_MIN,
            "elo_max": ELO_MAX,
            "depth": 10,
            "revision": None,
            "ship_ready": False,
        }

    def list_summaries(self) -> dict:
        return {"pools": [self.summary(pool) for pool in POOL_ORDER]}

    def card(self, pool: str) -> dict:
        if pool not in self.bundles:
            body = self.summary(pool)
            blank = next(item for item in UNAVAILABLE if item["id"] == pool)
            body.update(
                {
                    "source_label": blank["label"],
                    "test_players": 0,
                    "bins": [],
                    "summary": [blank["reason"]],
                }
            )
            return body
        model = self.bundles[pool]["card"]
        meta = PUBLISHED[pool]
        body = self.summary(pool)
        bins = format_bins(model["test_bins"])
        body.update(
            {
                "source_label": meta["source_label"],
                "test_players": sum(row["n"] for row in bins),
                "bins": bins,
                "summary": summary_sentences(model),
            }
        )
        return body

    def limits(self, pool: str, q50: float) -> dict:
        card = self.bundles[pool]["card"]
        return {
            "typical_error": int(round(float(card["test_mae"]))),
            "within_150": float(card["test_within_150"]),
            "coverage": float(card["test_coverage"]),
            "tail_note": tail_note(q50, card["test_bins"]),
        }

    def score_frame(self, pool: str, frame: pd.DataFrame) -> dict:
        """Quantiles and driver lines for one averaged player row."""
        bundle = self.bundles[pool]
        with self._lock:
            scored = predict_rows(frame, bundle, contributions=False)
            matrix = design_matrix(
                frame,
                bundle["card"]["feature_names"],
                bundle["card"]["numeric_features"],
                bundle["encoder"],
            )
            contributions = top_contributions(
                bundle["models"]["q50"],
                matrix,
                list(DRIVER_COPY),
                k=len(DRIVER_COPY),
            )
        row = scored.iloc[0]
        return {
            "q10": float(row["q10"]),
            "q50": float(row["q50"]),
            "q90": float(row["q90"]),
            "drivers": drivers_from(contributions[0]),
        }


def load_catalog(root: Path) -> Catalog:
    bundles = {}
    for pool, meta in PUBLISHED.items():
        directory = root / meta["directory"]
        card_path = directory / "model_card.json"
        if not card_path.exists():
            raise SystemExit(f"model bundle is incomplete: {directory}")
        bundle = load_bundle(directory)
        if bundle["card"].get("pool") != pool:
            raise SystemExit(
                f"{directory} is a {bundle['card'].get('pool')} model and {meta['directory']} is for {pool}"
            )
        bundles[pool] = bundle
    return Catalog(root, bundles)
