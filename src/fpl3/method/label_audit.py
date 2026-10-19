"""Rules of the Q and J label audit (A5, decision record 2026-10-19), kept free of GPU code.

NIST notes that not every palm capture shows the labelled position (sd302c README). Each usable Q
and J image is scored against every usable V image of its subject and against the V images of its
finger position from every other subject. T is the threshold that at most 0.01 % of the
cross-subject comparisons score above (the E2 rule). The image is
- confirmed: its own V (same subject, labelled position) scores above T and above every other candidate;
- suspected: another candidate scores above T and above its own V (or it has no own V);
- undetermined: anything else; or no_reference when its own V is excluded and nothing else is clear.
Undetermined images stay in; a suspected one leaves only with the researcher's approval (R4).
"""

from __future__ import annotations

import math

CROSS_SHARE = 0.0001
FIELDS = ("image_id", "set", "subject", "frgp", "own_score", "best_other_score", "best_other_image_id",
          "best_other_same_subject", "category")


def candidate_references(row: dict, references: list[dict]) -> list[dict]:
    """Every usable V image of the same subject, and the V images of the same finger position of every other subject."""
    return [v for v in references if v["subject"] == row["subject"] or int(v["frgp"]) == int(row["frgp"])]


def is_own(row: dict, reference: dict) -> bool:
    return reference["subject"] == row["subject"] and int(reference["frgp"]) == int(row["frgp"])


def cross_subject(row: dict, scored: list[tuple[dict, float]]) -> list[float]:
    return [s for v, s in scored if v["subject"] != row["subject"]]


def threshold(scores: list[float], share: float = CROSS_SHARE) -> float:
    """The lowest threshold that at most `share` of `scores` lie above; -inf when every score may."""
    allowed = math.floor(share * len(scores) + 1e-9)
    ordered = sorted(scores, reverse=True)
    return ordered[allowed] if allowed < len(ordered) else -math.inf


def classify(row: dict, scored: list[tuple[dict, float]], t: float) -> dict:
    """The audit record of one Q or J image from its candidates' scores."""
    own = next((s for v, s in scored if is_own(row, v)), None)
    others = [(v, s) for v, s in scored if not is_own(row, v)]
    best_v, best = max(others, key=lambda vs: vs[1]) if others else (None, None)
    if best is not None and best > t and (own is None or best > own):
        category = "suspected"
    elif own is None:
        category = "no_reference"
    elif own > t and (best is None or own > best):
        category = "confirmed"
    else:
        category = "undetermined"
    return {
        "image_id": row["image_id"], "set": row["set"], "subject": row["subject"], "frgp": int(row["frgp"]),
        "own_score": "" if own is None else own, "best_other_score": "" if best is None else best,
        "best_other_image_id": best_v["image_id"] if best_v else "",
        "best_other_same_subject": int(best_v["subject"] == row["subject"]) if best_v else "",
        "category": category,
    }
