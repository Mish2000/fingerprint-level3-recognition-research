"""Score statistics used before the full evaluation layer exists (F1 gate, timing pilots)."""

from __future__ import annotations

import bisect
import statistics


def auc(genuine: list[float], impostor: list[float]) -> float:
    """Probability that a random genuine score exceeds a random impostor score; ties count one half."""
    if not genuine or not impostor:
        raise ValueError("AUC needs at least one genuine and one impostor score")
    ordered = sorted(impostor)
    wins = 0.0
    for score in genuine:
        below = bisect.bisect_left(ordered, score)
        ties = bisect.bisect_right(ordered, score) - below
        wins += below + 0.5 * ties
    return wins / (len(genuine) * len(ordered))


def no_false_accept_point(genuine: list[float | None], impostor: list[float | None]) -> dict:
    """The strictest operating point that accepts no impostor: a pair is accepted when its score is
    above every impostor score. A failure (None) is never accepted (E1)."""
    scored = [s for s in impostor if s is not None]
    top = max(scored) if scored else None
    true_accepts = sum(1 for s in genuine if s is not None and (top is None or s > top))
    return {
        "accept_if_score_above": top,
        "genuine": len(genuine),
        "true_accepts": true_accepts,
        "false_rejects": len(genuine) - true_accepts,
        "impostor": len(impostor),
        "false_accepts": 0,
        "tar": true_accepts / len(genuine),
        "frr": 1 - true_accepts / len(genuine),
        "far": 0.0,
    }


def summary(values: list[float]) -> dict:
    """Count, mean, median, 95th percentile (nearest rank), min and max."""
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, max(0, round(0.95 * len(ordered)) - 1))]
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": p95,
        "min": ordered[0],
        "max": ordered[-1],
    }
