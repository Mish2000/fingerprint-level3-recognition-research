"""Verification metrics with subject weights (E1-E4).

Every rate takes per-pair weights, so one code path gives the point estimate (every weight 1) and
each subject-bootstrap replicate, where a genuine pair weighs as much as its subject was drawn and
an impostor pair as much as the product for its two subjects. A pair is accepted when its score is
above the threshold; failures carry -inf and are never accepted (E1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FAILED = -np.inf


@dataclass
class Scores:
    """One algorithm's scores on one set of pairs, with subject indices 0..n_subjects-1."""

    genuine: np.ndarray
    genuine_subject: np.ndarray
    impostor: np.ndarray
    impostor_probe: np.ndarray
    impostor_reference: np.ndarray
    n_subjects: int


class Grid:
    """The distinct scores, highest first and -inf last, and where every pair sits on them."""

    def __init__(self, scores: Scores):
        ascending = np.unique(np.concatenate([scores.genuine, scores.impostor, [FAILED]]))
        self.values = ascending[::-1]
        last = len(ascending) - 1
        self.genuine = last - np.searchsorted(ascending, scores.genuine)
        self.impostor = last - np.searchsorted(ascending, scores.impostor)


@dataclass
class Rates:
    """FAR and FRR at every grid threshold, highest threshold first."""

    far: np.ndarray
    frr: np.ndarray
    impostor_above: np.ndarray  # impostor weight strictly above each threshold
    impostor_total: float
    genuine_total: float
    thresholds: np.ndarray


def rates(scores: Scores, grid: Grid, subject_weights: np.ndarray | None = None) -> Rates:
    if subject_weights is None:
        w_gen, w_imp = np.ones(len(scores.genuine)), np.ones(len(scores.impostor))
    else:
        w_gen = subject_weights[scores.genuine_subject]
        w_imp = subject_weights[scores.impostor_probe] * subject_weights[scores.impostor_reference]
    gen = np.bincount(grid.genuine, weights=w_gen, minlength=len(grid.values))
    imp = np.bincount(grid.impostor, weights=w_imp, minlength=len(grid.values))
    gen_above, imp_above = np.cumsum(gen) - gen, np.cumsum(imp) - imp
    return Rates(imp_above / imp.sum(), 1 - gen_above / gen.sum(), imp_above, imp.sum(), gen.sum(), grid.values)


def tar_at_far(r: Rates, far: float) -> dict:
    """At the lowest threshold whose FAR does not exceed `far` (decision record 2026-10-08, E2)."""
    k = int(np.nonzero(r.impostor_above <= far * r.impostor_total + 1e-9 * max(1.0, r.impostor_total))[0].max())
    return {"tar": 1 - r.frr[k], "frr": r.frr[k], "far": r.far[k], "accept_if_score_above": r.thresholds[k]}


def eer(r: Rates) -> float:
    """Where FAR meets FRR, interpolated linearly between the two thresholds around the crossing."""
    gap = r.far - r.frr  # rises as the threshold falls
    crossed = np.nonzero(gap >= 0)[0]
    if len(crossed) == 0:
        return float((r.far[-1] + r.frr[-1]) / 2)
    k = int(crossed[0])
    if k == 0:
        return float((r.far[0] + r.frr[0]) / 2)
    share = gap[k - 1] / (gap[k - 1] - gap[k])
    return float(r.far[k - 1] + share * (r.far[k] - r.far[k - 1]))


def summarise(r: Rates, targets: tuple[float, ...]) -> dict:
    return {"eer": eer(r), **{f"tar_at_{t!r}": tar_at_far(r, t)["tar"] for t in targets}}


def subject_bootstrap(sets: dict[str, Scores], targets: tuple[float, ...], replicates: int, seed: int) -> dict[str, dict]:
    """Per algorithm, the EER and TAR at each target FAR in every replicate. All algorithms share
    each replicate's subject draw, so their differences are paired (E4)."""
    n = next(iter(sets.values())).n_subjects
    grids = {name: Grid(s) for name, s in sets.items()}
    out = {name: {key: [] for key in summarise(rates(s, grids[name]), targets)} for name, s in sets.items()}
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        weights = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        for name, s in sets.items():
            for key, value in summarise(rates(s, grids[name], weights), targets).items():
                out[name][key].append(value)
    return {name: {key: np.array(values) for key, values in d.items()} for name, d in out.items()}


def interval(values: np.ndarray, level: float = 0.95) -> list[float]:
    tail = 100 * (1 - level) / 2
    return [float(v) for v in np.percentile(values, [tail, 100 - tail])]
