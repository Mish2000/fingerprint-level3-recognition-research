"""Subject-level development/test split (C1).

Defined now so that it is fixed before any score is seen (R5, R7); it is not
applied until C1 is activated in configs/protocol.toml.
"""

from __future__ import annotations

from collections import defaultdict

from .sampling import keyed_rank


def stratified_subject_split(strata: dict[str, str], dev_fraction: float, master_seed: str) -> dict[str, str]:
    """Map subject -> 'dev' or 'test'.

    Within each stratum, subjects are ranked by keyed hash and the first
    round(n * dev_fraction) become 'dev'.
    """
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for subject, stratum in strata.items():
        by_stratum[stratum].append(subject)
    assignment: dict[str, str] = {}
    for stratum in sorted(by_stratum):
        ranked = keyed_rank(by_stratum[stratum], master_seed, f"split:{stratum}", key=lambda subject: subject)
        n_dev = round(len(ranked) * dev_fraction)
        for position, subject in enumerate(ranked):
            assignment[subject] = "dev" if position < n_dev else "test"
    return assignment


def has_r_strata(manifest: list[dict]) -> dict[str, str]:
    """Stratum per subject: whether the subject has R (slap) images at all."""
    subjects = {row["subject"] for row in manifest}
    with_r = {row["subject"] for row in manifest if row["set"] == "R"}
    return {subject: "has_R" if subject in with_r else "no_R" for subject in subjects}
