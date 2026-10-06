"""Deterministic samples drawn from C1's development half: the F1 gate and the timing pilots.

C1 is not activated for anything else (decision record, 2026-10-06): these
helpers only read which subjects the fixed split would place in development.
"""

from __future__ import annotations

from .pairs import pair_record
from .sampling import keyed_rank
from .split import has_r_strata, stratified_subject_split

PROBE_SETS = {"U": "UxV", "R": "RxV"}


def development_subjects(images: list[dict], master_seed: str, dev_fraction: float) -> tuple[set[str], dict[str, str]]:
    strata = has_r_strata([row for row in images if row["set"] in ("U", "V", "R")])
    split = stratified_subject_split(strata, dev_fraction, master_seed)
    return {subject for subject, role in split.items() if role == "dev"}, strata


def usable_index(images: list[dict]) -> dict[tuple[str, str, int], dict]:
    return {(row["set"], row["subject"], int(row["frgp"])): row for row in images if not int(row["excluded"])}


def f1_sample(
    images: list[dict], master_seed: str, dev_fraction: float, n_subjects: int = 10, n_fingers: int = 2
) -> tuple[list[str], list[int], list[dict], list[dict]]:
    """F1 sample (docs/algorithms/pore-sift.md): subjects, finger positions, images and pairs.

    Finger positions are the first `n_fingers` of 1..10 by keyed hash ('f1-fingers').
    Subjects are development subjects that have R images, in keyed-hash order
    ('f1-subjects'), keeping the first `n_subjects` whose U, V and R images at
    both positions are all usable.
    """
    development, strata = development_subjects(images, master_seed, dev_fraction)
    index = usable_index(images)
    fingers = sorted(int(f) for f in keyed_rank([str(f) for f in range(1, 11)], master_seed, "f1-fingers", key=str)[:n_fingers])
    candidates = keyed_rank(sorted(s for s in development if strata[s] == "has_R"), master_seed, "f1-subjects", key=str)
    subjects = [
        s for s in candidates if all((image_set, s, f) in index for image_set in ("U", "V", "R") for f in fingers)
    ][:n_subjects]
    if len(subjects) < n_subjects:
        raise ValueError(f"only {len(subjects)} eligible development subjects for F1")

    sample_images = [index[(image_set, s, f)] for s in subjects for f in fingers for image_set in ("U", "V", "R")]
    pairs = []
    for f in fingers:
        for probe_set, scenario in PROBE_SETS.items():
            for s in subjects:
                probe = index[(probe_set, s, f)]
                pairs.append(pair_record(scenario, "genuine", probe, index[("V", s, f)]))
                pairs.extend(
                    pair_record(scenario, "impostor", probe, index[("V", other, f)]) for other in subjects if other != s
                )
    return subjects, fingers, sample_images, pairs


def pilot_sample(
    images: list[dict], master_seed: str, dev_fraction: float, seed_images: list[dict], n_images: int = 100, n_pairs: int = 2000
) -> tuple[list[dict], list[dict]]:
    """Timing-pilot sample: `seed_images` (the F1 images) plus development images up to `n_images`,
    and up to `n_pairs` probe (U or R) x reference (V) pairs among them, both by keyed hash."""
    development, _ = development_subjects(images, master_seed, dev_fraction)
    taken = {row["image_id"] for row in seed_images}
    pool = [
        row
        for row in images
        if row["set"] in ("U", "V", "R") and not int(row["excluded"]) and row["subject"] in development and row["image_id"] not in taken
    ]
    extra = keyed_rank(pool, master_seed, "pilot-images", key=lambda row: row["image_id"])[: n_images - len(seed_images)]
    pilot_images = list(seed_images) + extra
    probes = [row for row in pilot_images if row["set"] in PROBE_SETS]
    references = [row for row in pilot_images if row["set"] == "V"]
    candidates = [(p, r) for p in probes for r in references]
    chosen = keyed_rank(candidates, master_seed, "pilot-pairs", key=lambda c: f"{c[0]['image_id']}|{c[1]['image_id']}")[:n_pairs]
    pairs = []
    for probe, reference in chosen:
        same = probe["subject"] == reference["subject"] and int(probe["frgp"]) == int(reference["frgp"])
        pairs.append(pair_record("pilot", "genuine" if same else "impostor", probe, reference))
    return pilot_images, sorted(pairs, key=lambda p: (p["probe_image_id"], p["reference_image_id"]))
