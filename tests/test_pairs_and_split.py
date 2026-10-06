from collections import Counter

import pytest

from fpl3.data.manifest import image_id
from fpl3.data.pairs import genuine_pairs, impostor_pairs
from fpl3.data.sampling import keyed_rank
from fpl3.data.split import has_r_strata, stratified_subject_split

SUBJECTS = [f"0000{n:04d}" for n in range(2300, 2312)]


def synthetic_manifest(excluded: set[str] = frozenset()) -> list[dict]:
    rows = []
    for set_code, subjects in (("U", SUBJECTS), ("V", SUBJECTS), ("R", SUBJECTS[:5])):
        for subject in subjects:
            for frgp in range(1, 11):
                iid = image_id(set_code, subject, frgp)
                rows.append({"image_id": iid, "set": set_code, "subject": subject, "frgp": frgp, "excluded": int(iid in excluded)})
    return rows


def test_genuine_pairs_join_on_subject_and_finger():
    pairs = genuine_pairs(synthetic_manifest(), "UxV", "U", "V")
    assert len(pairs) == len(SUBJECTS) * 10
    assert all(p["probe_subject"] == p["reference_subject"] for p in pairs)
    assert all(p["probe_image_id"].split("_")[1:] == p["reference_image_id"].split("_")[1:] for p in pairs)


def test_excluded_images_never_enter_pairs():
    excluded = {image_id("V", SUBJECTS[0], 4), image_id("R", SUBJECTS[1], 1)}
    manifest = synthetic_manifest(excluded)
    pairs = genuine_pairs(manifest, "UxV", "U", "V") + genuine_pairs(manifest, "RxV", "R", "V")
    pairs += impostor_pairs(manifest, "UxV", "U", "V", 20, "seed") + impostor_pairs(manifest, "RxV", "R", "V", 20, "seed")
    used = {p["probe_image_id"] for p in pairs} | {p["reference_image_id"] for p in pairs}
    assert used.isdisjoint(excluded)
    assert len(genuine_pairs(manifest, "UxV", "U", "V")) == len(SUBJECTS) * 10 - 1


def test_impostor_pairs_are_same_finger_different_subject_unique_and_sized():
    pairs = impostor_pairs(synthetic_manifest(), "RxV", "R", "V", 30, "seed")
    assert len(pairs) == 300
    assert Counter(p["frgp"] for p in pairs) == {f: 30 for f in range(1, 11)}
    assert all(p["probe_subject"] != p["reference_subject"] for p in pairs)
    assert all(p["probe_image_id"].endswith(f"_{p['frgp']:02d}") and p["reference_image_id"].endswith(f"_{p['frgp']:02d}") for p in pairs)
    assert len({p["pair_id"] for p in pairs}) == len(pairs)


def test_impostor_sampling_is_deterministic_and_seed_dependent():
    manifest = synthetic_manifest()
    first = impostor_pairs(manifest, "UxV", "U", "V", 25, "seed-a")
    assert first == impostor_pairs(list(reversed(manifest)), "UxV", "U", "V", 25, "seed-a")
    assert first != impostor_pairs(manifest, "UxV", "U", "V", 25, "seed-b")


def test_impostor_sampling_refuses_too_few_candidates():
    with pytest.raises(ValueError):
        impostor_pairs(synthetic_manifest(), "RxV", "R", "V", 10_000, "seed")


def test_keyed_rank_ignores_input_order_and_separates_purposes():
    items = [str(n) for n in range(50)]
    ranked = keyed_rank(items, "seed", "p1", key=str)
    assert ranked == keyed_rank(list(reversed(items)), "seed", "p1", key=str)
    assert ranked != keyed_rank(items, "seed", "p2", key=str)


def test_stratified_split_halves_each_stratum_deterministically():
    strata = has_r_strata(synthetic_manifest())
    assert Counter(strata.values()) == {"has_R": 5, "no_R": 7}
    split = stratified_subject_split(strata, 0.5, "seed")
    assert split == stratified_subject_split(dict(reversed(list(strata.items()))), 0.5, "seed")
    dev = Counter(strata[s] for s, role in split.items() if role == "dev")
    assert dev == {"has_R": round(5 * 0.5), "no_R": round(7 * 0.5)}
