from collections import Counter

import pytest

from fpl3.data.samples import development_subjects, f1_sample, pilot_sample, usable_index
from fpl3.eval.metrics import auc, summary
from fpl3.experiments.common import read_images, workload


def test_auc_extremes_and_ties():
    assert auc([3, 4], [1, 2]) == 1.0
    assert auc([1, 2], [3, 4]) == 0.0
    assert auc([5, 5], [5, 5]) == 0.5
    assert auc([2, 4], [1, 3]) == 0.75


def test_auc_needs_both_classes():
    with pytest.raises(ValueError):
        auc([], [1])


def test_summary():
    s = summary([float(v) for v in range(1, 101)])
    assert (s["count"], s["median"], s["p95"], s["min"], s["max"]) == (100, 50.5, 95.0, 1.0, 100.0)


@pytest.fixture(scope="module")
def images():
    return read_images()


def test_f1_sample_follows_the_approved_protocol(images, protocol):
    seed, fraction = protocol["randomness"]["master_seed"], protocol["split"]["dev_fraction"]
    subjects, fingers, sample, pairs = f1_sample(images, seed, fraction)
    development, strata = development_subjects(images, seed, fraction)
    assert len(subjects) == 10 and len(fingers) == 2 and len(sample) == 60
    assert all(s in development and strata[s] == "has_R" for s in subjects)
    assert all(not int(r["excluded"]) for r in sample)
    assert Counter((p["scenario"], p["kind"]) for p in pairs) == {
        ("UxV", "genuine"): 20, ("UxV", "impostor"): 180, ("RxV", "genuine"): 20, ("RxV", "impostor"): 180,
    }
    assert all(p["reference_image_id"].startswith("V_") for p in pairs)
    assert (subjects, fingers, sample, pairs) == f1_sample(list(reversed(images)), seed, fraction)


def test_pilot_sample_extends_f1_within_development(images, protocol):
    seed, fraction = protocol["randomness"]["master_seed"], protocol["split"]["dev_fraction"]
    _, _, f1_images, _ = f1_sample(images, seed, fraction)
    pilot_images, pairs = pilot_sample(images, seed, fraction, f1_images)
    development, _ = development_subjects(images, seed, fraction)
    assert len(pilot_images) == 100 and len(pairs) == 2000
    assert pilot_images[:60] == f1_images
    assert all(r["subject"] in development for r in pilot_images)
    assert len({p["pair_id"] for p in pairs}) == 2000


def test_workload_counts_match_the_verified_numbers(images):
    import csv

    from fpl3.experiments.common import REPO_ROOT

    with open(REPO_ROOT / "manifests" / "pairs.csv", newline="", encoding="utf-8") as f:
        compact = list(csv.DictReader(f))
    counts = workload(images, compact)
    assert counts["compact"]["comparisons"] == 22_913
    assert counts["full"] == {"images": 4_915, "comparisons": 582_724}
    assert len(usable_index(images)) == 7_059 - 5
