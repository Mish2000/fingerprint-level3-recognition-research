"""Integration tests against the local SD302 copy; the expected numbers were verified on 2026-10-05/06."""

from collections import Counter

import pytest

from conftest import REPO_ROOT
from fpl3.data.build import build
from fpl3.data.manifest import build_image_manifest, load_exclusions
from fpl3.data.pairs import impostor_pairs
from fpl3.data.split import has_r_strata, stratified_subject_split

pytestmark = pytest.mark.dataset

EXPECTED_IMAGES = {"U": 2000, "V": 2000, "R": 920, "Q": 864, "J": 1275}
EXPECTED_SUBJECTS = {"U": 200, "V": 200, "R": 92, "Q": 108, "J": 161}
EXPECTED_EXCLUDED = {"V_00002361_02", "V_00002361_04", "R_00002420_01", "R_00002420_06", "V_00002472_05"}
EXPECTED_GENUINE = {"UxV": 1997, "RxV": 916}


@pytest.fixture(scope="module")
def built(sd302_root, tmp_path_factory):
    out = tmp_path_factory.mktemp("manifests")
    report = build(REPO_ROOT / "configs" / "protocol.toml", REPO_ROOT / "configs" / "exclusions.csv", out)
    return report, out


def test_image_sets_match_verified_counts(built):
    report, _ = built
    assert {code: c["total"] for code, c in report["images"].items()} == EXPECTED_IMAGES
    assert {code: c["subjects"] for code, c in report["images"].items()} == EXPECTED_SUBJECTS


def test_every_image_passes_checksum_and_is_native_1000_ppi(built):
    report, _ = built
    assert report["checksum"] == {"ok": sum(EXPECTED_IMAGES.values())}
    assert report["header_ppi"] == {"1000.0x1000.0": sum(EXPECTED_IMAGES.values())}
    assert report["dataset_problems"] == []


def test_only_the_approved_exclusions_apply(built):
    report, _ = built
    assert set(report["excluded_images"]) == EXPECTED_EXCLUDED


def test_scenario_pair_counts(built, protocol):
    report, _ = built
    for name, expected in EXPECTED_GENUINE.items():
        scenario = report["scenarios"][name]
        per_finger = protocol["scenarios"][name]["impostors_per_finger"]
        assert scenario["genuine"] == expected
        assert scenario["impostor"] == 10 * per_finger
        assert set(scenario["impostor_per_finger"].values()) == {per_finger}


def test_rebuild_is_byte_identical(built, sd302_root, tmp_path):
    _, first = built
    build(REPO_ROOT / "configs" / "protocol.toml", REPO_ROOT / "configs" / "exclusions.csv", tmp_path)
    for name in ("images.csv", "pairs.csv", "build_report.json"):
        assert (first / name).read_bytes() == (tmp_path / name).read_bytes()


def test_impostors_avoid_same_subject_and_excluded_images(built, sd302_root, protocol):
    images, _ = build_image_manifest(sd302_root, load_exclusions(REPO_ROOT / "configs" / "exclusions.csv"))
    for name, scenario in protocol["scenarios"].items():
        pairs = impostor_pairs(images, name, scenario["probe"], scenario["reference"], scenario["impostors_per_finger"], protocol["randomness"]["master_seed"])
        assert all(p["probe_subject"] != p["reference_subject"] for p in pairs)
        used = {p["probe_image_id"] for p in pairs} | {p["reference_image_id"] for p in pairs}
        assert used.isdisjoint(EXPECTED_EXCLUDED)


def test_split_would_halve_r_and_non_r_subjects(built, sd302_root, protocol):
    images, _ = build_image_manifest(sd302_root, load_exclusions(REPO_ROOT / "configs" / "exclusions.csv"))
    strata = has_r_strata([row for row in images if row["set"] in ("U", "V", "R")])
    split = stratified_subject_split(strata, protocol["split"]["dev_fraction"], protocol["randomness"]["master_seed"])
    assert Counter((strata[s], role) for s, role in split.items()) == {
        ("has_R", "dev"): 46,
        ("has_R", "test"): 46,
        ("no_R", "dev"): 54,
        ("no_R", "test"): 54,
    }
