import math
from collections import Counter

from fpl3.data.samples import development_subjects, pilot_sample_plain
from fpl3.experiments import plain_run
from fpl3.experiments.common import read_images
from fpl3.method import label_audit, pores


def test_plain_design_matches_the_approved_counts(protocol):
    images = read_images()
    rows, pairs = plain_run.plain_design(images, protocol)
    assert Counter((p["scenario"], p["half"], p["kind"]) for p in pairs) == {
        ("RxJ", "development", "genuine"): 194, ("RxJ", "development", "impostor"): 28_602,
        ("RxJ", "test", "genuine"): 226, ("RxJ", "test", "impostor"): 29_628,
        ("QxJ", "development", "genuine"): 432, ("QxJ", "development", "impostor"): 33_372,
        ("QxJ", "test", "genuine"): 423, ("QxJ", "test", "impostor"): 34_623,
    }
    assert len({p["pair_id"] for p in pairs}) == 127_500
    development, _ = development_subjects(images, protocol["randomness"]["master_seed"], protocol["split"]["dev_fraction"])
    for p in pairs:
        assert p["reference_image_id"].startswith("J_") and p["probe_image_id"][0] in "RQ" and p["frgp"] not in (1, 6)
        assert (p["probe_subject"] == p["reference_subject"]) == (p["kind"] == "genuine")
        assert (p["probe_subject"] in development) == (p["reference_subject"] in development) == (p["half"] == "development")
    assert Counter(r["set"] for r in rows) == {"J": 1_275, "Q": 864, "R": 736}


def test_the_plain_pilot_sample_is_fixed_and_inside_development(protocol):
    images = read_images()
    seed, fraction = protocol["randomness"]["master_seed"], protocol["split"]["dev_fraction"]
    rows, pairs = pilot_sample_plain(images, seed, fraction)
    again, pairs_again = pilot_sample_plain(images, seed, fraction)
    development, _ = development_subjects(images, seed, fraction)
    assert [r["image_id"] for r in rows] == [r["image_id"] for r in again] and pairs == pairs_again
    assert len(rows) == 100 and 0 < len(pairs) <= 2000
    assert all(r["subject"] in development and r["set"] in "RQJ" and int(r["frgp"]) not in (1, 6) for r in rows)
    assert all(p["probe_image_id"][0] in "RQ" and p["reference_image_id"][0] == "J" for p in pairs)


def v(subject, frgp):
    return {"image_id": f"V_{subject}_{frgp:02d}", "subject": subject, "frgp": str(frgp)}


def test_label_audit_rules():
    row = {"image_id": "J_A_03", "set": "J", "subject": "A", "frgp": "3"}
    references = [v("A", 3), v("A", 4), v("B", 3), v("B", 4), v("C", 3)]
    assert [r["image_id"] for r in label_audit.candidate_references(row, references)] == ["V_A_03", "V_A_04", "V_B_03", "V_C_03"]
    assert label_audit.threshold([float(s) for s in range(10_000)]) == 9_998.0  # one of 10,000 lies above
    assert label_audit.threshold([1.0]) == 1.0 and label_audit.threshold([1.0], 1.0) == -math.inf
    candidates = label_audit.candidate_references(row, references)

    def category(scores):
        return label_audit.classify(row, list(zip(candidates, scores)), 10)["category"]

    assert category([40, 5, 3, 2]) == "confirmed"
    assert category([4, 40, 3, 2]) == "suspected"  # another finger of the same subject
    assert category([4, 5, 40, 2]) == "suspected"  # another subject
    assert category([8, 5, 3, 2]) == "undetermined"  # nothing clear
    assert category([40, 41, 3, 2]) == "suspected"
    record = label_audit.classify(row, list(zip(candidates, [4, 5, 40, 2])), 10)
    assert (record["best_other_image_id"], record["best_other_same_subject"]) == ("V_B_03", 0)
    lonely = label_audit.classify(row, list(zip(candidates[1:], [5, 3, 2])), 10)
    assert lonely["category"] == "no_reference" and lonely["own_score"] == ""


def test_pore_file_falls_back_to_the_plain_run(tmp_path, monkeypatch):
    monkeypatch.setattr(pores, "METHOD_PORES", tmp_path / "method")
    monkeypatch.setattr(pores, "PLAIN_RUN_PORES", tmp_path / "plain")
    (tmp_path / "method").mkdir()
    (tmp_path / "method" / "J_A_03.npz").write_bytes(b"")
    assert pores.pore_file("J_A_03") == tmp_path / "method" / "J_A_03.npz"
    assert pores.pore_file("Q_B_04") == tmp_path / "plain" / "Q_B_04.npz"
    assert pores.pore_file("V_B_04") == pores.FULL_RUN_PORES / "V_B_04.npz"
