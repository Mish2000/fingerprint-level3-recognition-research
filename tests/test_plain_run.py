import math
from collections import Counter

import numpy as np

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


def test_a_plain_run_writes_its_results_and_scores(tmp_path, monkeypatch):
    import csv
    import json
    import time

    from fpl3.experiments import full_run
    from fpl3.experiments.common import write_json

    monkeypatch.setattr(full_run, "CHUNK_SIZE", {key: 3 for key in full_run.CHUNK_SIZE})
    monkeypatch.setattr(plain_run, "REUSE", {"opencv-sift": {}, "pore-sift": {}})
    rows = [{"image_id": f"{s}_{n:08d}_02", "set": s, "subject": f"{n:08d}", "frgp": "2"} for s in "RJ" for n in range(4)]
    half = {"00000000": "development", "00000001": "development", "00000002": "test", "00000003": "test"}
    pairs = [{"pair_id": f"RxJ:{p['image_id']}:{j['image_id']}", "scenario": "RxJ", "half": half[p["subject"]],
              "kind": "genuine" if p["subject"] == j["subject"] else "impostor", "frgp": 2,
              "probe_image_id": p["image_id"], "reference_image_id": j["image_id"],
              "probe_subject": p["subject"], "reference_subject": j["subject"]}
             for p in rows if p["set"] == "R" for j in rows if j["set"] == "J" and half[p["subject"]] == half[j["subject"]]]

    def executor(stage, todo, ctx, on_done, tick):
        for index, chunk in todo:
            results = ([{"image_id": r["image_id"]} for r in chunk] if stage.kind == "extract" else
                       [{"pair_id": p["pair_id"], "status": "ok", "reason": "", "score": 20 if p["kind"] == "genuine" else 1} for p in chunk])
            on_done(index, results, time.time(), time.time() + 0.1)

    stages = plain_run.build_stages(rows, pairs)
    ctx = full_run.Context(tmp_path, tmp_path, 2, 0, {})
    write_json(tmp_path / "report.json", {"algorithms": {}})
    progress = full_run.Progress(tmp_path, stages, 2, time.time(), page=plain_run.PAGE)
    full_run.execute(ctx, stages, progress, {"opencv-sift": executor, "pore-sift": executor},
                     after_stage=lambda stage: plain_run.write_results(ctx, progress, stage))
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    for algorithm in ("opencv-sift", "pore-sift"):
        test = report["algorithms"][algorithm]["test"]["RxJ"]["at_far"]["1%"]
        assert (test["genuine"], test["true_accepts"], test["impostor"]) == (2, 2, 2)
        with open(tmp_path / "scores" / f"{algorithm}.csv", newline="", encoding="utf-8") as f:
            assert len(list(csv.DictReader(f))) == len(pairs) == 8


def scores_of(genuine, impostor):
    from fpl3.eval import verification as v

    g, i = np.array(genuine, float), np.array(impostor, float)
    return v.Scores(g, np.zeros(len(g), int), i, np.zeros(len(i), int), np.zeros(len(i), int), 1)


def test_operational_threshold_comes_from_the_fit_pairs_only():
    from fpl3.experiments.plain_eval import operational

    fit = scores_of([5, 9, 12], [0] * 98 + [7, 10])  # at FAR 1 %, one of 100 impostors may pass: threshold 7
    r = operational(fit, scores_of([8, 6, 20, -np.inf], [7.5, 3, 11, 7]), 0.01)
    assert r["accept_if_score_above"] == 7
    assert (r["true_accepts"], r["genuine"], r["false_accepts"], r["impostor"]) == (2, 4, 2, 4)


def test_pore_file_falls_back_to_the_plain_run(tmp_path, monkeypatch):
    monkeypatch.setattr(pores, "METHOD_PORES", tmp_path / "method")
    monkeypatch.setattr(pores, "PLAIN_RUN_PORES", tmp_path / "plain")
    (tmp_path / "method").mkdir()
    (tmp_path / "method" / "J_A_03.npz").write_bytes(b"")
    assert pores.pore_file("J_A_03") == tmp_path / "method" / "J_A_03.npz"
    assert pores.pore_file("Q_B_04") == tmp_path / "plain" / "Q_B_04.npz"
    assert pores.pore_file("V_B_04") == pores.FULL_RUN_PORES / "V_B_04.npz"
