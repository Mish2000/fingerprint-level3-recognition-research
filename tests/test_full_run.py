import csv
import json
import sys
import time
from collections import Counter

import pytest

from fpl3.algorithms import pore_sift
from fpl3.eval.metrics import no_false_accept_point, tar_at_far
from fpl3.experiments import full_run
from fpl3.experiments.common import read_images, write_json


def test_tar_at_far_never_lets_ties_or_failures_exceed_the_target():
    r = tar_at_far([4, 5, 6, None], [0] * 95 + [1, 2, 3, 4, 5], 0.01)
    assert (r["accept_if_score_above"], r["false_accepts"], r["true_accepts"], r["false_rejects"]) == (4, 1, 2, 2)
    tied = tar_at_far([3, 4], [3, 3, 3] + [0] * 97, 0.01)
    assert (tied["accept_if_score_above"], tied["false_accepts"], tied["tar"]) == (3, 0, 0.5)
    everything = tar_at_far([1, None], [None, 1], 1.0)
    assert (everything["accept_if_score_above"], everything["false_accepts"], everything["true_accepts"]) == (None, 1, 1)


def test_tar_at_far_zero_is_the_no_false_accept_point():
    genuine, impostor = [5, 9, 12, None], [1, 7, 3, None]
    strict, r = no_false_accept_point(genuine, impostor), tar_at_far(genuine, impostor, 0.0)
    assert (r["accept_if_score_above"], r["true_accepts"]) == (strict["accept_if_score_above"], strict["true_accepts"])


def test_full_design_matches_the_approved_counts(protocol):
    images, compact, rest = full_run.full_design(read_images(), protocol)
    assert (len(images), len(compact), len(rest)) == (4_915, 22_913, 559_811)
    assert Counter((p["scenario"], p["kind"]) for p in compact + rest) == {
        ("UxV", "genuine"): 1_997, ("UxV", "impostor"): 397_403, ("RxV", "genuine"): 916, ("RxV", "impostor"): 182_408,
    }
    assert Counter((p["scenario"], p["kind"]) for p in rest) == {("UxV", "impostor"): 387_403, ("RxV", "impostor"): 172_408}
    assert len({p["pair_id"] for p in compact + rest}) == 582_724
    assert all(p["reference_image_id"].startswith("V_") and p["frgp"] == int(p["reference_image_id"][-2:]) for p in rest)
    assert all((p["probe_subject"] == p["reference_subject"]) == (p["kind"] == "genuine") for p in compact + rest)


def small_design():
    images = [{"image_id": f"{s}_{n:08d}_01", "set": s, "subject": f"{n:08d}", "frgp": "1"} for s in "UVR" for n in range(4)]
    references = [r for r in images if r["set"] == "V"]
    pairs = [
        {"pair_id": f"{p['set']}xV:{p['image_id']}:{v['image_id']}", "scenario": f"{p['set']}xV",
         "kind": "genuine" if p["subject"] == v["subject"] else "impostor", "frgp": 1,
         "probe_image_id": p["image_id"], "reference_image_id": v["image_id"],
         "probe_subject": p["subject"], "reference_subject": v["subject"]}
        for p in images if p["set"] in "UR" for v in references
    ]
    compact = [p for i, p in enumerate(pairs) if p["kind"] == "genuine" or i % 3 == 0]
    return images, compact, [p for p in pairs if p not in compact]


def fake_executor(calls, crash_after=None):
    def run(stage, todo, ctx, on_done, tick):
        for index, chunk in todo:
            if crash_after is not None and len(calls) >= crash_after:
                raise RuntimeError("simulated crash")
            calls.append((stage.name, index))
            if stage.kind == "extract":
                results = [{"image_id": item["image_id"]} for item in chunk]
            else:
                results = [{"pair_id": p["pair_id"], "status": "ok", "reason": "", "score": 20 if p["kind"] == "genuine" else 1}
                           for p in chunk]
            started = time.time()
            on_done(index, results, started, started + 0.5)
            tick()
    return run


def test_a_resumed_run_runs_only_the_missing_chunks_and_writes_every_result(tmp_path, monkeypatch):
    monkeypatch.setattr(full_run, "CHUNK_SIZE", {key: 3 for key in full_run.CHUNK_SIZE})
    images, compact, rest = small_design()
    stages = full_run.build_stages(images, compact, rest)
    ctx = full_run.Context(tmp_path, tmp_path, 2, 0, {})
    write_json(tmp_path / "report.json", {"algorithms": {}})

    def attempt(executor):
        progress = full_run.Progress(tmp_path, stages, 2, time.time())
        full_run.execute(ctx, stages, progress, {"opencv-sift": executor, "pore-sift": executor},
                         after_stage=lambda stage: full_run.finish_stage(ctx, stages, progress, stage))
        return progress

    first, second = [], []
    with pytest.raises(RuntimeError, match="simulated crash"):
        attempt(fake_executor(first, crash_after=6))
    progress = attempt(fake_executor(second))

    every = [(s.name, i) for s in stages for i in range(s.n_chunks)]
    assert sorted(first + second) == sorted(every) and not set(first) & set(second)
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    for algorithm in full_run.NAMES:
        uxv = report["algorithms"][algorithm]["full"]["UxV"]["at_far"]["0.01%"]
        assert (uxv["genuine"], uxv["true_accepts"], uxv["impostor"], uxv["false_accepts"]) == (4, 4, 12, 0)
        with open(tmp_path / "scores" / f"{algorithm}.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 32 and sum(int(r["compact"]) for r in rows) == len(compact)
    snapshot = progress.snapshot(time.time())
    assert snapshot["remaining_s"] == 0 and {s["status"] for s in snapshot["stages"]} == {"done"}
    progress.state = "finished"
    progress.write(force=True)
    assert "ההרצה הסתיימה" in (tmp_path / "progress.html").read_text(encoding="utf-8")


def test_forecast_scales_the_expected_time_by_the_measured_pace(tmp_path, monkeypatch):
    monkeypatch.setattr(full_run, "CHUNK_SIZE", {key: 3 for key in full_run.CHUNK_SIZE})
    stages = full_run.build_stages(*small_design())
    progress = full_run.Progress(tmp_path, stages, 1, time.time())
    extract = stages[0]
    chunk = extract.chunks()[0]
    expected = extract.expected_seconds(chunk)
    progress.record(extract, {"index": 0, "count": len(chunk), "expected_s": expected, "started": 0.0, "finished": 2 * expected})
    left = progress.snapshot(time.time())["stages"][0]["remaining_s"]
    assert left == pytest.approx(2 * (progress.expected[extract.name] - expected))


def test_hebrew_durations():
    assert full_run.hebrew_duration(13 * 3600 + 12 * 60) == "13 שעות ו-12 דקות"
    assert full_run.hebrew_duration(2 * 3600 + 60) == "שעתיים ודקה"
    assert full_run.hebrew_duration(45 * 60) == "45 דקות"
    assert (full_run.about("13 שעות"), full_run.about("שעתיים")) == ("כ-13 שעות", "כשעתיים")


FAKE_WORKER = """
import json, sys, time
with open(sys.argv[3]) as f:
    job = json.load(f)
items = job.get("pairs") or job.get("images")
if any(item.get("fail") for item in items):
    sys.exit(3)
time.sleep(0.3)
key = "pair_id" if "pairs" in job else "image_id"
with open(job["out_path"], "w") as f:
    json.dump([{key: item[key], "score": len(item[key])} for item in items], f)
"""


@pytest.fixture
def fake_worker(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(FAKE_WORKER, encoding="utf-8")
    settings = {"python": sys.executable, "survey_dir": "", "dahia_dir": "", "model_sha256": ""}
    return worker, settings


def test_pore_sift_chunks_share_a_bounded_number_of_workers(tmp_path, fake_worker):
    worker, settings = fake_worker
    chunks = [(f"{i:05d}", [{"pair_id": f"p{i}-{j}"} for j in range(3)]) for i in range(7)]
    done = []
    pore_sift.run_chunks("compare", chunks, 3, settings, tmp_path / "features", tmp_path / "work",
                         on_done=lambda key, results, started, ended: done.append((key, results, started, ended)), worker=worker)
    assert sorted(key for key, *_ in done) == [key for key, _ in chunks]
    assert all([r["pair_id"] for r in results] == [p["pair_id"] for p in dict(chunks)[key]] for key, results, _, _ in done)
    spans = [(started, ended) for _, _, started, ended in done]
    assert max(sum(1 for s, e in spans if s < end and start < e) for start, end in spans) <= 3
    assert not list((tmp_path / "work").glob("*.json"))


def test_a_failed_pore_sift_chunk_stops_the_run(tmp_path, fake_worker):
    worker, settings = fake_worker
    chunks = [("00000", [{"pair_id": "a"}]), ("00001", [{"pair_id": "b", "fail": True}]), ("00002", [{"pair_id": "c"}])]
    with pytest.raises(RuntimeError, match="00001"):
        pore_sift.run_chunks("compare", chunks, 2, settings, tmp_path / "features", tmp_path / "work",
                             on_done=lambda *args: None, worker=worker)
