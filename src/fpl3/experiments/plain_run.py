"""Plain vs plain run: R and Q probes against J references at native 1000 ppi (decision record 2026-10-19).

    python -m fpl3.experiments.plain_run --until extract   # only the missing features (B4)
    python -m fpl3.experiments.plain_run                   # every stage; continues where it stopped
    python -m fpl3.experiments.plain_run --status          # progress and the expected finish
    python -m fpl3.experiments.plain_run --stop            # stop; the start command continues it

Scenarios RxJ and QxJ (B1/B2): every probe against every J reference of the same finger position,
only for pairs whose two subjects are both development or both test (B3/C1). Features are reused
where they exist (R from the full run, development Q and J pore detections from runs/method/pores)
and hard-linked into <out>/features, so each algorithm reads one folder without a second copy; the
rest is extracted here: OpenCV SIFT for Q and J, Pore SIFT for test Q and J. The machinery is the
full run's: chunks saved as they finish, a fixed plan, a progress page. Results go to
<out>/report.json and <out>/scores/<algorithm>.csv; features never enter git (R8).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from collections import Counter
from pathlib import Path

from ..algorithms import opencv_sift, pore_sift
from ..data.build import write_csv
from ..data.pairs import all_impostor_pairs, genuine_pairs
from ..data.samples import development_subjects
from . import full_run
from .common import REPO_ROOT, host, load_toml, read_images, write_json

SCENARIOS = {"RxJ": ("R", "J"), "QxJ": ("Q", "J")}  # probe set, reference set (B1/B2)
REUSE = {  # algorithm -> image set -> folder whose feature files this run reuses (B4)
    "opencv-sift": {"R": REPO_ROOT / "runs" / "full" / "features" / "opencv-sift"},
    "pore-sift": {
        "R": REPO_ROOT / "runs" / "full" / "features" / "pore-sift",
        "Q": REPO_ROOT / "runs" / "method" / "pores",
        "J": REPO_ROOT / "runs" / "method" / "pores",
    },
}
HALVES = ("development", "test")
SET_ORDER = {"J": 0, "Q": 1, "R": 2}
SCORE_FIELDS = ("pair_id", "scenario", "half", "kind", "frgp", "probe_image_id", "reference_image_id", "status", "reason", "score")
PAGE = {
    "module": "fpl3.experiments.plain_run",
    "title": "שטוחה מול שטוחה",
    "empty_results": "התוצאות נכתבות ל-report.json בסוף כל שלב השוואות.",
}


def plain_design(images: list[dict], protocol: dict) -> tuple[list[dict], list[dict]]:
    """Images and pairs of RxJ and QxJ. Each pair carries its C1 half; pairs whose subjects sit in
    different halves are not used (C1)."""
    seed, fraction = protocol["randomness"]["master_seed"], protocol["split"]["dev_fraction"]
    development, _ = development_subjects(images, seed, fraction)
    pairs = []
    for name, (probe_set, reference_set) in SCENARIOS.items():
        for p in genuine_pairs(images, name, probe_set, reference_set) + all_impostor_pairs(images, name, probe_set, reference_set):
            sides = {p["probe_subject"] in development, p["reference_subject"] in development}
            if len(sides) == 1:
                pairs.append({**p, "half": HALVES[0] if sides == {True} else HALVES[1]})
    used = {p["probe_image_id"] for p in pairs} | {p["reference_image_id"] for p in pairs}
    return [r for r in images if r["image_id"] in used], pairs


def missing(rows: list[dict], algorithm: str) -> list[dict]:
    """Images this run extracts: those whose features no reuse folder holds."""
    reuse = REUSE[algorithm]
    return [r for r in rows if r["set"] not in reuse or not (reuse[r["set"]] / f"{r['image_id']}.npz").exists()]


def link_reused(rows: list[dict], ctx: full_run.Context) -> int:
    """Hard-link the reused feature files into this run's folders (same volume, no copy)."""
    linked = 0
    for algorithm, reuse in REUSE.items():
        folder = ctx.features_dir(algorithm)
        folder.mkdir(parents=True, exist_ok=True)
        for r in rows:
            if r["set"] not in reuse:
                continue
            source, target = reuse[r["set"]] / f"{r['image_id']}.npz", folder / f"{r['image_id']}.npz"
            if source.exists() and not target.exists():
                os.link(source, target)
                linked += 1
    return linked


def build_stages(rows: list[dict], pairs: list[dict]) -> list[full_run.Stage]:
    def image_order(r: dict) -> tuple:
        return SET_ORDER[r["set"]], r["image_id"]

    def pair_order(p: dict) -> tuple:
        return SET_ORDER[p["probe_image_id"][0]], p["frgp"], p["probe_image_id"], p["reference_image_id"]

    pairs = sorted(pairs, key=pair_order)
    return [
        full_run.Stage("opencv-sift", "extract", "", sorted(missing(rows, "opencv-sift"), key=image_order)),
        full_run.Stage("pore-sift", "extract", "", sorted(missing(rows, "pore-sift"), key=image_order)),
        full_run.Stage("pore-sift", "compare", "all", pairs),
        full_run.Stage("opencv-sift", "compare", "all", pairs),
    ]


def write_results(ctx: full_run.Context, progress: full_run.Progress, stage: full_run.Stage) -> None:
    """After an algorithm's comparisons: TAR, FRR and FAR per scenario and half (R6, E2) and its scores file."""
    if stage.kind != "compare":
        return
    path = ctx.run_dir / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    scores = full_run.stage_results(ctx.run_dir, stage)
    entry = {half: full_run.result_tables([p for p in stage.items if p["half"] == half], scores, full_run.FAR_TARGETS["full"])
             for half in HALVES}
    report["algorithms"][stage.algorithm] = {**entry, "runtime": full_run.runtime(progress, stage.algorithm)}
    rows = []
    for p in sorted(stage.items, key=lambda p: (p["scenario"], p["half"], p["frgp"], p["probe_image_id"], p["reference_image_id"])):
        r = scores[p["pair_id"]]
        rows.append({**{k: p[k] for k in SCORE_FIELDS[:7]}, "status": r["status"], "reason": r["reason"], "score": r["score"]})
    write_csv(ctx.run_dir / "scores" / f"{stage.algorithm}.csv", SCORE_FIELDS, rows)
    write_json(path, report)
    full_run.log(ctx.run_dir, f"{stage.name}: results written to report.json")


def finish_stage(ctx: full_run.Context, progress: full_run.Progress, stage: full_run.Stage) -> None:
    """A failure here does not stop the run: the scores are in the chunk files and the next start writes them."""
    try:
        write_results(ctx, progress, stage)
    except Exception:
        full_run.log(ctx.run_dir, f"{stage.name}: results not written:\n" + traceback.format_exc())
        progress.warnings.append(f"התוצאות של {stage.label()} לא נכתבו בגלל שגיאה; הפרטים ב-run.log.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "plain")
    parser.add_argument("--until", choices=("extract",), help="stop after the extraction stages")
    parser.add_argument("--status", action="store_true", help="print the progress and the expected finish")
    parser.add_argument("--stop", action="store_true", help="stop a running run; the start command continues it")
    args = parser.parse_args(argv)
    run_dir = args.out.resolve()
    if args.status:
        full_run.print_status(run_dir)
        return
    if args.stop:
        if not (run_dir / "progress.json").exists():
            raise SystemExit(f"no run in {run_dir}")
        (run_dir / "stop-requested").touch()
        print("stop requested: the run stops within a few seconds; the start command continues it")
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    lock = full_run.hold_lock(run_dir)
    full_run.end_workers_with_this_process()
    full_run.keep_awake()
    (run_dir / "stop-requested").unlink(missing_ok=True)
    protocol, algorithms = load_toml("protocol.toml"), load_toml("algorithms.toml")
    settings = pore_sift.resolve(algorithms["pore_sift"], REPO_ROOT)
    pore_sift.check_upstream(settings)
    ctx = full_run.Context(run_dir, Path(protocol["dataset"]["root"]), args.workers,
                           opencv_sift.rng_seed(protocol["randomness"]["master_seed"]), settings)
    rows, pairs = plain_design(read_images(), protocol)
    stages = build_stages(rows, pairs)
    plan = full_run.plan_record(stages, ctx, protocol, algorithms)
    plan_path = run_dir / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8"))["plan"] != plan:
            raise SystemExit(f"{run_dir} holds a run with other inputs; choose another --out")
    else:
        write_json(plan_path, {"plan": plan, "created": time.time()})
        design = {"images": len(rows), "pairs": dict(sorted(Counter(f"{p['scenario']} {p['half']} {p['kind']}" for p in pairs).items())),
                  "extracted_here": {s.algorithm: len(s.items) for s in stages if s.kind == "extract"}}
        write_json(run_dir / "report.json", {"design": design, "far_rule": full_run.FAR_RULE, "host": host(), "algorithms": {}})
    linked = link_reused(rows, ctx)
    progress = full_run.Progress(run_dir, stages, args.workers, json.loads(plan_path.read_text(encoding="utf-8"))["created"], page=PAGE)
    full_run.check_disk(run_dir, stages, progress)
    todo = [s for s in stages if s.kind == "extract"] if args.until == "extract" else stages
    done = sum(len(d) for d in progress.done.values())
    full_run.log(run_dir, f"start: {args.workers} workers, code {full_run.git_state()}, {linked} reused feature files linked, "
                          f"{done} of {sum(s.n_chunks for s in stages)} chunks already done, until {args.until or 'the end'}")
    try:
        full_run.execute(ctx, todo, progress, after_stage=lambda stage: finish_stage(ctx, progress, stage))
        progress.state = "finished" if todo is stages else "stopped"
    except (full_run.StopRequested, KeyboardInterrupt):
        progress.state = "stopped"
        (run_dir / "stop-requested").unlink(missing_ok=True)
    except BaseException as error:
        progress.state, progress.error = "failed", f"{type(error).__name__}: {error}"
        full_run.log(run_dir, "failed:\n" + traceback.format_exc())
        raise
    finally:
        progress.write(force=True)
        full_run.log(run_dir, progress.state)
        lock.close()


if __name__ == "__main__":
    main()
