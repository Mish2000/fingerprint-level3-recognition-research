"""Run an approved algorithm on a saved sample and report plain TAR / FAR / FRR.

    python -m fpl3.experiments.sample_run --algorithm opencv-sift --sample f1 [--workers 12] [--out runs/<sample>-<algorithm>]

The sample is read from manifests/<sample>_images.csv and manifests/<sample>_pairs.csv.
Results are reported at the strictest operating point: no impostor accepted.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

from ..algorithms import opencv_sift, pore_sift
from ..data.build import write_csv
from ..eval.metrics import no_false_accept_point
from .common import REPO_ROOT, host, image_path, load_toml, new_run_dir, read_images, write_json

ALGORITHMS = ("opencv-sift", "pore-sift")
SCORE_FIELDS = ("pair_id", "scenario", "kind", "frgp", "probe_image_id", "reference_image_id", "status", "reason", "score")


def read_sample(name: str) -> tuple[list[dict], list[dict]]:
    with open(REPO_ROOT / "manifests" / f"{name}_images.csv", newline="", encoding="utf-8") as f:
        ids = [row["image_id"] for row in csv.DictReader(f)]
    with open(REPO_ROOT / "manifests" / f"{name}_pairs.csv", newline="", encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))
    rows = {row["image_id"]: row for row in read_images()}
    return [rows[i] for i in ids], pairs


def run_opencv_sift(rows, pairs, dataset_root, features_dir, workers, seed):
    with Pool(workers, initializer=opencv_sift.init_worker, initargs=(seed,)) as pool:
        start = time.perf_counter()
        extraction = pool.map(opencv_sift.extract_job, [(r["image_id"], image_path(r, dataset_root), str(features_dir)) for r in rows])
        middle = time.perf_counter()
        comparisons = pool.map(opencv_sift.compare_job, [(p, str(features_dir)) for p in pairs], chunksize=4)
        end = time.perf_counter()
    return extraction, comparisons, middle - start, end - middle


def run_pore_sift(rows, pairs, dataset_root, features_dir, workers, work_dir):
    settings = pore_sift.resolve(load_toml("algorithms.toml")["pore_sift"], REPO_ROOT)
    pore_sift.check_upstream(settings)
    jobs = [{"image_id": r["image_id"], "path": image_path(r, dataset_root)} for r in rows]
    extraction, extract_wall = pore_sift.run_parallel("extract", jobs, workers, settings, features_dir, work_dir)
    comparisons, compare_wall = pore_sift.run_parallel("compare", pairs, workers, settings, features_dir, work_dir)
    return extraction, comparisons, extract_wall, compare_wall


def plain_results(pairs: list[dict], results: dict[str, dict]) -> dict:
    by_scenario = defaultdict(lambda: {"genuine": [], "impostor": [], "fingers": defaultdict(list)})
    for p in pairs:
        score = results[p["pair_id"]]["score"]
        group = by_scenario[p["scenario"]]
        group[p["kind"]].append(score)
        if p["kind"] == "genuine":
            group["fingers"][p["frgp"]].append(score)
    out = {}
    for scenario, group in sorted(by_scenario.items()):
        point = no_false_accept_point(group["genuine"], group["impostor"])
        top = point["accept_if_score_above"]
        point["genuine_accepted_by_finger"] = {
            finger: f"{sum(1 for s in scores if s is not None and (top is None or s > top))}/{len(scores)}"
            for finger, scores in sorted(group["fingers"].items(), key=lambda kv: int(kv[0]))
        }
        point["failures"] = sum(1 for s in group["genuine"] + group["impostor"] if s is None)
        out[scenario] = point
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    parser.add_argument("--sample", default="f1")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    protocol = load_toml("protocol.toml")
    dataset_root = Path(protocol["dataset"]["root"])
    seed = opencv_sift.rng_seed(protocol["randomness"]["master_seed"])
    rows, pairs = read_sample(args.sample)
    run_dir = new_run_dir(args.out or REPO_ROOT / "runs" / f"{args.sample}-{args.algorithm}")
    features_dir = run_dir / "features"
    features_dir.mkdir()
    if args.algorithm == "opencv-sift":
        extraction, comparisons, extract_wall, compare_wall = run_opencv_sift(rows, pairs, dataset_root, features_dir, args.workers, seed)
    else:
        extraction, comparisons, extract_wall, compare_wall = run_pore_sift(
            rows, pairs, dataset_root, features_dir, args.workers, run_dir / "work"
        )

    results = {c["pair_id"]: c for c in comparisons}
    write_csv(run_dir / "scores.csv", SCORE_FIELDS,
              [{**{k: p[k] for k in SCORE_FIELDS if k in p}, **{k: results[p["pair_id"]][k] for k in ("status", "reason", "score")}}
               for p in pairs])
    report = {
        "algorithm": args.algorithm,
        "sample": args.sample,
        "images": len(rows),
        "pairs": len(pairs),
        "results": plain_results(pairs, results),
        "execution": {"workers": args.workers, "extract_wall_s": extract_wall, "compare_wall_s": compare_wall, "host": host()},
    }
    write_json(run_dir / "report.json", report)
    for scenario, r in report["results"].items():
        print(f"{scenario}: TAR {r['true_accepts']}/{r['genuine']}  FRR {r['false_rejects']}/{r['genuine']}  "
              f"FAR {r['false_accepts']}/{r['impostor']}  (accept if score > {r['accept_if_score_above']})  "
              f"by finger {r['genuine_accepted_by_finger']}  failures {r['failures']}")
    print(f"extract {extract_wall:.0f}s, compare {compare_wall:.0f}s with {args.workers} workers")


if __name__ == "__main__":
    main()
