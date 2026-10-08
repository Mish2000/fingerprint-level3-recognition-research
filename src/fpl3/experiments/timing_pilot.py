"""Timing pilot (B4, E5): measure an algorithm on 100 development images and 2,000 comparisons,
check determinism, and project the compact and full runs.

    python -m fpl3.experiments.timing_pilot --algorithm opencv-sift [--workers 12]

Steps: single-thread extraction of the 100 images; single-thread comparison of the first
200 pairs; parallel extraction of the same images; parallel comparison of all 2,000 pairs;
features and the 200 scores must match between the single-thread and parallel runs.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ..algorithms import opencv_sift
from ..data.build import write_csv
from ..data.samples import f1_sample, pilot_sample
from ..eval.metrics import summary
from .common import REPO_ROOT, host, image_path, load_toml, new_run_dir, read_images, read_pairs, save_sample, workload, write_json
from .sample_run import run_opencv_sift, run_pore_sift

ALGORITHMS = ("opencv-sift", "pore-sift")  # the algorithms with a timing pilot so far

SINGLE_THREAD_PAIRS = 200


def single_thread(algorithm, rows, pairs, dataset_root, features_dir, work_dir, seed):
    if algorithm == "opencv-sift":
        opencv_sift.init_worker(seed)
        extraction = [opencv_sift.extract_job((r["image_id"], image_path(r, dataset_root), str(features_dir))) for r in rows]
        comparisons = [opencv_sift.compare_job((p, str(features_dir))) for p in pairs]
        return extraction, comparisons
    extraction, comparisons, _, _ = run_pore_sift(rows, pairs, dataset_root, features_dir, 1, work_dir)
    return extraction, comparisons


def same_features(a: Path, b: Path, image_ids: list[str]) -> bool:
    for image_id in image_ids:
        with np.load(a / f"{image_id}.npz") as x, np.load(b / f"{image_id}.npz") as y:
            if x.files != y.files or not all(np.array_equal(x[k], y[k]) for k in x.files):
                return False
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Timing pilot")
    parser.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    protocol = load_toml("protocol.toml")
    master_seed, fraction = protocol["randomness"]["master_seed"], protocol["split"]["dev_fraction"]
    dataset_root = Path(protocol["dataset"]["root"])
    rng = opencv_sift.rng_seed(master_seed)
    images = read_images()
    _, _, f1_images, _ = f1_sample(images, master_seed, fraction)
    rows, pairs = pilot_sample(images, master_seed, fraction, f1_images)
    save_sample("pilot", rows, pairs)

    run_dir = new_run_dir(args.out or REPO_ROOT / "runs" / f"pilot-{args.algorithm}")
    single_dir, parallel_dir = run_dir / "features-single", run_dir / "features-parallel"
    single_dir.mkdir()
    parallel_dir.mkdir()

    start = time.perf_counter()
    single_extraction, single_comparisons = single_thread(
        args.algorithm, rows, pairs[:SINGLE_THREAD_PAIRS], dataset_root, single_dir, run_dir / "work-single", rng
    )
    single_wall = time.perf_counter() - start
    if args.algorithm == "opencv-sift":
        _, parallel_comparisons, extract_wall, compare_wall = run_opencv_sift(rows, pairs, dataset_root, parallel_dir, args.workers, rng)
    else:
        _, parallel_comparisons, extract_wall, compare_wall = run_pore_sift(
            rows, pairs, dataset_root, parallel_dir, args.workers, run_dir / "work-parallel"
        )

    parallel_scores = {c["pair_id"]: c["score"] for c in parallel_comparisons}
    fields = ("pair_id", "status", "score")
    write_csv(run_dir / "scores-single.csv", fields, [{k: c[k] for k in fields} for c in single_comparisons])
    write_csv(run_dir / "scores-parallel.csv", fields, [{k: c[k] for k in fields} for c in parallel_comparisons])
    extract_times = [e["extract_s"] for e in single_extraction]
    compare_times = [c["load_s"] + c["compare_s"] for c in single_comparisons]
    images_per_s = len(rows) / extract_wall
    comparisons_per_s = len(pairs) / compare_wall
    projections = {}
    for design, counts in workload(images, read_pairs()).items():
        single_hours = (counts["images"] * np.mean(extract_times) + counts["comparisons"] * np.mean(compare_times)) / 3600
        parallel_hours = (counts["images"] / images_per_s + counts["comparisons"] / comparisons_per_s) / 3600
        projections[design] = {**counts, "single_thread_hours": round(float(single_hours), 2),
                               "parallel_hours": round(float(parallel_hours), 2)}

    report = {
        "algorithm": args.algorithm,
        "pilot": {"images": len(rows), "pairs": len(pairs), "single_thread_pairs": SINGLE_THREAD_PAIRS},
        "single_thread_seconds": {"extract_per_image": summary(extract_times), "compare_per_pair": summary(compare_times),
                                  "wall": single_wall},
        "parallel": {"workers": args.workers, "extract_wall_s": extract_wall, "compare_wall_s": compare_wall,
                     "images_per_s": images_per_s, "comparisons_per_s": comparisons_per_s},
        "deterministic": {
            "features": same_features(single_dir, parallel_dir, [r["image_id"] for r in rows]),
            "scores": all(parallel_scores[c["pair_id"]] == c["score"] for c in single_comparisons),
        },
        "feature_storage_mb_per_image": float(np.mean([e["bytes"] for e in single_extraction]) / 1e6),
        "projections": projections,
        "host": host(),
    }
    write_json(run_dir / "report.json", report)
    s = report["single_thread_seconds"]
    print(f"single thread: extract median {s['extract_per_image']['median']:.2f}s/image, "
          f"compare median {s['compare_per_pair']['median']:.3f}s/pair")
    print(f"parallel ({args.workers}): {images_per_s:.1f} images/s, {comparisons_per_s:.1f} comparisons/s")
    print(f"deterministic: {report['deterministic']}")
    for design, p in projections.items():
        print(f"{design}: {p['images']} images, {p['comparisons']} comparisons -> "
              f"{p['parallel_hours']} h parallel ({p['single_thread_hours']} h single thread)")


if __name__ == "__main__":
    main()
