"""F1 pore-feasibility gate (docs/algorithms/pore-sift.md, decision record 2026-10-06).

    python -m fpl3.experiments.f1 [--workers 12] [--out runs/f1]

Pass: AUC >= 0.95 in both UxV and RxV, plus the researcher's visual review of the
overlays written to <out>/overlays (local only, never committed).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ..algorithms import pore_sift
from ..data.samples import f1_sample
from ..eval.metrics import auc, summary
from .common import REPO_ROOT, host, image_path, load_toml, new_run_dir, read_images, save_sample, write_json

AUC_PASS = 0.95
ZOOM = 300  # crop side in pixels (0.3 inch at 1000 ppi)


def draw_overlays(rows: list[dict], dataset_root: Path, features_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir()
    for row in rows:
        gray = cv2.imread(image_path(row, dataset_root), cv2.IMREAD_GRAYSCALE)
        with np.load(features_dir / f"{row['image_id']}.npz") as data:
            points = data["points"]
        full = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for r, c in points:
            cv2.circle(full, (int(c), int(r)), 4, (0, 0, 255), 1)
        cv2.imwrite(str(out_dir / f"{row['image_id']}.png"), full)
        if len(points):
            cy, cx = (int(v) for v in np.median(points, axis=0))
            y0 = min(max(0, cy - ZOOM // 2), max(0, gray.shape[0] - ZOOM))
            x0 = min(max(0, cx - ZOOM // 2), max(0, gray.shape[1] - ZOOM))
            crop = cv2.cvtColor(cv2.resize(gray[y0 : y0 + ZOOM, x0 : x0 + ZOOM], None, fx=2, fy=2,
                                           interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
            for r, c in points:
                if y0 <= r < y0 + ZOOM and x0 <= c < x0 + ZOOM:
                    cv2.circle(crop, (int(c - x0) * 2 + 1, int(r - y0) * 2 + 1), 6, (0, 0, 255), 1)
            cv2.imwrite(str(out_dir / f"{row['image_id']}_zoom.png"), crop)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="F1 pore-feasibility gate")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "f1")
    args = parser.parse_args(argv)

    protocol = load_toml("protocol.toml")
    settings = pore_sift.resolve(load_toml("algorithms.toml")["pore_sift"], REPO_ROOT)
    pore_sift.check_upstream(settings)
    dataset_root = Path(protocol["dataset"]["root"])
    images = read_images()
    subjects, fingers, sample, pairs = f1_sample(
        images, protocol["randomness"]["master_seed"], protocol["split"]["dev_fraction"]
    )
    save_sample("f1", sample, pairs)

    run_dir = new_run_dir(args.out)
    features_dir = run_dir / "features"
    features_dir.mkdir()
    jobs = [{"image_id": r["image_id"], "path": image_path(r, dataset_root)} for r in sample]
    extraction, extract_wall = pore_sift.run_parallel("extract", jobs, args.workers, settings, features_dir, run_dir / "work")
    comparisons, compare_wall = pore_sift.run_parallel("compare", pairs, args.workers, settings, features_dir, run_dir / "work")

    scores = {c["pair_id"]: c["score"] for c in comparisons}
    by_kind = defaultdict(list)
    for p in pairs:
        by_kind[(p["scenario"], p["kind"])].append(scores[p["pair_id"]])
    scenarios = {}
    for scenario in ("UxV", "RxV"):
        value = auc(by_kind[(scenario, "genuine")], by_kind[(scenario, "impostor")])
        scenarios[scenario] = {
            "auc": value,
            "auc_pass": value >= AUC_PASS,
            "genuine": summary(by_kind[(scenario, "genuine")]),
            "impostor": summary(by_kind[(scenario, "impostor")]),
        }
    pores = {t["image_id"]: t["pores"] for t in extraction}
    sets = {r["image_id"]: r["set"] for r in sample}
    report = {
        "gate": "F1",
        "algorithm": "Pore SIFT",
        "subjects": subjects,
        "finger_positions": fingers,
        "images": len(sample),
        "pairs": {f"{s}:{k}": len(v) for (s, k), v in sorted(by_kind.items())},
        "scenarios": scenarios,
        "auc_criterion_met": all(s["auc_pass"] for s in scenarios.values()),
        "visual_review": "pending (researcher)",
        "pores_per_image": {code: summary([n for i, n in pores.items() if sets[i] == code]) for code in ("U", "V", "R")},
        "images_without_pores": sorted(i for i, n in pores.items() if n == 0),
        "execution": {"workers": args.workers, "extract_wall_s": extract_wall, "compare_wall_s": compare_wall,
                      "host": host(), "worker_runtime": pore_sift.runtime_versions(settings),
                      "upstream": {k: settings[k] for k in ("survey_revision", "dahia_revision", "model_sha256")}},
    }
    write_json(run_dir / "report.json", report)
    draw_overlays(sample, dataset_root, features_dir, run_dir / "overlays")

    for scenario, s in scenarios.items():
        print(f"{scenario}: AUC {s['auc']:.4f} ({'pass' if s['auc_pass'] else 'FAIL'})  "
              f"genuine median {s['genuine']['median']}  impostor median {s['impostor']['median']}")
    print("pores per image:", {k: round(v.get("median", 0)) for k, v in report["pores_per_image"].items()})
    print(f"overlays: {run_dir / 'overlays'}")


if __name__ == "__main__":
    main()
