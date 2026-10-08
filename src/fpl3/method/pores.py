"""Pore detections for our method (docs/methods/learned-pore-descriptor.md).

U, V and R reuse the full run's Pore SIFT features (runs/full/features/pore-sift). Q and J of
development subjects, training data only (decision record 2026-10-08), are detected here with the
same survey detector through the Pore SIFT driver; the files hold the same arrays (pore points
and their SIFT descriptors).

    python -m fpl3.method.pores [--workers 12]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..algorithms import pore_sift
from ..data.samples import development_subjects
from ..experiments.common import REPO_ROOT, image_path, load_toml, read_images

FULL_RUN_PORES = REPO_ROOT / "runs" / "full" / "features" / "pore-sift"
METHOD_PORES = REPO_ROOT / "runs" / "method" / "pores"


def development_images(sets: str = "UVRQJ") -> list[dict]:
    protocol = load_toml("protocol.toml")
    images = read_images()
    development, _ = development_subjects(images, protocol["randomness"]["master_seed"], protocol["split"]["dev_fraction"])
    return [r for r in images if r["set"] in sets and not int(r["excluded"]) and r["subject"] in development]


def pore_file(image_id: str) -> Path:
    """Where an image's pore points and SIFT descriptors are stored."""
    return (METHOD_PORES if image_id[0] in "QJ" else FULL_RUN_PORES) / f"{image_id}.npz"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args(argv)
    dataset_root = Path(load_toml("protocol.toml")["dataset"]["root"])
    settings = pore_sift.resolve(load_toml("algorithms.toml")["pore_sift"], REPO_ROOT)
    pore_sift.check_upstream(settings)
    METHOD_PORES.mkdir(parents=True, exist_ok=True)
    rows = [r for r in development_images("QJ") if not pore_file(r["image_id"]).exists()]
    jobs = [{"image_id": r["image_id"], "path": image_path(r, dataset_root)} for r in rows]
    chunks = [(f"{i:05d}", jobs[i * 20 : (i + 1) * 20]) for i in range(-(-len(jobs) // 20))]
    done, start = [], time.time()

    def on_done(key, results, started, finished):
        done.extend(results)
        print(f"{len(done)} of {len(jobs)} images, {time.time() - start:.0f}s", flush=True)

    pore_sift.run_chunks("extract", chunks, args.workers, settings, METHOD_PORES, REPO_ROOT / "runs" / "method" / "work", on_done)
    (METHOD_PORES / "extraction.json").write_text(json.dumps(done), encoding="utf-8")


if __name__ == "__main__":
    main()
