"""OpenCV SIFT check and side-by-side images of the suspected Q and J labels (A5, decision record 2026-10-19).

Runs after fpl3.method.audit and the plain run's OpenCV SIFT features:
    python -m fpl3.experiments.audit_check   # -> runs/plain/audit/check.csv, check.json, side-by-side/*.png

A suspected image is compared with OpenCV SIFT against its own V and against the candidate that
the audit found above it. OpenCV SIFT agrees when that candidate scores above the full run's RxV
development impostor threshold at FAR 0.01 % and above the own V. The side-by-side images stay
local (R8); nothing here changes a label or excludes an image (R4, A2).
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from ..algorithms import opencv_sift
from ..data.build import write_csv
from ..data.samples import development_subjects
from ..eval.metrics import tar_at_far
from .common import REPO_ROOT, image_path, load_toml, read_images, write_json

AUDIT_DIR = REPO_ROOT / "runs" / "plain" / "audit"
FULL_FEATURES = REPO_ROOT / "runs" / "full" / "features" / "opencv-sift"
PLAIN_FEATURES = REPO_ROOT / "runs" / "plain" / "features" / "opencv-sift"
CHECK_FAR = 0.0001
FIELDS = ("image_id", "set", "subject", "frgp", "best_other_image_id", "best_other_same_subject", "learned_own",
          "learned_other", "opencv_own", "opencv_other", "opencv_agrees")
PANEL_HEIGHT = 520


def development_threshold(development: set[str]) -> float:
    """OpenCV SIFT's RxV impostor threshold at FAR 0.01 % on development pairs of the full run (C1)."""
    impostors = []
    with open(REPO_ROOT / "runs" / "full" / "scores" / "opencv-sift.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r["scenario"], r["kind"]) == ("RxV", "impostor") and {r["probe_image_id"].split("_")[1],
                                                                       r["reference_image_id"].split("_")[1]} <= development:
                impostors.append(int(r["score"]) if r["score"] != "" else None)
    return tar_at_far([0], impostors, CHECK_FAR)["accept_if_score_above"]


def load(path: Path):
    with np.load(path) as f:
        return f["points"], f["descriptors"]


def panel(path: str, caption: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    scale = PANEL_HEIGHT / image.shape[0]
    image = cv2.resize(image, (round(image.shape[1] * scale), PANEL_HEIGHT), interpolation=cv2.INTER_AREA)
    header = np.full((44, image.shape[1]), 255, np.uint8)
    cv2.putText(header, caption, (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1, cv2.LINE_AA)
    return np.vstack([header, image])


def side_by_side(panels: list[np.ndarray]) -> np.ndarray:
    gap = np.full((panels[0].shape[0], 24), 255, np.uint8)
    out = [panels[0]]
    for p in panels[1:]:
        out += [gap, p]
    return np.hstack(out)


def main() -> None:
    protocol = load_toml("protocol.toml")
    root = Path(protocol["dataset"]["root"])
    images = {r["image_id"]: r for r in read_images()}
    development, _ = development_subjects(list(images.values()), protocol["randomness"]["master_seed"], protocol["split"]["dev_fraction"])
    seed = opencv_sift.rng_seed(protocol["randomness"]["master_seed"])
    opencv_sift.init_worker(seed)
    t = development_threshold(development)
    with open(AUDIT_DIR / "audit.csv", newline="", encoding="utf-8") as f:
        suspects = [r for r in csv.DictReader(f) if r["category"] == "suspected"]
    out_dir = AUDIT_DIR / "side-by-side"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in suspects:
        own_id = f"V_{r['subject']}_{int(r['frgp']):02d}"
        own_path = FULL_FEATURES / f"{own_id}.npz"  # excluded V images have no features (A3, A4)
        probe = load(PLAIN_FEATURES / f"{r['image_id']}.npz")
        own = opencv_sift.compare(probe, load(own_path), seed)["score"] if own_path.exists() else None
        other = opencv_sift.compare(probe, load(FULL_FEATURES / f"{r['best_other_image_id']}.npz"), seed)["score"]
        agrees = other is not None and other > t and (own is None or other > own)
        rows.append({"image_id": r["image_id"], "set": r["set"], "subject": r["subject"], "frgp": r["frgp"],
                     "best_other_image_id": r["best_other_image_id"], "best_other_same_subject": r["best_other_same_subject"],
                     "learned_own": r["own_score"], "learned_other": r["best_other_score"],
                     "opencv_own": "" if own is None else own, "opencv_other": "" if other is None else other, "opencv_agrees": int(agrees)})
        panels = [panel(image_path(images[r["image_id"]], root), f"{r['image_id']} (labelled)")]
        if own_id in images and not int(images[own_id]["excluded"]):
            panels.append(panel(image_path(images[own_id], root), f"own {own_id}: learned {r['own_score']}, SIFT {own}"))
        panels.append(panel(image_path(images[r["best_other_image_id"]], root),
                            f"best {r['best_other_image_id']}: learned {r['best_other_score']}, SIFT {other}"))
        cv2.imwrite(str(out_dir / f"{r['image_id']}.png"), side_by_side(panels))
    write_csv(AUDIT_DIR / "check.csv", FIELDS, rows)
    kinds = Counter(f"{'same subject' if r['best_other_same_subject'] == '1' else 'other subject'}, "
                    f"{'agrees' if r['opencv_agrees'] else 'does not agree'}" for r in rows)
    write_json(AUDIT_DIR / "check.json", {"opencv_threshold": t, "check_far": CHECK_FAR, "suspected": len(rows),
                                          "opencv_agrees": sum(r["opencv_agrees"] for r in rows), "by_kind": dict(sorted(kinds.items()))})
    print(f"{len(rows)} suspected, OpenCV SIFT agrees on {sum(r['opencv_agrees'] for r in rows)}; threshold {t}")


if __name__ == "__main__":
    main()
