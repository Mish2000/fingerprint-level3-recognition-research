"""OpenCV SIFT (docs/algorithms/opencv-sift.md).

Every step and value is copied from OpenCV's official tutorial "Feature Matching
+ Homography to find Objects" (opencv/opencv, branch 4.x,
doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.markdown).
The only addition is the approved score: the number of RANSAC inliers, 0 when the
tutorial's condition len(good) > MIN_MATCH_COUNT fails or no homography is found.
"""

from __future__ import annotations

import functools
import time
from pathlib import Path

import cv2
import numpy as np

FLANN_INDEX_KDTREE = 1  # tutorial
FLANN_TREES = 5  # tutorial
FLANN_CHECKS = 50  # tutorial
RATIO = 0.7  # tutorial: m.distance < 0.7 * n.distance
MIN_MATCH_COUNT = 10  # tutorial: if len(good) > MIN_MATCH_COUNT
RANSAC_REPROJ_THRESHOLD = 5.0  # tutorial: cv.findHomography(src, dst, cv.RANSAC, 5.0)


def extract(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Keypoint coordinates (N x 2, float32) and SIFT descriptors (N x 128, float32) with OpenCV defaults."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    keypoints, descriptors = cv2.SIFT_create().detectAndCompute(image, None)
    points = np.float32([kp.pt for kp in keypoints]).reshape(-1, 2)
    if descriptors is None:
        descriptors = np.zeros((0, 128), np.float32)
    return points, descriptors


def rng_seed(master_seed: str) -> int:
    """Seed for OpenCV's random generator, derived from the master seed (R5)."""
    from ..data.sampling import keyed_hash

    return int(keyed_hash(master_seed, "opencv-sift-rng", "")[:8], 16)


def compare(
    probe: tuple[np.ndarray, np.ndarray], reference: tuple[np.ndarray, np.ndarray], seed: int | None = None
) -> dict:
    """Probe = query, reference = train (B1). Returns status, score and the number of ratio-test matches.

    FLANN's randomised KD-trees draw from OpenCV's random generator, so unseeded scores vary from run
    to run (115 of 200 pilot pairs did). Seeding it before every comparison makes each score
    reproducible regardless of process or order; no documented parameter changes.
    """
    if seed is not None:
        cv2.setRNGSeed(seed)
    points1, descriptors1 = probe
    points2, descriptors2 = reference
    if len(descriptors1) == 0 or len(descriptors2) < 2:
        return {"status": "failure", "reason": "insufficient_features", "score": None, "good": None}
    matcher = cv2.FlannBasedMatcher(dict(algorithm=FLANN_INDEX_KDTREE, trees=FLANN_TREES), dict(checks=FLANN_CHECKS))
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    if any(len(pair) < 2 for pair in matches):
        return {"status": "failure", "reason": "fewer_than_two_neighbours", "score": None, "good": None}
    good = [m for m, n in matches if m.distance < RATIO * n.distance]
    inliers = 0
    if len(good) > MIN_MATCH_COUNT:
        source = points1[[m.queryIdx for m in good]].reshape(-1, 1, 2)
        destination = points2[[m.trainIdx for m in good]].reshape(-1, 1, 2)
        _, mask = cv2.findHomography(source, destination, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)
        inliers = int(mask.sum()) if mask is not None else 0
    return {"status": "ok", "reason": "", "score": inliers, "good": len(good)}


# ---- process-pool helpers (one OpenCV thread per process, E5) ----


_SEED: int | None = None


def init_worker(seed: int | None = None) -> None:
    global _SEED
    cv2.setNumThreads(1)
    _SEED = seed


def extract_job(job: tuple[str, str, str]) -> dict:
    image_id, path, out_dir = job
    start = time.perf_counter()
    points, descriptors = extract(Path(path))
    elapsed = time.perf_counter() - start
    out = Path(out_dir) / f"{image_id}.npz"
    np.savez(out, points=points, descriptors=descriptors)
    return {"image_id": image_id, "extract_s": elapsed, "keypoints": len(points), "bytes": out.stat().st_size}


@functools.lru_cache(maxsize=4)
def _load(path: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        return data["points"], data["descriptors"]


def compare_job(job: tuple[dict, str]) -> dict:
    pair, features_dir = job
    start = time.perf_counter()
    probe = _load(str(Path(features_dir) / f"{pair['probe_image_id']}.npz"))
    reference = _load(str(Path(features_dir) / f"{pair['reference_image_id']}.npz"))
    loaded = time.perf_counter()
    result = compare(probe, reference, _SEED)
    done = time.perf_counter()
    return {"pair_id": pair["pair_id"], **result, "load_s": loaded - start, "compare_s": done - loaded}
