"""scikit-image Harris (docs/algorithms/skimage-harris.md).

Every step and value is copied from scikit-image's official gallery example "Robust matching
using RANSAC" (scikit-image v0.26.0, doc/examples/transform/plot_matching.py): Harris corners,
sub-pixel corner positions, correspondences by the smallest Gaussian-weighted sum of squared
differences (SSD) of 11 x 11 windows against every corner of the other image, and RANSAC with
an affine model. The comparison loop is kept as written in the example.

Adaptations needed to run it on SD302 fingerprints (eligibility note): windows are grayscale
instead of the example's three colour channels; a probe corner is skipped when corner_subpix
gives it no position (NaN) or when its rounded window would leave the image, because the
example's slicing fails there; RANSAC is seeded for every comparison (R5). The score is the
number of RANSAC inliers, 0 when fewer than three correspondences remain or RANSAC finds no model.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import corner_harris, corner_peaks, corner_subpix
from skimage.measure import ransac
from skimage.transform import AffineTransform
from skimage.util import img_as_float

THRESHOLD_REL = 0.001  # example: corner_peaks(corner_harris(...), threshold_rel=0.001, min_distance=5)
MIN_DISTANCE = 5
SUBPIX_WINDOW = 9  # example: corner_subpix(..., window_size=9)
WINDOW_EXT = 5  # example: match_corner(coord, window_ext=5)
WEIGHT_SIGMA = 3  # example: gaussian_weights(window_ext, 3)
MIN_SAMPLES = 3  # example: ransac(..., min_samples=3, residual_threshold=2, max_trials=100)
RESIDUAL_THRESHOLD = 2
MAX_TRIALS = 100


def gaussian_weights(window_ext: int, sigma: float = 1) -> np.ndarray:
    """As in the example."""
    y, x = np.mgrid[-window_ext : window_ext + 1, -window_ext : window_ext + 1]
    g = np.zeros(y.shape, dtype=np.double)
    g[:] = np.exp(-0.5 * (x**2 / sigma**2 + y**2 / sigma**2))
    g /= 2 * np.pi * sigma * sigma
    return g


def windows_at(image: np.ndarray, centres: np.ndarray) -> np.ndarray:
    e = WINDOW_EXT
    return np.array([image[r - e : r + e + 1, c - e : c + e + 1] for r, c in centres], dtype=np.double).reshape(-1, 2 * e + 1, 2 * e + 1)


def extract_image(image: np.ndarray) -> dict[str, np.ndarray]:
    """Harris corners of one grayscale image, in both roles of the example: as the probe
    ('orig': sub-pixel points and the windows around their rounded positions) and as the
    reference ('warped': integer corners, their windows and their sub-pixel positions)."""
    image = img_as_float(image)
    corners = corner_peaks(corner_harris(image), threshold_rel=THRESHOLD_REL, min_distance=MIN_DISTANCE)
    subpix = corner_subpix(image, corners, window_size=SUBPIX_WINDOW)
    located = subpix[~np.isnan(subpix).any(axis=1)]
    rounded = np.round(located).astype(np.intp)
    e, (height, width) = WINDOW_EXT, image.shape
    inside = (rounded[:, 0] >= e) & (rounded[:, 1] >= e) & (rounded[:, 0] < height - e) & (rounded[:, 1] < width - e)
    return {
        "points": located[inside],
        "windows": windows_at(image, rounded[inside]),
        "corners": corners,
        "corner_windows": windows_at(image, corners),
        "corner_subpix": subpix,
    }


def extract(path: Path) -> dict[str, np.ndarray]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return extract_image(image)


def rng_seed(master_seed: str) -> int:
    """Seed for RANSAC's sample draws, derived from the master seed (R5)."""
    from ..data.sampling import keyed_hash

    return int(keyed_hash(master_seed, "skimage-harris-ransac", "")[:8], 16)


def compare(probe: dict, reference: dict, seed: int | None = None) -> dict:
    """Probe = the example's original image, reference V = its warped image (B1)."""
    if len(probe["points"]) == 0 or len(reference["corners"]) == 0:
        return {"status": "failure", "reason": "insufficient_features", "score": None, "correspondences": None}
    weights = gaussian_weights(WINDOW_EXT, WEIGHT_SIGMA)
    src, dst = [], []
    for coord, window_orig in zip(probe["points"], probe["windows"]):
        # compute sum of squared differences to all corners in the reference image (example's loop)
        SSDs = []
        for window_warped in reference["corner_windows"]:
            SSD = np.sum(weights * (window_orig - window_warped) ** 2)
            SSDs.append(SSD)
        min_idx = np.argmin(SSDs)
        src.append(coord)
        dst.append(reference["corner_subpix"][min_idx])
    src, dst = np.array(src), np.array(dst)
    inliers = None
    if len(src) >= MIN_SAMPLES:
        _, inliers = ransac((src, dst), AffineTransform, min_samples=MIN_SAMPLES,
                            residual_threshold=RESIDUAL_THRESHOLD, max_trials=MAX_TRIALS, rng=seed)
    score = int(inliers.sum()) if inliers is not None else 0
    return {"status": "ok", "reason": "", "score": score, "correspondences": len(src)}


# ---- process-pool helpers (one thread per process, E5) ----


_SEED: int | None = None


def init_worker(seed: int | None = None) -> None:
    global _SEED
    cv2.setNumThreads(1)
    _SEED = seed


def extract_job(job: tuple[str, str, str]) -> dict:
    image_id, path, out_dir = job
    start = time.perf_counter()
    features = extract(Path(path))
    elapsed = time.perf_counter() - start
    out = Path(out_dir) / f"{image_id}.npz"
    np.savez(out, **features)
    return {"image_id": image_id, "extract_s": elapsed, "corners": len(features["corners"]),
            "probe_points": len(features["points"]), "bytes": out.stat().st_size}


def _load(path: str) -> dict:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def compare_job(job: tuple[dict, str]) -> dict:
    pair, features_dir = job
    start = time.perf_counter()
    probe = _load(str(Path(features_dir) / f"{pair['probe_image_id']}.npz"))
    reference = _load(str(Path(features_dir) / f"{pair['reference_image_id']}.npz"))
    loaded = time.perf_counter()
    result = compare(probe, reference, _SEED)
    done = time.perf_counter()
    return {"pair_id": pair["pair_id"], **result, "load_s": loaded - start, "compare_s": done - loaded}
