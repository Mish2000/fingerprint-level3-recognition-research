"""Pore identities for training the learned descriptor (docs/methods/learned-pore-descriptor.md, step 2).

As in Dahia & Pamplona Segundo's annotation (polyu/aligned_images.py), every pore detected in a
finger's reference impression (V, else U) gets the windows at its corresponding positions in the
finger's other impressions (U, R, Q, J). Their prints were small and nearly rigid, so one
similarity transform found those positions; SD302's full prints stretch, so here the positions
come from registration in three steps:

1. a similarity transform from OpenCV SIFT matches (FLANN, ratio 0.7, RANSAC);
2. dense control points: 64 x 64 windows on a grid over the reference, block-matched (normalised
   cross-correlation) around their predicted positions, then a smooth moving-least-squares warp;
3. every pore's own 32 x 32 window block-matched within a few pixels of its warped position,
   kept only when the correlation is high.

    python -m fpl3.method.annotate [--workers 12]
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool

import cv2
import numpy as np

from ..data.sampling import keyed_rank
from ..experiments.common import REPO_ROOT
from .data import DESCRIPTOR_DATA, dataset_root, fingers, protocol_and_images, split_development
from .pores import pore_file

FULL_RUN_SIFT = REPO_ROOT / "runs" / "full" / "features" / "opencv-sift"

PATCH = 32  # polyu.preprocess --patch_size 32
GRID = 40  # spacing of the dense control points, px
CONTROL_WINDOW, CONTROL_SEARCH, CONTROL_NCC = 64, 12, 0.5
PORE_SEARCH, PORE_NCC = 5, 0.6
SIGMA_COARSE, SIGMA_FINE = 120.0, 60.0  # moving-least-squares neighbourhoods, px
PRIOR = 1.0  # pull of each local fit towards the similarity transform
MIN_INLIERS = 12
MAX_IDENTITIES = 600  # per finger, chosen by keyed hash (R5)


def sift_features(image_id: str, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV SIFT keypoints (row, col) and descriptors: the full run's for U, V, R; computed for Q, J."""
    path = FULL_RUN_SIFT / f"{image_id}.npz"
    if path.exists():
        with np.load(path) as f:
            return f["points"][:, ::-1].astype(float), f["descriptors"]
    keypoints, descriptors = cv2.SIFT_create().detectAndCompute(image, None)
    return np.float64([k.pt for k in keypoints]).reshape(-1, 2)[:, ::-1], descriptors


def similarity(points_r, descs_r, points_o, descs_o):
    """Similarity transform (2 x 3, in x/y) from reference to other, with its RANSAC inliers (row, col)."""
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    cv2.setRNGSeed(0)
    good = [m for m, n in (p for p in matcher.knnMatch(descs_r, descs_o, k=2) if len(p) == 2) if m.distance < 0.7 * n.distance]
    if len(good) < MIN_INLIERS:
        return None, None, None
    src = np.float32([points_r[m.queryIdx][::-1] for m in good])
    dst = np.float32([points_o[m.trainIdx][::-1] for m in good])
    m, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=15.0, maxIters=5000,
                                             confidence=0.999, refineIters=10)
    if m is None or inliers.sum() < MIN_INLIERS:
        return None, None, None
    keep = inliers.ravel().astype(bool)
    return m, src[keep][:, ::-1].astype(float), dst[keep][:, ::-1].astype(float)


def mls_affine(src: np.ndarray, dst: np.ndarray, queries: np.ndarray, prior: np.ndarray, sigma: float) -> np.ndarray:
    """Map queries (row, col) by a locally weighted affine fit of src -> dst, pulled towards `prior` (2 x 3, x/y)."""
    xs, ys = src[:, 1], src[:, 0]
    w = np.exp(-((queries[:, None, :] - src[None, :, :]) ** 2).sum(-1) / (2 * sigma**2))
    m = w @ np.stack([xs * xs, xs * ys, xs, ys * ys, ys, np.ones_like(xs)], 1)
    xtx = np.stack([m[:, [0, 1, 2]], m[:, [1, 3, 4]], m[:, [2, 4, 5]]], 1)
    targets = dst[:, ::-1]
    xty = np.stack([w @ (xs[:, None] * targets), w @ (ys[:, None] * targets), w @ targets], 1)
    affine = np.linalg.solve(xtx + PRIOR * np.eye(3), xty + PRIOR * prior.T[None])
    q = np.stack([queries[:, 1], queries[:, 0], np.ones(len(queries))], 1)
    return np.einsum("qk,qkd->qd", q, affine)[:, ::-1]


def block_match(reference: np.ndarray, other: np.ndarray, centres: np.ndarray, predicted: np.ndarray,
                window: int, search: int, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """For each reference centre, the best normalised-correlation position near its predicted position in
    `other`; returns the indices kept and their positions (row, col)."""
    half = window // 2
    kept, positions = [], []
    for k, ((r, c), (pr, pc)) in enumerate(zip(centres.astype(int), np.round(predicted).astype(int))):
        if not (half <= r < reference.shape[0] - half and half <= c < reference.shape[1] - half):
            continue
        r0, c0 = pr - half - search, pc - half - search
        if r0 < 0 or c0 < 0 or r0 + window + 2 * search > other.shape[0] or c0 + window + 2 * search > other.shape[1]:
            continue
        template = reference[r - half : r + half, c - half : c + half]
        if template.std() < 1.0:
            continue
        scores = cv2.matchTemplate(other[r0 : r0 + window + 2 * search, c0 : c0 + window + 2 * search], template, cv2.TM_CCOEFF_NORMED)
        _, best, _, (bc, br) = cv2.minMaxLoc(scores)
        if best >= threshold and 0 < br < 2 * search and 0 < bc < 2 * search:
            kept.append(k)
            positions.append((r0 + br + half, c0 + bc + half))
    return np.array(kept, int), np.array(positions, float).reshape(-1, 2)


def dense_map(shape: tuple[int, int], src: np.ndarray, dst: np.ndarray, prior: np.ndarray, sigma: float, step: int = 16):
    """Position in the other image (row map, column map) of every reference pixel: the warp is evaluated
    every `step` pixels and interpolated bilinearly."""
    rows, cols = np.arange(0, shape[0] + step, step), np.arange(0, shape[1] + step, step)
    grid = np.stack(np.meshgrid(rows, cols, indexing="ij"), -1).reshape(-1, 2).astype(float)
    coarse = mls_affine(src, dst, grid, prior, sigma).reshape(len(rows), len(cols), 2).astype(np.float32)
    rr, cc = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
    return (cv2.remap(coarse[..., 0], cc / step, rr / step, cv2.INTER_LINEAR),
            cv2.remap(coarse[..., 1], cc / step, rr / step, cv2.INTER_LINEAR))


def warp_into(other: np.ndarray, maps) -> np.ndarray:
    return cv2.remap(other, maps[1], maps[0], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def register(reference: np.ndarray, other: np.ndarray, sift_r, sift_o, pores_r: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Reference pores found in `other`: their indices and positions there, with statistics. Block matching
    runs between the reference and the other image warped into the reference frame, so rotation and
    stretch are already undone; positions are mapped back into the other image."""
    m, ctrl_r, ctrl_o = similarity(*sift_r, *sift_o)
    if m is None:
        return np.zeros(0, int), np.zeros((0, 2)), {"aligned": False}
    maps = dense_map(reference.shape, ctrl_r, ctrl_o, m, SIGMA_COARSE)
    cells = {}
    for i, (r, c) in enumerate(pores_r.astype(int)):
        cells.setdefault((r // GRID, c // GRID), i)
    grid = pores_r[sorted(cells.values())].astype(float)
    kept, found = block_match(reference, warp_into(other, maps), grid, grid, CONTROL_WINDOW, CONTROL_SEARCH, CONTROL_NCC)
    if len(kept) < MIN_INLIERS:
        return np.zeros(0, int), np.zeros((0, 2)), {"aligned": False, "sift_inliers": len(ctrl_r)}
    found_o = np.stack([maps[0][tuple(found.astype(int).T)], maps[1][tuple(found.astype(int).T)]], 1)
    maps = dense_map(reference.shape, np.vstack([ctrl_r, grid[kept]]), np.vstack([ctrl_o, found_o]), m, SIGMA_FINE)
    index, positions = block_match(reference, warp_into(other, maps), pores_r, pores_r.astype(float), PATCH, PORE_SEARCH, PORE_NCC)
    positions_o = np.round(np.stack([maps[0][tuple(positions.astype(int).T)], maps[1][tuple(positions.astype(int).T)]], 1))
    half = PATCH // 2
    fits = ((positions_o[:, 0] >= half) & (positions_o[:, 0] < other.shape[0] - half)
            & (positions_o[:, 1] >= half) & (positions_o[:, 1] < other.shape[1] - half))
    index, positions_o = index[fits], positions_o[fits]
    return index, positions_o, {"aligned": True, "sift_inliers": len(ctrl_r), "controls": len(kept), "pores": len(index),
                                "pore_share": round(len(index) / max(1, len(pores_r)), 3)}


def finger_task(task: tuple) -> dict:
    (subject, frgp), views, root, seed = task
    reference = "V" if "V" in views else "U"
    images = {code: cv2.imread(str(root / row["relpath"]), cv2.IMREAD_GRAYSCALE) for code, row in views.items()}
    with np.load(pore_file(views[reference]["image_id"])) as f:
        pores_r = f["points"].astype(int)
    sift = {code: sift_features(views[code]["image_id"], images[code]) for code in views}
    found: dict[str, dict[int, tuple[int, int]]] = {}
    stats = {}
    for code in sorted(views):
        if code != reference:
            index, positions, stats[code] = register(images[reference], images[code], sift[reference], sift[code], pores_r)
            found[code] = {int(i): (int(p[0]), int(p[1])) for i, p in zip(index, positions)}
    half = PATCH // 2
    usable = [i for i in range(len(pores_r)) if any(i in f for f in found.values())
              and half <= pores_r[i][0] < images[reference].shape[0] - half and half <= pores_r[i][1] < images[reference].shape[1] - half]
    chosen = keyed_rank(usable, seed, f"descriptor-identities:{subject}:{frgp:02d}", key=str)[:MAX_IDENTITIES]
    patches, labels, views_used = [], [], []
    for k, i in enumerate(chosen):
        r, c = pores_r[i]
        patches.append(images[reference][r - half : r + half, c - half : c + half])
        labels.append(k)
        views_used.append(reference)
        for code, f in sorted(found.items()):
            if i in f:
                r, c = f[i]
                patches.append(images[code][r - half : r + half, c - half : c + half])
                labels.append(k)
                views_used.append(code)
    return {"finger": f"{subject}_{frgp:02d}", "reference": reference, "stats": stats, "identities": len(chosen),
            "patches": np.stack(patches) if patches else np.zeros((0, PATCH, PATCH), np.uint8),
            "labels": np.array(labels, np.int64), "views": views_used}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args(argv)
    protocol, images = protocol_and_images()
    train, validation = split_development(protocol, images)
    groups = fingers(images, train)
    seed, root = protocol["randomness"]["master_seed"], dataset_root(protocol)
    tasks = [(key, views, root, seed) for key, views in sorted(groups.items()) if len(views) >= 2]
    DESCRIPTOR_DATA.mkdir(parents=True, exist_ok=True)
    start, results = time.time(), []
    with Pool(args.workers) as pool:
        for result in pool.imap(finger_task, tasks, chunksize=2):
            results.append(result)
            if len(results) % 100 == 0:
                print(f"{len(results)} of {len(tasks)} fingers, {time.time() - start:.0f}s", flush=True)
    patches, labels, offset = [], [], 0
    for r in results:
        patches.append(r["patches"])
        labels.append(r["labels"] + offset)
        offset += r["identities"]
    np.save(DESCRIPTOR_DATA / "train_patches.npy", np.concatenate(patches))
    np.save(DESCRIPTOR_DATA / "train_labels.npy", np.concatenate(labels))
    registrations = [s for r in results for s in r["stats"].values()]
    by_view: dict[str, list] = {}
    for r in results:
        for code, s in r["stats"].items():
            by_view.setdefault(code, []).append(s)
    summary = {
        "train_subjects": train, "validation_subjects": validation, "fingers": len(results), "identities": offset,
        "patches": int(sum(len(p) for p in patches)), "registrations": len(registrations),
        "per_view": {code: {"registered": sum(s["aligned"] for s in ss), "of": len(ss),
                            "median_pore_share": float(np.median([s["pore_share"] for s in ss if s["aligned"]] or [0]))}
                     for code, ss in sorted(by_view.items())},
        "per_finger": [{k: r[k] for k in ("finger", "reference", "identities", "stats")} for r in results],
    }
    (DESCRIPTOR_DATA / "annotation.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print({k: v for k, v in summary.items() if k not in ("per_finger", "train_subjects", "validation_subjects")})


if __name__ == "__main__":
    main()
