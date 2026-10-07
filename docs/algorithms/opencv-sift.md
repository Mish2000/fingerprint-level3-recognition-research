# OpenCV SIFT on the native image (eligibility note)

Status: **approved 2026-10-06** (D2), including the score definition below.

## Identity

| Item | Value |
|---|---|
| Algorithm | SIFT keypoints and descriptors (Lowe 2004) with ratio-test matching and RANSAC geometric verification |
| Implementation | OpenCV 4.x, main module (`cv.SIFT_create`, `cv.FlannBasedMatcher`, `cv.findHomography`) |
| Pipeline source | OpenCV's official tutorial *Feature Matching + Homography to find Objects* (`doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.markdown`, opencv/opencv branch 4.x) — every step and parameter below is copied from it |
| Licence | Apache 2.0 (OpenCV ≥ 4.5); the SIFT patent expired in 2020 |
| Training data | none (hand-crafted), so no SD302 exposure (D3 not applicable) |

## D1 — resolution

- **Screening:** SIFT has no resolution parameter; it is scale-invariant by
  design, and OpenCV documents that "the number of octaves is computed
  automatically from the image resolution".
- **Qualification:** with the default settings OpenCV builds the first octave
  from the image **upscaled ×2** (`firstOctave = -1`, `createInitialImage(...,
  doubleImageSize=true, ...)` in `sift.dispatch.cpp`). Detail at the native
  1000 ppi is therefore analysed; the coarser octaves are part of SIFT's scale
  space, which D1 accepts. Nothing normalises the input below 1000 ppi.

## D2 / D2b — exactly what the wrapper does

1. Read the PNG with `cv.imread(path, cv.IMREAD_GRAYSCALE)`. The SD302 files are
   already 8-bit grayscale, so the pixels are passed unchanged; no crop, resize,
   enhancement or segmentation.
2. `sift = cv.SIFT_create()` with every default: `nfeatures=0`,
   `nOctaveLayers=3`, `contrastThreshold=0.04`, `edgeThreshold=10`,
   `sigma=1.6`, `enable_precise_upscale=False`.
3. `kp, des = sift.detectAndCompute(img, None)` once per image.
4. For a pair, probe = query (`des1`), reference V = train (`des2`) (B1):
   `cv.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50)).knnMatch(des1, des2, k=2)`.
5. Ratio test exactly as in the tutorial: keep `m` if `m.distance < 0.7 * n.distance`.
6. As in the tutorial, geometric verification only if `len(good) > 10`:
   `M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)`.

**Score (the one interpretive step, approved 2026-10-06):** the tutorial locates
an object and does not define a similarity score. Score = number of RANSAC
inliers (`mask.sum()`), and 0 when the tutorial's condition `len(good) > 10`
fails or no homography is found. Higher means more similar.

**Failures (E1):** an image with no keypoints, or a reference with fewer than two
descriptors (where the tutorial's `for m, n in matches` cannot run), is recorded
as a failure, not as a score; metrics then treat it as the lowest score.

**Reproducibility (R5, added 2026-10-07):** FLANN's randomised KD-trees draw from
OpenCV's random generator. Unseeded, the timing pilot gave 115 of 200 pairs a
different score on a second run (by up to 9). The wrapper therefore calls
`cv.setRNGSeed(seed)` before every comparison, with the seed derived from the
master seed. Scores are then identical across runs, processes and order. No
documented parameter changes.

## Execution and runtime (B4, E5)

- Environment: the project environment plus `opencv` from conda-forge, version
  pinned at installation and recorded with every run.
- Runtime is unknown until measured. Upscaling a 1600×1500 rolled image to
  3200×3000 can yield tens of thousands of keypoints, so both extraction and
  FLANN matching may cost seconds.
- Storage: float32 descriptors for every image could reach tens of GB, so the
  choice between caching descriptors and re-extracting per pair is made from the
  pilot's numbers.
- **Timing pilot:** about 100 images and 2,000 comparisons from development
  subjects (C1 half). It measures single-thread latency for extraction and for
  comparison, and the parallel throughput on this PC. It projects the compact
  run (22,913 comparisons) and the full run (582,724), and it checks determinism
  by running 100 pairs twice (R5: FLANN uses randomised trees). The projection is
  approved before any full run.
