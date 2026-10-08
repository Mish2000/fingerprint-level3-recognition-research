# scikit-image Harris on the native image (eligibility note)

Status: **F1 check, 2026-10-08**, decided by the assistant on the researcher's
delegation. Whether it joins the comparison, which needs the exception below, is
decided after the F1 result.

## Identity

| Item | Value |
|---|---|
| Algorithm | Harris corners (Harris & Stephens 1988), sub-pixel corner positions, correspondences by the smallest Gaussian-weighted SSD of 11 × 11 windows, and RANSAC with an affine model |
| Implementation | scikit-image 0.26.0 from conda-forge (`corner_harris`, `corner_peaks`, `corner_subpix`, `ransac`, `AffineTransform`) |
| Pipeline source | scikit-image's official gallery example *Robust matching using RANSAC* (`doc/examples/transform/plot_matching.py`, tag v0.26.0) — every step and parameter below is copied from it |
| Licence | BSD-3-Clause |
| Training data | none (hand-crafted), so no SD302 exposure (D3 not applicable) |

## Why this pipeline (D2a)

Harris alone is a corner detector: it finds points in one image and gives no
comparison or score. OpenCV documents it only that way (its Harris tutorials
detect corners in a single image; every OpenCV tutorial that compares two images
uses SIFT, SURF, ORB or AKAZE). scikit-image documents complete Harris
pipelines. Its *Robust matching using RANSAC* example has the same structure as
the OpenCV SIFT tutorial (points, matching, geometric verification), with every
component documented together by the library's authors, so D2a is met without
an exception. Its *BRIEF* example has no geometric verification and, by its own
note, no rotation invariance, so it is not used.

## D1 — resolution

- **Screening:** the operators have no resolution parameter and take any image size.
- **Qualification:** the image is analysed at its native 1000 ppi; nothing is resized.

## D2 / D2b — exactly what the wrapper does

1. Read the PNG with `cv.imread(path, cv.IMREAD_GRAYSCALE)` and `img_as_float`
   (8-bit to [0, 1]), as the example does for its image; no crop, resize or enhancement.
2. `corner_peaks(corner_harris(image), threshold_rel=0.001, min_distance=5)`.
3. `corner_subpix(image, coords, window_size=9)`.
4. For a pair, probe = the example's original image, reference V = its warped
   image (B1). For every probe corner, the 11 × 11 window around its rounded
   sub-pixel position is compared with the window around every reference corner
   by `np.sum(weights * (window_orig - window_warped) ** 2)`, with the example's
   `gaussian_weights(5, 3)`; the reference corner with the smallest SSD gives the
   correspondence (its sub-pixel position). The loop is kept as written.
5. `ransac((src, dst), AffineTransform, min_samples=3, residual_threshold=2, max_trials=100)`.

Corners and windows are computed once per image and cached; the windows hold the
same pixels the example slices from the image.

**Adaptations needed on fingerprints:**
- Windows are grayscale; the example compares windows of a three-channel colour image.
- On SD302 `corner_subpix` gives no position (NaN) to about 45 % of the corners
  (1,856 of 4,288 on a rolled pilot image), and a few rounded positions lie too
  close to the border for a full window. The example's slicing fails on such a
  probe corner, so it is skipped. As in the example, reference corners stay
  candidates; a correspondence that lands on one without a sub-pixel position
  has a NaN destination, which RANSAC never counts as an inlier.

**Score (as for OpenCV SIFT):** the example estimates a transform and defines no
similarity score. Score = number of RANSAC inliers, 0 when fewer than three
correspondences remain or RANSAC finds no model.

**Failures (E1):** a probe without a usable corner, or a reference without
corners, is recorded as a failure.

**Reproducibility (R5):** RANSAC draws random samples; the wrapper passes a seed
derived from the master seed to every call, so scores repeat across runs,
processes and order.

## Execution and runtime (B4, E5)

- On three pilot images: about 4,300 corners on a rolled print and 1,600 on a
  plain one. The example's loop compares every probe corner with every reference
  corner, one window at a time: about 2.5 minutes per rolled pair on one core.
- The full design as written would take months with 12 workers, far over the
  24 h limit. A full run therefore needs an approved exception: a faster
  implementation of the same computation, checked pair by pair against the
  example's loop.
- The F1 sample (400 pairs) as written takes about an hour with 12 workers.
