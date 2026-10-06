# Pore SIFT — SIFT descriptors at detected pores (eligibility note)

Status: **approved 2026-10-06** (D2): score mode `basic`, the F1 protocol and its
pass criterion, and the separate environment below. The composition of
components from two publications was approved as a D2a exception on the same day.

## Identity

| Component | Source | Revision / file SHA-256 | Licence |
|---|---|---|---|
| Pore detector: `Net17NoMax`, 40 features, out-of-the-box checkpoint `models/40` | Ibragimov & Pamplona Segundo, *Fingerprint Pore Detection: A Survey* (arXiv 2211.14716), github.com/azimIbragimov/Fingerprint-Pore-Detection-A-Survey | `47b128e5`; `models/40` 3787e82d…ffc99, `entireImage.py` b8fa062d…e03ca, `util/utils.py` 9af0761c…a864f | MIT |
| SIFT descriptors at pores, correspondence search, matching scores | Dahia & Pamplona Segundo, *Automatic Dataset Annotation to Learn CNN Pore Description for Fingerprint Recognition* (arXiv 1809.10229). The original repository was deleted; mirror github.com/xiaochengcike/high-res-fingerprint-recognition | `b3c46518`; `utils.py` 57ac08ee…c681, `matching.py` 46ea10d8…dc05 | CC BY-NC-SA 4.0 |

All five hashes equal those pinned in fingerprint-benchmark V2. Mauricio Pamplona
Segundo co-authored both publications and the spatial score's source paper
(Pamplona Segundo & Lemes 2015). The authors declined to share their own trained
detector (2026-09-30), which is why the survey detector stands in.

## D1 — resolution

- **Screening:** both components were built and evaluated on high-resolution
  fingerprints: PolyU-HRF at about 1200 ppi, and L3-SF, which mimics it.
- **Qualification (verified in the code):**
  - The detector reads the image with `cv2.imread(..., IMREAD_GRAYSCALE)`, applies
    only `ToTensor()` (scaling to [0, 1]) and runs the fully convolutional network
    on the whole image (`out_of_the_box_detect.py`).
  - Dahia's `load_image` divides by 255, and `sift_descriptors` applies a median
    blur and CLAHE.
  - No resize occurs anywhere on this path, so SD302 is processed at its native
    1000 ppi.
- **Known mismatch:** the components were developed at about 1200 ppi, so pores
  in SD302 are about 17 % smaller in pixels. D2 forbids rescaling, so this is
  tested by F1 rather than corrected.

## D2 / D2b — exactly what the wrapper does

1. **Weights check.** Verify the SHA-256 of `models/40`, and confirm it loads with
   `torch.load(..., weights_only=True)` as a plain state dict, before upstream's
   own `loadModel` (which uses `torch.load`, i.e. pickle) touches it.
2. **Detection, exactly as in `scripts/detect.sh` and `out_of_the_box_detect.py`.**
   - `loadModel(models/40, NUMBERLAYERS=8, NUMBERFEATURES=40, MAXPOOLING=False,
     WINDOWSIZE=17, residual=False, gabriel=False, su=False)`, then `eval()` on
     CPU.
   - Prediction on the whole image.
   - `entireImage.apply_nms(pred, 0.65, 17, 0.2, ..., 17)`, whose coordinate file
     holds (row, column) in the original image frame (its `+8` restores the
     border consumed by the eight valid 3×3 convolutions). The wrapper only reads
     that file back.
3. **Hand-off between the two publications (the D2a glue).** The detected
   (row, column) points are passed to Dahia's `sift_descriptors`, which expects
   (row, column) and swaps to OpenCV's (x, y) itself. They are passed as float32,
   because OpenCV 3.4.18 rejects integer arrays in `KeyPoint.convert` (the pinned
   3.4.0 accepted them); the whole-number values are unchanged. No border filtering
   is applied: Dahia's SIFT route filters borders only when a patch size is given,
   and it is not given for SIFT.
4. **Descriptors.** `utils.load_image(path)` followed by
   `utils.sift_descriptors(img, pts)` with defaults: `scale=4`, `normalize=True`
   (median blur 3, CLAHE clip 3), computed by `cv2.xfeatures2d.SIFT_create()` at
   the given keypoints.
5. **Matching (needs a decision, see below).** `matching.<mode>(descs_probe,
   descs_reference, pts_probe, pts_reference, thr=0.7)`. The ratio threshold 0.7
   is the value in every documented command and in the code's docstring ("SIFT's
   original criterion").
6. **TensorFlow stub.** Dahia's `utils.py` imports TensorFlow at module level, but
   none of the functions used touch it, and TensorFlow 1.10 cannot be installed
   for a current Python. The wrapper registers an empty `tensorflow` module before
   importing `utils.py`. No upstream line is changed.

### Score mode — `basic` (approved 2026-10-06)

- **`basic` (code default, used).** The number of bidirectional correspondences
  that pass the ratio check. Every documented command, including the paper's
  SIFT experiments, uses it, and `recognize.py` thresholds this count.
- **`spatial` (option, not used).** The Pamplona Segundo & Lemes (2015) score,
  which sums 1/(1+|d1−d2|) over pairs of correspondences. fingerprint-benchmark
  V2 used it; D2b keeps the default instead.

## D3 — training data

The training data of the survey's out-of-the-box checkpoint is not documented.
Pore annotations exist publicly for PolyU-HRF and L3-SF, so SD302 exposure is
unlikely but unproven; results are flagged "detector training data unknown".
SIFT itself is not trained.

## Environment

A separate conda environment:

| Package | Version | Note |
|---|---|---|
| Python | 3.10 (conda-forge) | The newest Python that torch 1.13 supports |
| torch / torchvision | 1.13.0 / 0.14.0 CPU | The survey's own pins |
| numpy | 1.26.4 | |
| opencv-contrib-python | 3.4.18.65 | Closest available to Dahia's pin 3.4.0.12; provides `xfeatures2d.SIFT` and `KeyPoint.convert(size=)` |
| psutil, tqdm | — | Imported by the survey's utilities |

The deviations from the original pins (Python version, OpenCV 3.4.18 instead of
3.4.0.12) are recorded with every run. torch 1.13 has no support for the RTX 5080,
so Pore SIFT runs on CPU only, parallelised across processes.

## F1 — pore-feasibility gate (first run of Pore SIFT)

- **Sample.** Ten subjects from C1's development half who have R images, chosen
  by keyed hash (`f1-subjects`), and two finger positions chosen by keyed hash
  (`f1-fingers`). That gives 20 fingers × U, V, R = 60 images, none of them
  excluded.
- **Pairs.**
  - Genuine: 20 UxV and 20 RxV.
  - Impostor: each U and R image against the V images of the other nine subjects
    at the same position, i.e. 2 positions × 2 probe sets × 10 × 9 = 360.
- **Recorded.**
  - Pores per image.
  - Pore overlays for all 60 images, local only and never committed, for your
    visual review.
  - Genuine and impostor score distributions per scenario.
  - AUC, the probability that a random genuine pair outscores a random impostor
    pair, with ties counted half.
- **Pass criterion (approved 2026-10-06):**
  - AUC ≥ 0.95 in both UxV and RxV, and
  - your visual review confirms that detections sit on ridges at pore-like
    spots.
  - Otherwise F2 applies: stop and decide with the PI.
- **Scope.** This is a feasibility signal on 40 genuine pairs, not an
  evaluation; no parameter is tuned on it.

## Runtime (B4, E5)

- **Earlier measurements.** fingerprint-benchmark V2 ran the same components on
  SD300B at 1000 ppi: about 2.5 s detection per image (CPU, 4 threads), about
  0.08 s description per image and about 22 ms per comparison in `spatial` mode.
- **Compact run (4,915 images, 22,913 comparisons):** about 3.4 h of detection
  in one process, and well under 1 h with parallel processes; comparisons take
  minutes.
- **Timing pilot.** The 60 F1 images plus 40 more development images (100
  extractions) and 2,000 keyed-hash comparisons. It measures single-thread
  latency and parallel throughput, projects the compact and full runs, and runs
  100 pairs twice to check determinism. The projection is approved before any
  full run.
