# Decision record

Decisions approved by Michael Sirkovich in chat on 2026-10-05/06, before any
code was written. Changes are appended below with date and reason; existing
entries are not rewritten.

## R — standing rules

- **R1** SD302 stays read-only at its local path; nothing is copied into it or modified.
- **R2** Resolution is read from each file's PNG header (pHYs), not from folder
  names. SHA-256 is verified against NIST's checksum CSVs on every manifest
  build; a mismatch excludes the image and is reported.
- **R3** The manifest indexes every native-1000-ppi single-finger exemplar
  (U, V, R, Q, J) with flags; the choices below only shape the pair lists.
- **R4** Every exclusion carries its reason; nothing is removed silently.
  Exclusion is per image: a probe whose reference was excluded can still be an
  impostor probe.
- **R5** Randomness enters only the subject split (C1), impostor sampling (B3),
  the F1 sample, bootstrap resampling (E4) and training of learned components.
  It is implemented as keyed SHA-256 ranking from one master seed plus a
  purpose string, so a new purpose never changes an existing sample; generated
  samples are saved. A non-deterministic algorithm is checked by a double run
  on a sample.
- **R6** Each comparison type is reported separately, never pooled.
- **R7** Anything learned or tuned (new-method parameters, fusion weights,
  operational thresholds) is fit on development subjects only. Reading TAR at a
  fixed FAR from an algorithm's own scores on evaluation pairs is not tuning.
  Two sets only, development and test; a validation subset, if ever needed,
  comes from development subjects.
- **R8** No images, crops, templates or descriptors enter git.

## A — data cleaning

- **A1** Latents (sd302e/h/i) are out of scope.
- **A2** Images named in the errata for the included sets are excluded; NIST
  labels are never changed. Applies to R 00002420 fingers 01 and 06 (thumbs
  swapped, errata 11 July 2023).
- **A3** V 00002361 fingers 02 and 04 are excluded (errata 18 June 2024, marked
  unconfirmed: "right ring is actually right index"; content of V_02 unknown).
- **A4** V 00002472 finger 05 is excluded: not in the errata, but absent from
  NIST's examiner-annotated set (sd302g) and showing a vertical seam with an
  abrupt texture change.

## B — comparisons and pairs

- **B1** Reference set V (NIST's baseline exemplar set: one physical device,
  examiner-annotated in sd302g). Probes are compared against it in one fixed
  direction for every algorithm.
- **B2** Scenarios: UxV (rolled vs rolled) and RxV (plain from slap vs rolled).
  Q and J stay in the manifest only.
- **B3** Impostor = same finger position, different subject. Long term: the
  full set. Now: 1,000 per finger position per scenario (10,000), keyed-hash
  sample, identical for every algorithm.
- **B4** A full run must finish within 24 h wall clock on the research PC. A
  timing pilot (about 100 extractions and 2,000 comparisons) precedes every full
  run, and its projection is approved first.

## C — split

- **C1** Approved in principle: 50/50 by subject, stratified by whether the
  subject has R images, keyed hash. Activation is postponed until something has
  to be tuned; the split is defined in code and fixed by the master seed.

## G — infrastructure

- **G1** Root folder `C:\fingerprint-level3-recognition-research`.
- **G2** Git from the first commit; a GitHub repository (public is acceptable).
- **G3** Conda environment from conda-forge through the existing Miniconda,
  Python 3.12.

## D — algorithm eligibility

- **D1** Only systems designed for ≥1000 ppi input.
  - Screening: documentation shows the algorithm can take ≥1000-ppi input.
  - Qualification: the input is not normalised below 1000 ppi before feature
    extraction (open source: verified in the code; closed SDK: a written vendor
    statement). Multi-scale analysis that includes the native resolution, such
    as the SIFT scale space, is acceptable.
  - Systems documented to normalise internally to 500 ppi (for example
    SourceAFIS, Innovatrics IDKit, id3) do not qualify.
- **D2** Integration only, no change to algorithm code. Each algorithm gets an
  eligibility note stating exactly what its wrapper does, approved per algorithm.
  - **D2a** Separately published components may be joined only if their authors
    document them together; otherwise only as an explicitly approved exception.
  - **D2b** Score-affecting parameters keep their default values.
- **D3** Algorithms trained or validated on SD302 are included and flagged.
- **D4** Closed commercial SDKs are allowed as black boxes when the licence
  permits research use and publication of results.
- **D5** No quota: whatever qualifies is included.

## E — evaluation

- **E1** A failure to produce a score counts at the lowest score and failure
  rates are reported separately (ISO/IEC 19795-1 practice); a common-subset
  table is secondary.
- **E2** TAR at FAR 1 % and 0.1 % (0.01 % only if the runtime allows), EER, DET
  curves, failure rates, and runtime with extraction separate from comparison.
- **E3** Algorithms are compared by the ROC on the evaluation pairs (TAR at a
  fixed FAR from each algorithm's own scores); a development-set threshold is
  used only for operational claims about the new method.
- **E4** 95 % confidence intervals by subject-level bootstrap; algorithms are
  compared pairwise on identical pairs.
- **E5** All work runs on the current PC. Reported timings are single-thread on
  a sample; full runs may run in parallel.

## F — Level 3 and the new method

- **F1** A pore-feasibility gate on a small development sample of SD302 comes
  before developing the method.
- **F2** If the gate fails, stop and decide with the PI.
- **F3** Pores as an additional channel alongside minutiae; more features are
  welcome when they serve the goal.
- **F4** Primary success criterion: fusing the Level-3 channel with the best
  existing algorithm lowers FRR at FAR 0.1 % on the test set (paired
  subject-level bootstrap). Secondary: the new method alone versus the best
  existing algorithm.
- **F5** Liveness detection is out of scope for now.

## 2026-10-06 — confirmations

- **D1** Confirmed as two stages: screening by the documentation, then
  qualification by the operational test above.
- **D2a** Exception approved for Pore SIFT: the pore detector from the
  Fingerprint Pore Detection survey repository joined with Dahia & Pamplona
  Segundo's SIFT pore descriptors and the Pamplona Segundo & Lemes spatial
  matching score (score mode corrected below).
- **F1** The feasibility sample is drawn from the subjects that C1 assigns to
  development; C1 is not otherwise activated.
- **G2** The GitHub repository is public.

## 2026-10-06 — algorithm eligibility notes approved

- **Naming** Algorithms are referred to by name everywhere ("OpenCV SIFT",
  "Pore SIFT"), never by placeholder labels.
- **OpenCV SIFT** approved as described in `docs/algorithms/opencv-sift.md`.
  Score = number of RANSAC inliers; 0 when there are 10 or fewer ratio-test
  matches or no homography is found.
- **Pore SIFT** approved as described in `docs/algorithms/pore-sift.md`.
  Correction to the D2a entry: the score is the code-default `basic` mode (number
  of bidirectional correspondences passing the 0.7 ratio check), not the spatial
  score, per D2b.
- **F1** Pass criterion: AUC ≥ 0.95 in both UxV and RxV on the F1 sample, and the
  researcher's visual review confirms that detections sit on ridges at pore-like
  spots. Otherwise F2 applies.
- **G3** Two environments: the project environment (plus OpenCV from
  conda-forge) and a separate environment for Pore SIFT's pinned dependencies
  (Python 3.10, torch 1.13.0, opencv-contrib-python 3.4.18.65). One command
  drives both.

## 2026-10-07 — F1 no longer disqualifies

- **F1/F2** F1 is informational only. An algorithm is not dropped for missing an
  accuracy target, because so few candidates qualify (consistent with D5); F2 no
  longer applies. Pore SIFT continues after its F1 run of 2026-10-06 (rolled vs
  rolled TAR 90 %, FRR 10 %, FAR 0/180; plain vs rolled TAR 65 %, FRR 35 %,
  FAR 0/180, at the lowest threshold that accepts no impostor).
- **G2** Nothing is pushed to GitHub until the researcher asks.

## 2026-10-07 — full run

- **B3** The full impostor set runs now for OpenCV SIFT and Pore SIFT: every
  probe against every reference of the same finger position, 582,724 pairs per
  algorithm (UxV 1,997 genuine + 397,403 impostors; RxV 916 genuine + 182,408
  impostors). The compact pairs are a subset; they run first and are reported
  as well.
- **B4** Projection approved: about 15 h for both algorithms one after the other
  with 12 workers (OpenCV SIFT 8.5–9 h, Pore SIFT 6–7 h), from timings per
  comparison type. The earlier 14.25 h came from a pilot with a larger share of
  plain vs rolled pairs than the full design.
- **E2** TAR at FAR 0.01 % is reported as well, since the runtime allows it.

## 2026-10-08 — threshold rule and the research's own methods

- **E2** TAR at FAR x: a pair is accepted when its score is above the lowest
  threshold at which at most a share x of the impostor pairs is accepted; a
  failed comparison is never accepted (E1).
- **D2** D2 binds the existing algorithms used for comparison. The research's
  own methods are written and changed freely; the first is a machine-learning
  model fine-tuned on development subjects only (R7).
- **C1** Activated: 100 development and 100 test subjects, 46 with R images in
  each, from the split fixed by the master seed. A development pair has both
  subjects in development and a test pair both in test; pairs that mix the
  halves are not used. Test pairs: UxV 1,000 genuine and 99,000 impostors; RxV
  460 and 45,540. Anything learned or tuned uses development pairs (R7); the
  research's own methods are compared with the existing algorithms on test pairs,
  using the existing algorithms' full-run scores.
- **D1/D2a — scikit-image Harris** (decided by the assistant on the researcher's
  delegation): OpenCV documents Harris only as a corner detector; scikit-image's
  example *Robust matching using RANSAC* is a complete Harris pipeline
  (`docs/algorithms/skimage-harris.md`). As written it would need months for the
  full design (B4), so it first runs unchanged on the F1 sample, as OpenCV SIFT
  did. A faster implementation of the same computation, and with it a full run,
  is decided after that result.
- **G3** scikit-image (0.26.0) and SciPy are added to the project environment
  from conda-forge.
- **D1/D2a — scikit-image Harris, F1 result:** TAR 0/20 in UxV and RxV at the
  strictest threshold; genuine scores at chance level (median 3 inliers, RANSAC's
  minimum sample). It finds a shifted copy of an image (204 inliers) but no
  correspondences between two impressions of a finger, as its documentation
  warns. It stays at the F1 result: no faster implementation and no full run.
- **F3/F4 — our method, part 1:** `docs/methods/learned-pore-descriptor.md`
  approved: a reference fusion of the existing algorithms first, then Dahia &
  Pamplona Segundo's learned pore descriptor trained on development subjects and
  matched as in Pore SIFT. Harris is not part of it.
- **B2/R7** Q and J of development subjects serve as training data for our
  method; evaluation pairs stay UxV and RxV.
- **G3** A third environment, `fingerprint-level3-gpu`, for GPU work: PyTorch
  2.13 with CUDA 13.0 (the driver supports up to 13.1) from conda-forge. The GPU
  is to be used as much as possible (researcher, 2026-10-08).
