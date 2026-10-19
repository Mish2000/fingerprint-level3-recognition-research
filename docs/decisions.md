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

## 2026-10-10 — native resolution for the research's own methods

- **D1** The qualification now binds the research's own methods as well
  (researcher, 2026-10-10): every method, learned or not, compares prints from
  the native ≥1000 ppi image; a method that reduces the resolution before
  feature extraction, to 500 ppi or otherwise, does not qualify. The proposed
  part 2 of our method (a fine-tuned general vision model on the whole print at
  500 ppi) is withdrawn.

## 2026-10-16 — research overview and publication

- **G2** The researcher approved publishing the concise research overview and
  README updates across the five repositories, including the previously local
  October 10–11 documentation. This repository is the current research home;
  the previous Level-3 repository retains its evidence and an overview link
  before archival. Existing CV links remain usable.
- **D2/R7 — reporting correction:** the executed pore descriptor is trained
  from random initialization in PyTorch, rather than fine-tuned from pretrained
  descriptor weights. Its optimizer is SGD without weight decay. Automatic
  annotation uses SIFT/RANSAC alignment, block matching and a local
  moving-least-squares warp. The method note now describes that implementation;
  the earlier design wording remains part of the historical record. This
  correction authorizes no new experiment and changes no scores or protocol.
- **E2–E4/R8** The overview presents the first training and second-seed check
  separately, with their held-out pair counts and FAR targets. Publication
  contains documentation and aggregate results only; research artifacts stay
  local.

## 2026-10-18 — Stage B: the identification system

Approved by the researcher after the supervisor meeting of 2026-10-18. A new
group H covers the Stage B system.

- **H1** Stage B is a system build only. It embeds the fingerprint methods of
  this repository and replaces the 2026-10-10 task list (3–4 non-AI and 3–4 AI
  methods, accuracy proof on the full pair sets); the Stage B work order
  recorded on 2026-10-10 is withdrawn. Fingerprint is the primary modality,
  face and voice the further ones. Face and voice may use ready-made SDKs; D1
  keeps binding the fingerprint methods. The system offers enrolment, 1:1
  verification and 1:N identification.
- **H2** Every method keeps its native template and comparison: the fingerprint
  methods compare sets of local descriptors, face and voice one embedding each.
  The supervisor's "decide by stored vectors" is met by storing every template
  at enrolment and processing only the new sample at verification or
  identification. Each method's template and comparison are described briefly
  to the user.
- **H3** National ID: asked once at registration, validated by its check digit
  and never stored. A keyed hash (HMAC) serves duplicate checks and lookups; the
  account links to a random identifier stored encrypted (AES-GCM) under a key
  that our own function derives from a master secret and a per-record random
  value. Keys stay outside the databases and the code; only standard
  cryptographic primitives are used. The supervisor's confirmation of this
  reading of "a secret function we write ourselves" is pending.
- **H4** Stack: a React + TypeScript client packaged with Electron; a FastAPI
  server (Python 3.12); one engine process per method in its own environment;
  PostgreSQL with pgvector in two databases (accounts and identity link;
  biometric data keyed only by the random identifier); raw samples and the
  attempt history stored encrypted; OpenCV SFace with YuNet for face;
  SpeechBrain ECAPA-TDNN for voice; the Python `cryptography` package for every
  cryptographic primitive.
- **H5** Scanner: Biometrika HiScan PRO at 1000 ppi (it outputs 500 ppi unless
  1000 is selected). The integration is designed anew; the Stage A TWAIN helper
  is not reused.
- **G4** The system lives in a new public repository directly under `C:\`,
  named by the researcher. Its commits use the repository-local identity
  without co-author trailers. This repository stays the research home and the
  source of the methods.

## 2026-10-18 — Stage B follow-up

- **G4** Repository name: `multimodal-biometric-system`, at
  `C:\multimodal-biometric-system`.
- **H3** The supervisor leaves the design of the secret function to us; the
  design recorded above stands and nothing about it is pending.
- **H5** The build does not wait for the vendor: scanner integration goes
  through TWAIN with 1000 ppi set programmatically and no driver dialog. The
  FxISO SDK (now offered by DigitalPlatforms, which acquired Biometrika) is
  requested in parallel and used only if it arrives.
- Stage B proceeds one step at a time; each step is reported to the researcher
  before the next starts.

## 2026-10-19 — plain vs plain scenarios and a label audit

Approved by the researcher on 2026-10-19: the timing pilot, the missing
features and the label audit run now; the comparisons follow a report.

- **B1/B2** New scenarios RxJ and QxJ, plain vs plain at native 1000 ppi:
  probes R (slaps, Crossmatch L SCAN 1000PX) and Q (fingers segmented from
  upper palms on the same scanner), reference J (fingers segmented from upper
  palms, Morpho TouchPrint 5300). J has no thumbs, so positions 01 and 06 do
  not occur. Each scenario is reported separately (R6).
- **B3/C1** Full design inside each half: every probe against every J
  reference of the same finger position, only for pairs whose two subjects are
  both development or both test. Development: 626 genuine and 61,974 impostor
  pairs; test: 649 and 64,251; 127,500 comparisons per algorithm.
- **R7** Operational thresholds and fusion weights for these scenarios come
  from development pairs: all of them for OpenCV SIFT and Pore SIFT; only the
  validation subjects' pairs (152 genuine, 2,888 impostors) for the learned
  pore descriptor and for every fusion that includes it, since it was trained
  on the other development subjects. TAR at FAR is also read from the test
  pairs' own scores (E2).
- **A5** Label audit before any Q or J image is used, because NIST notes that
  not every palm capture shows the labelled position (sd302c README). The
  learned pore descriptor (second training) scores every usable Q and J image
  against every usable V image of the same subject and against the V images of
  the same finger position of every other subject. T is the threshold that at
  most 0.01 % of these cross-subject comparisons score above. An image is
  *confirmed* when its own V (same subject, labelled position) scores above T
  and above every other candidate; *suspected* when another candidate scores
  above T and above its own V; *undetermined* otherwise, and undetermined
  images stay in. Suspected images are checked with OpenCV SIFT against the
  full run's RxV development impostor threshold at FAR 0.01 % and shown side by
  side. One is excluded only with the researcher's approval and a recorded
  reason (R4); NIST labels are never changed (A2).
- **B4** A timing pilot of the new design (100 development images of R, Q and
  J; 2,000 probe × reference pairs; keyed hash) precedes the comparisons.
  Features are extracted only where none exist: OpenCV SIFT for Q and J, Pore
  SIFT for Q and J of test subjects. R features and the development Q and J
  pore detections are reused from the full run and from `runs/method/pores`
  (same worker, settings and model).

## 2026-10-19 — plain vs plain comparisons

- **B4** After the report of the pilots, the missing features and the label
  audit, the researcher approved the comparisons together with their
  thresholds, fusion weights and test results. Projection: OpenCV SIFT about
  40 min, Pore SIFT about 9 min, the learned pore descriptor 2–3 min on the
  GPU; they took 42 min, 8 min and 48 s.
- **A5** The audit found no suspected label, so no Q or J image is excluded;
  the 330 undetermined images and the 4 whose own V is excluded stay in.
- **H6/R7** The plain vs plain scenarios score the learned pore descriptor's
  second training only, the one the validation subjects chose; its first
  training is not scored on them.

## 2026-10-18 — defaults for the system specification (recorded 2026-10-19)

- **H6** Approved with the researcher's general approval of the plan on
  2026-10-18 and open to change in the specification:
  - enrolment stores templates for every algorithm of a modality, and the
    algorithm is chosen at verification or identification, so switching
    algorithms or choosing a fusion needs no new enrolment;
  - fingerprint enrolment: the user chooses the finger, which is captured three
    times as a quality check;
  - the system's learned pore descriptor is its second training;
  - each algorithm decides at FAR 0.1 % with a threshold from SD302
    development pairs until it is checked on the HiScan PRO;
  - fusion uses the weights recommended by the research, or weights the user
    chooses that sum to 100 (the supervisor's suggestion of June 2026);
  - every attempt is stored (time, modality, algorithm, score, threshold,
    decision, duration and the encrypted sample), and an administrator screen
    shows the history without national IDs;
  - SD302 images never enter the system's databases (R1, R8); demonstrations
    use real captures;
  - no liveness detection, as decided for the research (F5);
  - the user interface is in Hebrew;
  - 1:1 verification and 1:N identification are both built as an impressive,
    interactive experience, and the texts are written for a simple but curious
    user (researcher, 2026-10-18).

## 2026-10-19 — the system repository

- **G4** The researcher approved the scanner prototype and the NTwain download
  on 2026-10-19, and `C:\multimodal-biometric-system` was created (local only).
  Its `docs/decisions.md` carries H1–H6 and G4 over; system decisions from H7
  on are recorded there and no longer here.
