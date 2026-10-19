# AGENTS.md

Current-state context for coding agents. Keep it short and current: update it
when a decision or the project state changes. History belongs in
`docs/decisions.md` and git.

## Goal

Evaluate existing fingerprint verification algorithms that are designed for
≥1000 ppi input on native 1000 ppi live-scan images from NIST SD302, then use
the same data and evaluation layers to develop a new method that also uses
Level-3 features (sweat pores). Researcher: Michael Sirkovich, under academic
supervision. Every reply to the researcher is in Hebrew (he asked repeatedly);
code, docs and commits stay in English.

Stage B (supervisor meeting 2026-10-18, H1–H6 and G4; it replaces the
2026-10-10 task list): only a system build, in its own public repository
`C:\multimodal-biometric-system`, embedding this repository's fingerprint
methods. Login (email or username + password) → menu with fingerprint
(primary), face and voice → enrolment, 1:1 verification and 1:N
identification, both built as an impressive, interactive experience, with the
algorithm chosen from short accuracy, speed and how-it-works descriptions
written for a simple but curious user. Face and voice use ready-made SDKs. The
national ID is never stored. Raw samples and every attempt are stored
encrypted; decisions compare stored templates, each in its method's native form.

## Working rules

- Ask instead of assuming whenever a definition or constraint is ambiguous.
  Do exactly what was approved; no unrequested reports or scope growth.
- Every protocol choice has an ID in `docs/decisions.md` (R, A–H). Cite it in
  code and commits; a change needs the researcher's approval and an appended
  entry in that file.
- Never copy, modify or commit SD302 images, crops, templates or descriptors
  (R1, R8). The dataset path lives only in `configs/protocol.toml`.
- Call every algorithm by its name in chat, docs, file names and code; never
  use placeholder labels such as S1/S2.
- The agent sets the order of work (researcher, 2026-10-08); time limits are
  guidance, the aim is the strongest results. Stage B goes one step at a time,
  each step reported before the next starts (researcher, 2026-10-18).

## Data (verified 2026-10-05/06; see manifests/build_report.json)

- SD302 versions: sd302b 20180814, sd302c 20190207; 200 subjects.
- Native 1000 ppi single-finger exemplars, all live-scan:
  U, V = FBI operator-assisted rolls (2,000 each); R = segmented 4-4-2 slaps
  (920, 92 subjects); Q, J = fingers segmented from upper palms (864 / 1,275,
  no thumbs). R and Q are one physical scanner with complementary subjects.
- Every other SD302 exemplar set is below 1000 ppi (challengers A–H: 500, 600,
  648×550); latents are out of scope.
- Excluded images: `configs/exclusions.csv` (V 00002361 02+04, R 00002420
  01+06, V 00002472 05). NIST labels are never changed.
- Scenarios (probe × reference V): UxV 1,997 genuine; RxV 916 genuine; each
  10,000 impostors (1,000 per finger position, same finger, different subject).

## Algorithm rules

- D1: screening = documentation shows the algorithm can take ≥1000 ppi input;
  qualification = the input is not normalised below 1000 ppi before feature
  extraction (open source: check the code; closed SDK: vendor statement).
  Multi-scale analysis that includes the native resolution (SIFT) is fine.
  Documented 500-ppi designs (SourceAFIS, NBIS/MINDTCT, IDKit, id3, ...) fail.
  Since 2026-10-10 the qualification binds our own methods too, learned ones
  included: native ≥1000 ppi input, no reduction (500 ppi included).
- D2: wrappers only, no change to algorithm code; D2a: components from
  different publications only if their authors document them together, else
  an explicitly approved exception; D2b: default parameters.
- Each algorithm needs a written eligibility note approved by the researcher.
- B4/E5: a timing pilot precedes every full run; a full run must stay under
  24 h wall clock on this PC (Core Ultra 9 275HX, 24 cores, 31 GB RAM,
  RTX 5080 16 GB). Reported timings are single-thread on a sample.
- Extract each image's template once and compare cached templates whenever the
  algorithm's official interface separates extraction from comparison.

## Evaluation

Failures count at the lowest score (E1). TAR at FAR 1 / 0.1 / 0.01 % from each
algorithm's own scores, EER, DET; 95 % CIs by subject bootstrap; paired
comparisons on identical pairs (E2–E4): `fpl3.experiments.evaluate`. Anything
tuned uses development subjects only (R7); C1 is active (100/100 subjects).

## Layout

```
configs/       protocol.toml (paths, seed, scenarios), exclusions.csv,
               algorithms.toml (environment paths and upstream pins)
docs/          research-projects.md — public research overview; decisions.md — approved
               decisions with IDs; algorithms/ — eligibility notes
environments/  pinned requirements of the separate Pore SIFT environment
manifests/     generated and committed: images.csv, pairs.csv, build_report.json,
               f1_images.csv / f1_pairs.csv (F1 sample)
src/fpl3/      data/ (SD302 readers, manifest, pairs, keyed-hash sampling, split, samples)
               algorithms/ (OpenCV SIFT wrapper; Pore SIFT driver + worker)
               eval/ (AUC, TAR at FAR, EER, DET, subject bootstrap)
               experiments/ (F1 gate, sample runs, timing pilots, full run, evaluate,
               plain vs plain run and evaluation, label-audit check)
               method/ (our method: Q/J pores, pore-identity annotation, learned
               descriptor on the GPU, score fusion, Q/J label audit)
tests/         unit tests; integration tests marked `dataset`
runs/, third_party/   local only, git-ignored (features, overlays, upstream checkouts)
```

## Commands

```bash
# Project environment (conda-forge only; the Anaconda defaults channel is not used)
conda create -n fingerprint-level3-recognition-research --override-channels -c conda-forge python=3.12 pip setuptools pytest numpy "opencv=4.13.0" "scikit-image=0.26.0"
conda activate fingerprint-level3-recognition-research
pip install -e . --no-deps --no-build-isolation
# Pore SIFT environment: see environments/pore-sift-requirements.txt; upstream code
# checked out in third_party/ at the revisions pinned in configs/algorithms.toml

python -m fpl3.data.build        # rebuild manifests (about 12 s, byte-identical)
python -m fpl3.experiments.f1    # F1 gate -> runs/f1 (refuses to overwrite)
python -m fpl3.experiments.full_run            # full run -> runs/full; the same command resumes it
python -m fpl3.experiments.full_run --status   # progress and expected finish (also runs/full/progress.html)
python -m fpl3.experiments.evaluate            # E2-E4 on runs/full: EER, DET, bootstrap CIs, full and test pairs
python -m fpl3.experiments.results_page        # one self-contained results page -> runs/results/results.html
# Our method (docs/methods/learned-pore-descriptor.md); GPU steps run in fingerprint-level3-gpu
# (PyTorch 2.13, CUDA 13.0, conda-forge; `pip install -e . --no-deps` there too)
python -m fpl3.method.fusion                   # reference fusion of the existing algorithms (step 0)
python -m fpl3.method.pores && python -m fpl3.method.annotate    # Q/J pores, pore identities
python -m fpl3.method.descriptor train|score   # GPU environment
# Plain vs plain (RxJ, QxJ; decision record 2026-10-19)
python -m fpl3.experiments.timing_pilot --algorithm opencv-sift|pore-sift --design plain
python -m fpl3.experiments.plain_run --until extract   # missing Q/J features; without --until: every stage -> runs/plain
python -m fpl3.method.audit                            # GPU environment: Q/J label audit against V (A5)
python -m fpl3.experiments.audit_check                 # OpenCV SIFT check and side-by-side images of suspects
python -m fpl3.method.descriptor score --name repeat --pairs plain   # GPU environment, after plain_run
python -m fpl3.experiments.plain_eval                  # fusion, thresholds (R7), test TAR/FAR/FRR -> runs/plain/eval
pytest -m "not dataset"          # unit tests (on Windows, add --basetemp inside a writable folder)
pytest                           # + integration tests against local SD302
```

## Conventions

- Python 3.12, standard library first; configuration in TOML (`tomllib`).
- Randomness only through `fpl3.data.sampling.keyed_rank` with the master seed
  plus a unique purpose string (R5). Generated files have fixed column order,
  LF endings and no timestamps.
- Commits use the repo-local identity (Michael Sirkovich, GitHub noreply).
  Do not add automated co-author attribution to commit messages.

## Status

- Public presentation (done 2026-10-16, G2): this repository is the current
  research home; `docs/research-projects.md` is the concise CV entry point for
  all five projects. The other four READMEs link to it, the old CV URL
  (`fingerprint-l3-benchmark/docs/research-projects.md`) now points here, and
  `fingerprint-l3-benchmark` is archived as the historical SD300B study.
- Done (2026-10-06): data layer — manifest, exclusions, UxV/RxV pairs,
  compact impostors, split definition, 24 passing tests.
- Approved algorithms (2026-10-06), always called by name, never by labels:
  - OpenCV SIFT (`docs/algorithms/opencv-sift.md`): OpenCV's documented
    SIFT + FLANN + ratio 0.7 + RANSAC pipeline; score = RANSAC inliers.
  - Pore SIFT (`docs/algorithms/pore-sift.md`): survey pore detector →
    Dahia & Pamplona Segundo SIFT descriptors at pores → `basic` correspondence
    count; separate environment; its first run is the F1 gate.
- F1 run (2026-10-06), Pore SIFT, 400 pairs: rolled vs rolled TAR 90 %,
  FRR 10 %, FAR 0/180; plain vs rolled TAR 65 %, FRR 35 %, FAR 0/180 (lowest
  threshold accepting no impostor). Rejections are mostly left little fingers.
  Since 2026-10-07, F1 is informational only: Pore SIFT continues.
- OpenCV SIFT on the same 400 F1 pairs (2026-10-07, seeded): rolled vs rolled
  TAR 19/20, FAR 0/180; plain vs rolled TAR 14/20, FAR 0/180.
- Timing pilots (2026-10-07, 12 workers, both deterministic): OpenCV SIFT
  compact 0.42 h, full 8.0 h; Pore SIFT compact 1.83 h, full 6.25 h.
- Full run (2026-10-07 14:17 to 10-08 05:39, 15 h 22 min; B3, B4, E2): both
  algorithms on all 582,724 pairs each, no failures; `fpl3.experiments.full_run`
  wrote runs/full/report.json and scores/<algorithm>.csv. TAR at FAR 1 / 0.1 /
  0.01 %: rolled vs rolled OpenCV SIFT 86.9 / 67.9 / 44.0 %, Pore SIFT 84.4 /
  75.7 / 68.0 %; plain vs rolled OpenCV SIFT 79.0 / 58.4 / 36.7 %, Pore SIFT
  65.0 / 53.3 / 44.1 %.
- Approved 2026-10-08: our first own method is a learned pore descriptor (D2;
  implemented in PyTorch and trained from scratch, clarified 2026-10-16).
  C1 is active (100/100 subjects; test pairs have both subjects in test:
  UxV 1,000 + 99,000, RxV 460 + 45,540). scikit-image Harris
  (`docs/algorithms/skimage-harris.md`) on F1: TAR 0/20 in both scenarios,
  chance-level scores; not included, no full run.
- Bar on test pairs, TAR at FAR 0.1 % (runs/full/eval): rolled vs rolled Pore
  SIFT 76.5 % [72.9, 81.1]; plain vs rolled OpenCV SIFT 64.6 % [53.7, 75.1].
- Our method, part 1 (`docs/methods/learned-pore-descriptor.md`), approved
  2026-10-08. Step 0, reference fusion of OpenCV SIFT and Pore SIFT on test
  pairs, TAR at FAR 0.1 %: rolled vs rolled 87.4 % [83.5, 91.4], plain vs rolled
  76.5 % [68.3, 84.3] (runs/method/fusion/reference). Annotation: 453,149 pore
  identities, 1,064,023 patches from 80 training subjects; 20 development
  subjects are held out for early stopping and fusion weights.
- Learned pore descriptor on test pairs, TAR at FAR 0.1 %: alone 96.9 % (rolled)
  and 85.0 % (plain); fused with the best existing algorithm (F4) 98.0 % and
  88.9 %; with both existing algorithms 98.6 % and 91.5 %. Details and intervals
  in `docs/methods/learned-pore-descriptor.md`. Descriptors and scores in
  runs/method (local only).
- Robustness check (second training seed, `--name repeat`): alone 98.1 % rolled,
  88.9 % plain; fused with both existing algorithms 99.2 % and 93.9 %. The
  proposed part 2 (whole print at 500 ppi) was withdrawn on 2026-10-10 (D1).
- Stage B decisions (2026-10-18/19): H1–H6 and G4 in `docs/decisions.md`; the
  2026-10-10 work order (new methods 2–6) is withdrawn. The system repository
  `C:\multimodal-biometric-system` exists since 2026-10-19 (local only); it
  carries H1–H6 and G4 over, and system decisions from H7 on are recorded
  there. Scanner: Biometrika HiScan PRO, 1000 ppi only when selected (default
  500); integration through TWAIN with the resolution set in code (H5).
  The vendor's FxISO SDK exists locally only as x64 DLLs without headers or
  documentation; the request for FxISO SDK Light to DigitalPlatforms
  (Biometrika's owner), drafted on 2026-10-18, has been sent by the researcher
  (reported on 2026-10-19); no reply yet.
- Plain vs plain, done 2026-10-19 (B1–B4, R7, A5; runs/plain, runs/plain/eval):
  SD302 has it only across devices (R and Q on the Crossmatch L SCAN against J
  on the Morpho TouchPrint, no thumbs). Label audit (T = 17): Q 817 confirmed,
  46 undetermined, 1 without its own V; J 988, 284, 3; none suspected, so none
  excluded. Test pairs, TAR at FAR 0.1 %: RxJ (226 genuine, 29,628 impostors)
  OpenCV SIFT 160, Pore SIFT 176, learned descriptor 215, all three 215; QxJ
  (423, 34,623) 288, 332, 395, 394. Development thresholds at a 0.1 % target
  kept the test FAR below 0.1 % for every method; at 0.01 % the validation
  subjects' 1,584 impostors are too few for fusions (QxJ: 18 false accepts).
  Times: OpenCV SIFT 42 min, Pore SIFT 8 min, learned descriptor 48 s.
- Scanner prototype, approved 2026-10-19 with the NTwain download (H7 in the
  system repository): the skeleton and the capture tool (C#, .NET Framework
  4.8, x86, NTwain 3.7.6) are built and tested without scanning; its checks
  reproduce the 2026-10-18 analysis of the researcher's ten first captures
  exactly. A probe of the plugged-in scanner showed that its TWAIN driver
  ("TWAIN Biometrika Driver" 2.2) offers only 500 ppi and refuses 1000, with no
  saved settings (CAP_CUSTOMDSDATA); its own window offers "Save 1000 DPI...".
  So the approved route and its fallback are blocked; the route to 1000 ppi
  awaits the researcher.
- Then, one step at a time: the researcher captures a second set of his
  ten fingers at 1000 ppi (his first set, one capture per finger, stays
  outside the repository); our methods on those HiScan PRO captures (pores,
  time per attempt, same finger against different fingers at the plain vs
  plain thresholds); face and voice feasibility (OpenCV SFace with YuNet,
  SpeechBrain ECAPA-TDNN; downloads need approval); then the system
  specification for approval. Volunteers only if the supervisor wants accuracy
  on the HiScan PRO itself (about 10 people for FAR 1 %, 20 for 0.1 %).
- Awaiting the researcher: the route to 1000 ppi (the vendor's reply to the
  request already sent; a capture through the driver's own window, tried at
  the scanner; the undocumented OEM library is not recommended); the proposed
  system default (the learned descriptor alone at FAR 0.1 %, fusions offered
  as options).
- Report results to the researcher as plain TAR / FAR / FRR with pair counts;
  avoid AUC and statistical jargon unless asked.
- Do not push to GitHub until the researcher asks; the commits made since
  2026-10-18 are local only.
