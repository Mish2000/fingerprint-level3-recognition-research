# AGENTS.md

Current-state context for coding agents. Keep it short and current: update it
when a decision or the project state changes. History belongs in
`docs/decisions.md` and git.

## Goal

Evaluate existing fingerprint verification algorithms that are designed for
≥1000 ppi input on native 1000 ppi live-scan images from NIST SD302, then use
the same data and evaluation layers to develop a new method that also uses
Level-3 features (sweat pores). Researcher: Michael Sirkovich, under academic
supervision. Discussion with the researcher is in Hebrew.

## Working rules

- Ask instead of assuming whenever a definition or constraint is ambiguous.
  Do exactly what was approved; no unrequested reports or scope growth.
- Every protocol choice has an ID in `docs/decisions.md` (R, A–G). Cite it in
  code and commits; a change needs the researcher's approval and an appended
  entry in that file.
- Never copy, modify or commit SD302 images, crops, templates or descriptors
  (R1, R8). The dataset path lives only in `configs/protocol.toml`.
- Call every algorithm by its name in chat, docs, file names and code; never
  use placeholder labels such as S1/S2.

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
- D2: wrappers only, no change to algorithm code; D2a: components from
  different publications only if their authors document them together, else
  an explicitly approved exception; D2b: default parameters.
- Each algorithm needs a written eligibility note approved by the researcher.
- B4/E5: a timing pilot precedes every full run; a full run must stay under
  24 h wall clock on this PC (Core Ultra 9 275HX, 24 cores, 31 GB RAM,
  RTX 5080 16 GB). Reported timings are single-thread on a sample.
- Extract each image's template once and compare cached templates whenever the
  algorithm's official interface separates extraction from comparison.

## Evaluation (to be built)

Failures count at the lowest score (E1). TAR at FAR 1 % and 0.1 % from each
algorithm's own scores, EER, DET; 95 % CIs by subject bootstrap; paired
comparisons on identical pairs (E2–E4). Anything tuned uses development
subjects only (R7); the subject split is defined in `fpl3.data.split` but not
active (C1).

## Layout

```
configs/       protocol.toml (paths, seed, scenarios), exclusions.csv,
               algorithms.toml (environment paths and upstream pins)
docs/          decisions.md — approved decisions with IDs; algorithms/ — eligibility notes
environments/  pinned requirements of the separate Pore SIFT environment
manifests/     generated and committed: images.csv, pairs.csv, build_report.json,
               f1_images.csv / f1_pairs.csv (F1 sample)
src/fpl3/      data/ (SD302 readers, manifest, pairs, keyed-hash sampling, split, samples)
               algorithms/ (OpenCV SIFT wrapper; Pore SIFT driver + worker)
               eval/ (AUC, TAR at FAR, summaries)
               experiments/ (F1 gate, sample runs, timing pilots, full run)
tests/         unit tests; integration tests marked `dataset`
runs/, third_party/   local only, git-ignored (features, overlays, upstream checkouts)
```

## Commands

```bash
# Project environment (conda-forge only; the Anaconda defaults channel is not used)
conda create -n fingerprint-level3-recognition-research --override-channels -c conda-forge python=3.12 pip setuptools pytest numpy "opencv=4.13.0"
conda activate fingerprint-level3-recognition-research
pip install -e . --no-deps --no-build-isolation
# Pore SIFT environment: see environments/pore-sift-requirements.txt; upstream code
# checked out in third_party/ at the revisions pinned in configs/algorithms.toml

python -m fpl3.data.build        # rebuild manifests (about 12 s, byte-identical)
python -m fpl3.experiments.f1    # F1 gate -> runs/f1 (refuses to overwrite)
python -m fpl3.experiments.full_run            # full run -> runs/full; the same command resumes it
python -m fpl3.experiments.full_run --status   # progress and expected finish (also runs/full/progress.html)
pytest -m "not dataset"          # unit tests
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
- Full run approved (2026-10-07; B3, B4, E2 entries): both algorithms on all
  582,724 pairs each, compact pairs first, about 15 h; `fpl3.experiments.full_run`
  saves every finished chunk, resumes after an interruption and writes
  runs/full/progress.html, report.json and scores/<algorithm>.csv. TAR at FAR
  uses the lowest threshold at which FAR does not exceed the target; this rule
  awaits the researcher's confirmation before it enters decisions.md.
- Report results to the researcher as plain TAR / FAR / FRR with pair counts;
  avoid AUC and statistical jargon unless asked.
- Do not push to GitHub until the researcher asks.
