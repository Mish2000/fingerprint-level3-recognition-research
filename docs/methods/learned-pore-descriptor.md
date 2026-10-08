# Our method, part 1: a learned pore descriptor (design proposal)

Status: **proposal, 2026-10-08**, awaiting the researcher's approval.

## Goal and yardstick

- F4 primary: fusing the Level-3 channel with the best existing algorithm lowers
  FRR at FAR 0.1 % on the test pairs (C1), judged by the paired subject
  bootstrap. Secondary: the channel alone against the best existing algorithm.
- The bar on test pairs (`runs/full/eval`, 2026-10-08), TAR at FAR 0.1 %:
  rolled vs rolled, Pore SIFT 76.5 % [72.9, 81.1]; plain vs rolled, OpenCV SIFT
  64.6 % [53.7, 75.1].
- For scale only: published deep fingerprint embeddings report about 94–96 % TAR
  at FAR 0.1–0.01 % on SD302 subsets that are not comparable with ours, after
  training on hundreds of thousands of external prints; no weights are public.

## Why this comes first

- Pore SIFT is Dahia & Pamplona Segundo's pipeline with its hand-crafted SIFT
  descriptor. Their paper's own method replaces SIFT with a CNN descriptor
  learned from automatically annotated pores, and their ablation (same detector,
  same matching) shows the learned descriptor ahead of SIFT on PolyU-HRF. Their
  trained weights were refused, but the procedure and code are public
  (`third_party/dahia-pipeline`: `polyu/preprocess.py`, `align.py`,
  `models/description.py`, `train.py`).
- Training that descriptor on SD302 development subjects at native 1000 ppi
  gives a learned Level-3 channel whose comparison with Pore SIFT changes only
  the descriptor. It is the learning model the researcher approved (D2 entry,
  2026-10-08) and the Level-3 channel F4 asks for.

## Design

0. **Reference fusion first.** Fuse the two existing algorithms (Pore SIFT is
   the existing Level-3 channel) with a logistic model fitted on development
   pairs, and evaluate on test pairs. It shows what fusion alone gives, so the
   learned channel's own contribution can be told apart later.
1. **Pores:** the survey detector exactly as in Pore SIFT. U, V and R already
   have detections from the full run; Q and J (fingers segmented from upper
   palms, native 1000 ppi) are added for development subjects only.
2. **Pore identities**, following the authors' annotation: for each development
   finger, align its impressions by pore correspondences (Horn's absolute
   orientation, refined iteratively, as in `align.py`) and link pores that fall
   together after alignment. A patch is 32 × 32 px around a pore: 0.81 mm at
   1000 ppi, against 0.68 mm for the same 32 px at PolyU's 1200 dpi.
3. **Network**, as in `models/description.py`: six 3 × 3 convolutions (32, 32,
   64, 64, 128, 128 filters), an 8 × 8 convolution to a 128-dimensional
   L2-normalised descriptor, triplet semi-hard loss with weight decay, their
   augmentation and dropout 0.3, early stopping on the EER of a validation
   subset of development subjects (R7). Re-implemented in PyTorch, since
   TensorFlow 1.x does not install on current Python.
4. **Matching and score:** the same as Pore SIFT (`basic`, ratio 0.7) with the
   learned descriptors in place of SIFT.
5. **Evaluation:** on test pairs, only at milestones, alone and fused with the
   best existing algorithm (fusion fitted on development pairs).

## Later parts

- Part 2: a fine-tuned general vision model as a fixed-length embedding of the
  whole print (Level 1–2), the kind of model the supervisor asked about; then
  fusion of every channel.

## Needs the researcher's approval

1. This design, including step 0.
2. A GPU environment: PyTorch for the RTX 5080 from conda-forge, about 2–3 GB.
   Training also runs on the CPU, many times slower.
3. Training data: every native 1000 ppi impression of development subjects,
   including Q and J, which so far stay in the manifest only (B2).
