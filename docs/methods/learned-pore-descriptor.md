# Our method, part 1: a learned pore descriptor

Status: **approved and run, 2026-10-08** (results below).

[Research overview](../research-projects.md) · [Repository](../../README.md)

## Results on the test pairs (C1; 100 subjects never used for training, early stopping or fusion weights)

TAR at FAR 0.1 % with 95 % subject-bootstrap intervals (runs/method/fusion/*/report.json):

| | Rolled vs rolled (1,000 / 99,000) | Plain vs rolled (460 / 45,540) |
|---|---|---|
| OpenCV SIFT | 64.7 % [56.4, 74.8] | 64.6 % [53.7, 75.1] |
| Pore SIFT | 76.5 % [72.9, 81.1] | 53.7 % [44.6, 62.1] |
| Reference fusion of the two (step 0) | 87.4 % [83.5, 91.4] | 76.5 % [68.3, 84.3] |
| **Learned pore descriptor alone** | **96.9 % [95.3, 98.1]** | **85.0 % [80.0, 90.5]** |
| Learned + best existing (F4) | 98.0 % [97.0, 99.0] (+ Pore SIFT) | 88.9 % [84.0, 92.7] (+ OpenCV SIFT) |
| Learned + both existing | 98.6 % [97.8, 99.4] | 91.5 % [87.3, 94.2] |

- Against Pore SIFT, using the same detector and `basic` matching rule, the paired TAR gain at
  FAR 0.1 % is +16.1 to +23.4 points (rolled) and +26.7 to +37.7 (plain).
- F4 primary: FRR at FAR 0.1 % falls from 23.5 % to 2.0 % (rolled) and from
  35.4 % to 11.1 % (plain); the paired intervals exclude zero by a wide margin.
- EER of the learned descriptor alone: 1.46 % (rolled), 5.65 % (plain).
- Fusion weights involving the learned descriptor come from the 20 validation
  subjects; the reference fusion uses the full development half. Next to the
  learned channel, Pore SIFT gets a negative weight (both count pore matches).
- Training: 453,149 automatically assigned pore identities (1,064,023 patches) from 80 development
  subjects; best validation EER 0.71 % at step 3,000, early stop at step 8,000,
  about 15 minutes on the RTX 5080. Scoring all 582,724 pairs: 20 minutes.

At the 0.1% FAR target, the first three-method fusion accepts 986/1,000 genuine
rolled pairs and 421/460 genuine plain pairs. Its FRR is 1.4% (14/1,000) and
8.5% (39/460); observed FAR is 0.1% (99/99,000) and 0.0988% (45/45,540).
These are reporting points from the test scores (E3), not deployed decision
thresholds. Results compare the methods on this protocol, not across datasets.

### Repeat training with another seed (robustness check, 2026-10-08)

Same data and recipe, training seed from the purpose `descriptor-training:repeat`
(R5); runs/method/repeat, scores in learned-pore-repeat.csv. No subject overlaps
between the 80 training, 20 validation and 100 test subjects (checked).

| TAR at FAR 0.1 % | Rolled vs rolled | Plain vs rolled |
|---|---|---|
| First training (validation EER 0.71 %) | 96.9 % [95.3, 98.1] | 85.0 % [80.0, 90.5] |
| Second training (validation EER 0.50 %) | 98.1 % [97.0, 99.1] | 88.9 % [85.1, 92.9] |
| Second minus first, paired | +0.5 to +2.2 points | +1.2 to +6.4 points |
| Second, fused with both existing algorithms | 99.2 % [98.4, 99.7] | 93.9 % [91.4, 96.0] |

Both trainings stay far above the existing algorithms; the second, which the
validation subjects would also have chosen, is slightly better.

The second three-method fusion accepts 992/1,000 and 432/460 genuine pairs;
FRR is 0.8% (8/1,000) and 6.1% (28/460). False accepts remain 99/99,000 and
45/45,540 at the reported operating points. This is a repeat on the same split,
not a new evaluation population.

## Goal and yardstick

- F4 primary: fusing the Level-3 channel with the best existing algorithm lowers
  FRR at FAR 0.1 % on the test pairs (C1), judged by the paired subject
  bootstrap. Secondary: the channel alone against the best existing algorithm.
- The bar on test pairs (`runs/full/eval`, 2026-10-08), TAR at FAR 0.1 %:
  rolled vs rolled, Pore SIFT 76.5 % [72.9, 81.1]; plain vs rolled, OpenCV SIFT
  64.6 % [53.7, 75.1].

## Why this comes first

- The approved [Pore SIFT composition](../algorithms/pore-sift.md) combines the
  survey detector with Dahia & Pamplona Segundo's SIFT descriptors and `basic`
  matching. Their paper's own method replaces SIFT with a CNN descriptor
  learned from automatically annotated pores, and their ablation (same detector,
  same matching) shows the learned descriptor ahead of SIFT on PolyU-HRF. Their
  trained weights were refused, but the procedure and code are public
  (`third_party/dahia-pipeline`: `polyu/preprocess.py`, `align.py`,
  `models/description.py`, `train.py`).
- Training that descriptor on SD302 development subjects at native 1000 ppi
  gives a learned Level-3 channel sharing Pore SIFT's detector and matching rule.
  Its 32 × 32 patches must fit inside the image, so border detections can be
  omitted. It is the learning model the researcher approved (D2 entry,
  2026-10-08) and the Level-3 channel F4 asks for.

## Implemented method

0. **Reference fusion first.** Fuse the two existing algorithms (Pore SIFT is
   the existing Level-3 channel) with a logistic model fitted on development
   pairs, and evaluate on test pairs. It shows what fusion alone gives, so the
   learned channel's own contribution can be told apart later.
1. **Pores:** the survey detector exactly as in Pore SIFT. U, V and R already
   have detections from the full run; Q and J (fingers segmented from upper
   palms, native 1000 ppi) are added for development subjects only.
2. **Pore identities:** adapt the authors' correspondence-based annotation to
   full SD302 prints. [Alignment](../../src/fpl3/method/annotate.py) starts with
   OpenCV SIFT/FLANN and a RANSAC similarity transform, then uses block matching
   and a local moving-least-squares warp to accommodate deformation. A final
   correlation check locates corresponding 32 × 32 windows in other impressions.
   These are automatically assigned training identities, not manually verified
   anatomical labels. Patches retain the native sampling: 0.81 mm at 1000 ppi.
3. **Network**, as in `models/description.py`: six 3 × 3 convolutions (32, 32,
   64, 64, 128, 128 filters), an 8 × 8 convolution to a 128-dimensional
   L2-normalised descriptor. [The PyTorch implementation](../../src/fpl3/method/descriptor.py)
   is initialized randomly and trained from scratch, without pretrained
   descriptor weights. It uses triplet semi-hard loss, SGD at learning rate
   0.1 without weight decay, augmentation and dropout 0.3. Early stopping uses
   whole-print UxV validation EER on the 20 validation subjects (R7).
4. **Matching and score:** the same as Pore SIFT (`basic`, ratio 0.7) with the
   learned descriptors in place of SIFT.
5. **Evaluation:** on test pairs, alone and fused with the existing algorithms.
   Learned-channel fusion is fitted on validation pairs only; the reference
   OpenCV SIFT + Pore SIFT fusion is fitted on development pairs (R7).

The implementation details above correct the earlier design summary's references
to Horn alignment, fine-tuning and weight decay. This documentation correction
(2026-10-16, D2/R7) changes no code, training run, score or reported result.

## Later parts

- Part 2 as proposed (a fine-tuned general vision model on the whole print at
  500 ppi) was withdrawn on 2026-10-10: our methods take native ≥1000 ppi input
  (D1). The next methods follow the Stage B work order in AGENTS.md.

## Execution and approval

The method, reference fusion, GPU environment and use of Q/J development images
were approved on 2026-10-08 (D2, F3/F4, G3 and B2/R7 in the
[decision record](../decisions.md#2026-10-08--threshold-rule-and-the-researchs-own-methods)).
GPU work uses the separate `fingerprint-level3-gpu` Conda environment with
PyTorch 2.13 and CUDA 13.0. Model training and scoring are implemented in
`fpl3.method.descriptor`; fusion in `fpl3.method.fusion`. Reproducing a run also
requires the external data and locally retained artifacts. Software checks
for this path are `python -m pytest tests/test_method.py` in that environment.
