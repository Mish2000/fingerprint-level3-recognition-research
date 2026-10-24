# Learning from Fingerprint Pores

Fingerprint verification on **native 1000 ppi NIST SD302 live-scan images**,
combining a learned pore descriptor with OpenCV SIFT and Pore SIFT.
Research by **Michael Sirkovich**, under academic supervision.

**[Research overview](docs/research-projects.md)** ·
[Detailed results and method](docs/methods/learned-pore-descriptor.md) ·
[Protocol decisions](docs/decisions.md)

## Results

TAR (true-accept rate) at FAR (false-accept rate) **≤ 0.1%**, on subjects held out
from training, early stopping and fusion fitting:

| Method | Rolled vs rolled | Plain vs rolled |
|---|---:|---:|
| OpenCV SIFT + Pore SIFT | 87.4% | 76.5% |
| Learned pore descriptor alone | 96.9% | 85.0% |
| Learned pore descriptor + OpenCV SIFT + Pore SIFT | **98.6%** | **91.5%** |

Test pairs: **1,000 genuine / 99,000 impostor** rolled comparisons and
**460 / 45,540** plain comparisons. The first three-method fusion rejects
14/1,000 and 39/460 genuine pairs (FRR **1.4% / 8.5%**), with 99/99,000 and
45/45,540 false accepts. A second training seed reaches **99.2% / 93.9% TAR**
with the same fusion. [All methods, counts and intervals](docs/methods/learned-pore-descriptor.md).

Plain vs plain across two scanners (2026-10-19), TAR at FAR ≤ 0.1%: the learned
descriptor (second training) reaches **95.1% / 93.4%** (RxJ / QxJ), against
70.8% / 68.1% for OpenCV SIFT and 77.9% / 78.5% for Pore SIFT, on 226 / 423
genuine and 29,628 / 34,623 impostor test pairs.

## What is implemented

- **Data and evaluation:** verified image metadata and exclusions, deterministic
  pair lists, subject separation, resumable runs and paired evaluation.
- **Learning:** automatic correspondence generation, **1,064,023 patches** from
  80 training subjects, a PyTorch pore-description network trained from scratch,
  and 20 validation subjects for early stopping and learned-channel fusion.
- **Execution:** one extraction per image, cached descriptors and GPU batch
  matching. OpenCV SIFT, Pore SIFT and the learned descriptor were each scored
  on **582,724 pairs**; the reported test evaluation uses 100 held-out subjects.

The descriptor follows Dahia & Pamplona Segundo's published architecture.
The local contribution is its PyTorch implementation, SD302 training-data
pipeline, training, matching execution and controlled evaluation. The
[method note](docs/methods/learned-pore-descriptor.md) describes attribution and
implementation details.

| Code | Responsibility |
|---|---|
| [data](src/fpl3/data) | SD302 readers, manifests, pairs and subject split |
| [algorithms](src/fpl3/algorithms) | OpenCV SIFT and Pore SIFT integrations |
| [method](src/fpl3/method) | Alignment, learned pore descriptor and score fusion |
| [experiments](src/fpl3/experiments) / [eval](src/fpl3/eval) | Runs, timing, evaluation and result pages |

## Software checks

Core environment, using conda-forge:

```bash
conda create -n fingerprint-level3-recognition-research --override-channels -c conda-forge python=3.12 pip setuptools pytest numpy "opencv=4.13.0"
conda activate fingerprint-level3-recognition-research
pip install -e . --no-deps --no-build-isolation
```

```bash
pytest -m "not dataset"
```

Optional-runtime tests skip when their dependencies are absent. Pore SIFT uses
its [separate environment](environments/pore-sift-requirements.txt); learned-model
tests and execution require the PyTorch GPU environment described in the
[method note](docs/methods/learned-pore-descriptor.md).

Research runs additionally need the local SD302 data, the pinned upstream
components in [algorithm configuration](configs/algorithms.toml), and the
[data configuration](configs/protocol.toml). The dataset must be obtained from
NIST under its own terms. Images, crops, templates, descriptors and weights
remain local; public source and aggregate results can be inspected independently.

## Research context

This is the current research repository (Stage A). The [six-repository overview](docs/research-projects.md)
places it after the earlier application workbench, the completed 500 ppi
benchmark, the synthetic localization study and the historical SD300B Level-3
study, all now archived. The present results concern 1:1 verification; Stage B,
[multimodal-biometric-system](https://github.com/Mish2000/multimodal-biometric-system),
is building them into a system with 1:N identification.
