# Fingerprint Recognition Research

**Michael Sirkovich** · Research under academic supervision

Computer vision, metric learning and software engineering applied to fingerprint
recognition. The current study learns descriptions of sweat-pore neighborhoods
from **native 1000 ppi NIST SD302 live-scan images**, with a measured improvement
over the existing verification methods evaluated on the same pairs.

## Main result

True-accept rate (TAR) at a false-accept rate (FAR) **at or below 0.1%**:

| Method | Rolled vs rolled | Plain vs rolled |
|---|---:|---:|
| OpenCV SIFT + Pore SIFT | 87.4% | 76.5% |
| Learned pore descriptor alone | 96.9% | 85.0% |
| Learned pore descriptor + OpenCV SIFT + Pore SIFT | **98.6%** | **91.5%** |

Adding the learned channel to the existing fusion improves TAR by **11.2 and
15.0 percentage points**, respectively. These are the first training's results;
a second training seed reached **99.2% / 93.9%** with the same three-method fusion.

Evaluation uses **100 held-out subjects**: 1,000 genuine / 99,000 impostor rolled
pairs and 460 / 45,540 plain pairs. In the first three-method fusion, false
rejects are 14/1,000 and 39/460; false accepts are 99/99,000 and 45/45,540.
[Full results, intervals and protocol](methods/learned-pore-descriptor.md).

## My contribution

- **Learning pipeline:** automatic alignment and correspondence generation for
  **1,064,023 training patches**, a PyTorch implementation of Dahia & Pamplona
  Segundo's pore-description network trained from scratch, and GPU matching.
  [Training code](../src/fpl3/method/descriptor.py) ·
  [Alignment code](../src/fpl3/method/annotate.py)
- **Controlled evaluation:** 80 training / 20 validation / 100 test subjects;
  deterministic manifests, cached extraction and identical comparison pairs.
  OpenCV SIFT, Pore SIFT and the learned descriptor were each scored on
  **582,724 pairs** across the full cohort; the result above uses its test subset.
  [Evaluation code](../src/fpl3/experiments/evaluate.py)
- **Software integration:** an earlier React/TypeScript and FastAPI workbench,
  now archived, with PostgreSQL/pgvector retrieval, pairwise reranking, a Java
  SourceAFIS service and Windows scanner capture
  ([engineering case study](https://github.com/Mish2000/fingerprint-research/blob/main/docs/ENGINEERING_CASE_STUDY.md)). The research methods now run inside
  multimodal-biometric-system (Stage B).

## Explore the work

| Period | Repository | Status | Focus |
|---|---|---|---|
| Oct 2026 to now | [multimodal-biometric-system](https://github.com/Mish2000/multimodal-biometric-system) | **In progress** (Stage B, increment 1 of 6) | A desktop system for enrollment, 1:1 verification and 1:N identification by fingerprint, face and voice |
| Oct 2026 to now | [fingerprint-level3-recognition-research](../README.md) | **Current** (Stage A) | SD302, learned pore descriptors and score fusion |
| Sep 2026 | [fingerprint-l3-benchmark](https://github.com/Mish2000/fingerprint-l3-benchmark) | Archived | Historical Level-3 study: SD300B, SIFT/Direct Pore compositions and execution provenance |
| Aug 2026 | [fingerprint-new-method](https://github.com/Mish2000/fingerprint-new-method) | Archived | ML/CV case study: U-Net pore localization on synthetic labels; real-data transfer remained inconclusive |
| Jul to Sep 2026 | [fingerprint-benchmark](https://github.com/Mish2000/fingerprint-benchmark) | Completed, archived | Six verification routes on SD300 at a shared 500 ppi processing resolution |
| Feb to Sep 2026 | [fingerprint-research](https://github.com/Mish2000/fingerprint-research) | Archived | Application workbench: verification, gallery search, UI, storage and scanner integration |

The SD302 results measure **1:1 verification under the stated protocol**. The
workbench is an earlier prototype, now archived; Stage B, multimodal-biometric-system,
is integrating the research methods into a full system with 1:N identification.
Earlier studies retain their own datasets, metrics and upstream attribution.
