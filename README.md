# fingerprint-level3-recognition-research

Research code for evaluating fingerprint verification algorithms designed for
≥1000 ppi input, and for developing a new method that also uses Level-3
features (sweat pores), on native 1000 ppi live-scan images from NIST Special
Database 302. Research under academic supervision.

The dataset itself is not part of this repository and must be obtained from
NIST under its own terms. No fingerprint imagery or derived representation is
stored here.

## Setup

```bash
conda create -n fingerprint-level3-recognition-research --override-channels -c conda-forge python=3.12 pip setuptools pytest
conda activate fingerprint-level3-recognition-research
pip install -e . --no-deps --no-build-isolation
```

`environment.yml` lists the same packages. `--override-channels` keeps conda on
conda-forge even when a local configuration also names Anaconda's `defaults`
channel.

## Data layer

```bash
python -m fpl3.data.build      # writes manifests/images.csv, pairs.csv, build_report.json
pytest -m "not dataset"        # unit tests
pytest                         # also the integration tests against the local SD302 copy
```

The protocol and every decision behind it are in [docs/decisions.md](docs/decisions.md);
the data-layer settings are in [configs/protocol.toml](configs/protocol.toml) and
the image exclusions in [configs/exclusions.csv](configs/exclusions.csv).
