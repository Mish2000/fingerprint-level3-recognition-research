"""Shared data for our method: development fingers, the validation subjects inside development,
and where every artefact lives (all under runs/, never in git: R8)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ..data.sampling import keyed_rank
from ..data.samples import development_subjects
from ..experiments.common import REPO_ROOT, load_toml, read_images

METHOD_DIR = REPO_ROOT / "runs" / "method"
DESCRIPTOR_DATA = METHOD_DIR / "descriptor-data"
VALIDATION_SHARE = 0.2  # of development subjects, per has_R stratum, for early stopping (R7)


def protocol_and_images() -> tuple[dict, list[dict]]:
    return load_toml("protocol.toml"), read_images()


def split_development(protocol: dict, images: list[dict]) -> tuple[list[str], list[str]]:
    """Training and validation subjects inside the development half, by keyed hash per has_R stratum (R5)."""
    seed = protocol["randomness"]["master_seed"]
    development, strata = development_subjects(images, seed, protocol["split"]["dev_fraction"])
    train, validation = [], []
    for stratum in ("has_R", "no_R"):
        members = keyed_rank(sorted(s for s in development if strata[s] == stratum), seed, f"method-validation:{stratum}", key=str)
        cut = round(VALIDATION_SHARE * len(members))
        validation += members[:cut]
        train += members[cut:]
    return sorted(train), sorted(validation)


def fingers(images: list[dict], subjects: list[str], sets: str = "UVRQJ") -> dict[tuple[str, int], dict[str, dict]]:
    """(subject, finger position) -> {set code: image row} for usable native 1000 ppi impressions."""
    wanted = set(subjects)
    out: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for row in images:
        if row["set"] in sets and row["subject"] in wanted and not int(row["excluded"]):
            out[(row["subject"], int(row["frgp"]))][row["set"]] = row
    return dict(out)


def dataset_root(protocol: dict) -> Path:
    return Path(protocol["dataset"]["root"])
