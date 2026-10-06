"""Shared helpers for experiment entry points."""

from __future__ import annotations

import csv
import json
import platform
import tomllib
from collections import defaultdict
from pathlib import Path

from ..data.build import write_csv

REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_FIELDS = ("image_id", "set", "subject", "frgp")


def load_toml(name: str) -> dict:
    with open(REPO_ROOT / "configs" / name, "rb") as f:
        return tomllib.load(f)


def read_images() -> list[dict]:
    with open(REPO_ROOT / "manifests" / "images.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_pairs() -> list[dict]:
    with open(REPO_ROOT / "manifests" / "pairs.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def image_path(row: dict, dataset_root: Path) -> str:
    return str(dataset_root / row["relpath"])


def save_sample(name: str, images: list[dict], pairs: list[dict]) -> None:
    """Saved samples are committed so every run can be traced to its exact inputs (R5)."""
    from ..data.pairs import PAIR_FIELDS

    write_csv(REPO_ROOT / "manifests" / f"{name}_images.csv", SAMPLE_FIELDS, [{k: r[k] for k in SAMPLE_FIELDS} for r in images])
    write_csv(REPO_ROOT / "manifests" / f"{name}_pairs.csv", PAIR_FIELDS, pairs)


def new_run_dir(path: Path) -> Path:
    """Runs never overwrite an earlier run."""
    if path.exists():
        raise SystemExit(f"run directory already exists: {path}")
    path.mkdir(parents=True)
    return path


def write_json(path: Path, data: dict) -> None:
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def workload(images: list[dict], compact_pairs: list[dict]) -> dict:
    """Image and comparison counts of the compact design (manifests/pairs.csv) and of the full design (B3)."""
    usable = [r for r in images if not int(r["excluded"])]
    references = defaultdict(set)
    for r in usable:
        if r["set"] == "V":
            references[int(r["frgp"])].add(r["subject"])
    full = 0
    for r in usable:
        if r["set"] in ("U", "R"):
            others = references[int(r["frgp"])]
            full += len(others) - (1 if r["subject"] in others else 0)  # impostors
            full += 1 if r["subject"] in others else 0  # genuine
    compact_images = {p["probe_image_id"] for p in compact_pairs} | {p["reference_image_id"] for p in compact_pairs}
    all_images = [r for r in usable if r["set"] in ("U", "V", "R")]
    return {
        "compact": {"images": len(compact_images), "comparisons": len(compact_pairs)},
        "full": {"images": len(all_images), "comparisons": full},
    }


def host() -> dict:
    return {"platform": platform.platform(), "processor": platform.processor(), "python": platform.python_version()}
