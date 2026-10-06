"""Build the data layer: manifests/images.csv, manifests/pairs.csv and manifests/build_report.json.

Outputs are deterministic (fixed column order, LF line endings, no timestamps),
so a rebuild over unchanged inputs reproduces them byte for byte.

    python -m fpl3.data.build
"""

from __future__ import annotations

import argparse
import csv
import json
import tomllib
from collections import Counter
from pathlib import Path

from .manifest import MANIFEST_FIELDS, build_image_manifest, load_exclusions
from .pairs import PAIR_FIELDS, genuine_pairs, impostor_pairs
from .sd302 import ERRATA_FILE, IMAGE_SETS, read_versions, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[3]


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build(config_path: Path, exclusions_path: Path, out_dir: Path) -> dict:
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    root = Path(config["dataset"]["root"])
    master_seed = config["randomness"]["master_seed"]

    images, problems = build_image_manifest(root, load_exclusions(exclusions_path))

    pairs: list[dict] = []
    scenarios: dict[str, dict] = {}
    for name, scenario in sorted(config["scenarios"].items()):
        genuine = genuine_pairs(images, name, scenario["probe"], scenario["reference"])
        impostor = impostor_pairs(
            images, name, scenario["probe"], scenario["reference"], scenario["impostors_per_finger"], master_seed
        )
        pairs += genuine + impostor
        scenarios[name] = {
            "probe": scenario["probe"],
            "reference": scenario["reference"],
            "genuine": len(genuine),
            "impostor": len(impostor),
            "genuine_per_finger": dict(sorted(Counter(f"{p['frgp']:02d}" for p in genuine).items())),
            "impostor_per_finger": dict(sorted(Counter(f"{p['frgp']:02d}" for p in impostor).items())),
            "probe_subjects_in_genuine": len({p["probe_subject"] for p in genuine}),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "images.csv", MANIFEST_FIELDS, images)
    write_csv(out_dir / "pairs.csv", PAIR_FIELDS, pairs)

    report = {
        "dataset": {
            "root": root.as_posix(),
            "versions": read_versions(root),
            "errata_sha256": sha256_file(root / ERRATA_FILE),
        },
        "master_seed": master_seed,
        "images": {
            image_set.code: {
                "total": sum(1 for r in images if r["set"] == image_set.code),
                "excluded": sum(1 for r in images if r["set"] == image_set.code and r["excluded"]),
                "subjects": len({r["subject"] for r in images if r["set"] == image_set.code}),
            }
            for image_set in IMAGE_SETS
        },
        "checksum": dict(sorted(Counter(r["checksum"] for r in images).items())),
        "header_ppi": dict(sorted(Counter(f"{r['ppi_x']}x{r['ppi_y']}" for r in images).items())),
        "excluded_images": {r["image_id"]: r["exclusion_reason"] for r in images if r["excluded"]},
        "dataset_problems": problems,
        "scenarios": scenarios,
        "split": {"active": config["split"]["active"], "note": "C1 defined in fpl3.data.split; not applied"},
    }
    with open(out_dir / "build_report.json", "w", newline="\n", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "protocol.toml")
    parser.add_argument("--exclusions", type=Path, default=REPO_ROOT / "configs" / "exclusions.csv")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "manifests")
    args = parser.parse_args(argv)
    report = build(args.config, args.exclusions, args.out)
    for code, counts in report["images"].items():
        print(f"{code}: {counts['total']} images, {counts['excluded']} excluded, {counts['subjects']} subjects")
    for name, scenario in report["scenarios"].items():
        print(f"{name}: {scenario['genuine']} genuine, {scenario['impostor']} impostor")
    print(f"checksum: {report['checksum']}  header ppi: {report['header_ppi']}")
    if report["dataset_problems"]:
        print("dataset problems:", *report["dataset_problems"], sep="\n  ")


if __name__ == "__main__":
    main()
