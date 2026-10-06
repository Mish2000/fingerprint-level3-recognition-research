"""Image manifest: one row per native-1000-ppi single-finger exemplar image (R3)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .sd302 import (
    IMAGE_SETS,
    parse_filename,
    read_checksum_csv,
    read_participants,
    read_png_header,
    sha256_file,
)

MANIFEST_FIELDS = (
    "image_id",
    "set",
    "subject",
    "frgp",
    "impression",
    "capture_label",
    "device_model",
    "relpath",
    "width",
    "height",
    "bit_depth",
    "ppi_x",
    "ppi_y",
    "sha256",
    "checksum",
    "collection_day",
    "excluded",
    "exclusion_reason",
)

NATIVE_PPI = 1000.0


@dataclass(frozen=True)
class Exclusion:
    set_code: str
    subject: str
    frgp: int
    decision: str
    reason: str
    source: str


def load_exclusions(path: Path) -> dict[tuple[str, str, int], Exclusion]:
    exclusions: dict[tuple[str, str, int], Exclusion] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            entry = Exclusion(row["set"], row["subject"], int(row["frgp"]), row["decision"], row["reason"], row["source"])
            key = (entry.set_code, entry.subject, entry.frgp)
            if key in exclusions:
                raise ValueError(f"duplicate exclusion: {key}")
            exclusions[key] = entry
    return exclusions


def image_id(set_code: str, subject: str, frgp: int) -> str:
    return f"{set_code}_{subject}_{frgp:02d}"


def build_image_manifest(
    root: Path, exclusions: dict[tuple[str, str, int], Exclusion]
) -> tuple[list[dict], list[str]]:
    """Index every image of the five sets; return the rows and a list of dataset problems found."""
    participants = read_participants(root)
    rows: list[dict] = []
    problems: list[str] = []
    matched: set[tuple[str, str, int]] = set()

    for image_set in IMAGE_SETS:
        set_dir = root / image_set.directory
        expected: dict[str, str] = {}
        for checksum_csv in sorted(set_dir.glob("checksum*.csv")):
            expected.update(read_checksum_csv(checksum_csv))
        names = sorted(path.name for path in (set_dir / "png").glob("*.png"))
        for missing in sorted(set(expected) - set(names)):
            problems.append(f"{image_set.code}: listed in checksum CSV but missing on disk: {missing}")

        for name in names:
            fields = parse_filename(name)
            if (fields.device, fields.capture, fields.ppi) != (image_set.code, image_set.capture_label, 1000):
                raise ValueError(f"{name} does not belong to set {image_set.code}")
            path = set_dir / "png" / name
            header = read_png_header(path)
            digest = sha256_file(path)
            listed = expected.get(name)
            checksum = "ok" if listed == digest else ("not_listed" if listed is None else "mismatch")

            reasons = []
            key = (image_set.code, fields.subject, fields.frgp)
            if key in exclusions:
                matched.add(key)
                reasons.append(f"{exclusions[key].decision}: {exclusions[key].reason}")
            if checksum != "ok":
                reasons.append(f"R2: checksum {checksum}")
            if (header.ppi_x, header.ppi_y) != (NATIVE_PPI, NATIVE_PPI):
                reasons.append(f"R2: header resolution {header.ppi_x}x{header.ppi_y} ppi")
            participant = participants.get(fields.subject)
            if participant is None:
                reasons.append("subject missing from participants.csv")

            rows.append(
                {
                    "image_id": image_id(image_set.code, fields.subject, fields.frgp),
                    "set": image_set.code,
                    "subject": fields.subject,
                    "frgp": fields.frgp,
                    "impression": image_set.impression,
                    "capture_label": image_set.capture_label,
                    "device_model": image_set.device_model,
                    "relpath": f"{image_set.directory}/png/{name}",
                    "width": header.width,
                    "height": header.height,
                    "bit_depth": header.bit_depth,
                    "ppi_x": header.ppi_x,
                    "ppi_y": header.ppi_y,
                    "sha256": digest,
                    "checksum": checksum,
                    "collection_day": participant["collection_day"] if participant else "",
                    "excluded": 1 if reasons else 0,
                    "exclusion_reason": "; ".join(reasons),
                }
            )

    unmatched = sorted(set(exclusions) - matched)
    if unmatched:
        raise ValueError(f"exclusions that match no image: {unmatched}")
    return rows, problems
