"""NIST SD302 facts and file readers.

The manifest indexes only native-1000-ppi single-finger exemplar images
(NIST TN 2007, Sections 3 and 4.3):

- U, V: operator-assisted rolled impressions by two FBI experts (sd302b).
- R: plain 4-4-2 slap impressions segmented into fingers by NIST (sd302b).
- Q, J: fingers segmented by NIST from upper-palm captures (sd302c).

Every other exemplar set in SD302 is below 1000 ppi; latents are out of scope (A1).
"""

from __future__ import annotations

import csv
import hashlib
import re
import struct
from dataclasses import dataclass
from pathlib import Path

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
INCHES_PER_METRE = 0.0254


@dataclass(frozen=True)
class ImageSet:
    code: str
    directory: str  # relative to the SD302 root
    capture_label: str  # CAPTURE field of the NIST file names
    impression: str  # rolled or plain
    device_model: str


IMAGE_SETS = (
    ImageSet("U", "sd302b/images/baseline/U/1000/roll", "roll", "rolled", "Crossmatch L SCAN 1000PX"),
    ImageSet("V", "sd302b/images/baseline/V/1000/roll", "roll", "rolled", "Crossmatch L SCAN 1000PX"),
    ImageSet("R", "sd302b/images/baseline/R/1000/slap-segmented", "slap", "plain", "Crossmatch L SCAN 1000PX"),
    ImageSet("Q", "sd302c/images/auxiliary/palm/Q/1000/palm-segmented", "palm", "plain", "Crossmatch L SCAN 1000PX"),
    ImageSet("J", "sd302c/images/auxiliary/palm/J/1000/palm-segmented", "palm", "plain", "Morpho TouchPrint 5300"),
)

VERSION_FILES = {
    "sd302b": "sd302b/images/baseline/VERSION_302b.txt",
    "sd302c": "sd302c/images/auxiliary/palm/VERSION_302c.txt",
}
ERRATA_FILE = "ERRATA_SD302.txt"

_FILENAME = re.compile(
    r"(?P<subject>\d{8})_(?P<device>[A-Z])_(?P<ppi>\d+)_(?P<capture>[a-z]+)_(?P<frgp>\d{2})\.png"
)


@dataclass(frozen=True)
class FilenameFields:
    subject: str
    device: str
    ppi: int
    capture: str
    frgp: int


def parse_filename(name: str) -> FilenameFields:
    """Parse SUBJECT_DEVICE_RESOLUTION_CAPTURE_FRGP.png (TN 2007, Section 5.1)."""
    match = _FILENAME.fullmatch(name)
    if match is None:
        raise ValueError(f"not an SD302 exemplar file name: {name}")
    return FilenameFields(
        subject=match["subject"],
        device=match["device"],
        ppi=int(match["ppi"]),
        capture=match["capture"],
        frgp=int(match["frgp"]),
    )


@dataclass(frozen=True)
class PngHeader:
    width: int
    height: int
    bit_depth: int
    color_type: int
    ppi_x: float | None
    ppi_y: float | None


def read_png_header(path: Path) -> PngHeader:
    """Read IHDR and pHYs from the chunks before the first IDAT, without decoding pixels."""
    ihdr = None
    ppi_x = ppi_y = None
    with open(path, "rb") as f:
        if f.read(8) != PNG_SIGNATURE:
            raise ValueError(f"not a PNG file: {path}")
        while True:
            head = f.read(8)
            if len(head) < 8:
                break
            length, chunk_type = struct.unpack(">I4s", head)
            if chunk_type in (b"IDAT", b"IEND"):
                break
            data = f.read(length)
            f.read(4)  # CRC
            if chunk_type == b"IHDR":
                ihdr = struct.unpack(">IIBB", data[:10])
            elif chunk_type == b"pHYs":
                per_unit_x, per_unit_y, unit = struct.unpack(">IIB", data)
                if unit == 1:  # pixels per metre
                    ppi_x = round(per_unit_x * INCHES_PER_METRE, 2)
                    ppi_y = round(per_unit_y * INCHES_PER_METRE, 2)
    if ihdr is None:
        raise ValueError(f"PNG without IHDR: {path}")
    width, height, bit_depth, color_type = ihdr
    return PngHeader(width, height, bit_depth, color_type, ppi_x, ppi_y)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def read_checksum_csv(path: Path) -> dict[str, str]:
    """Map file name -> SHA-256. The hash column is 'sha256' in most NIST CSVs and 'checksum' in some."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    hash_column = "sha256" if "sha256" in rows[0] else "checksum"
    return {Path(row["filename"].strip()).name: row[hash_column].strip().lower() for row in rows}


def read_participants(root: Path) -> dict[str, dict[str, str]]:
    with open(root / "participants.csv", newline="", encoding="utf-8") as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def read_versions(root: Path) -> dict[str, str]:
    return {part: (root / rel).read_text(encoding="utf-8").strip() for part, rel in VERSION_FILES.items()}
