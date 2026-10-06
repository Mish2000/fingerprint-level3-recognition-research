import struct
import tomllib
import zlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_png(path: Path, width: int = 4, height: int = 3, per_metre: tuple[int, int] | None = (39370, 39370)) -> None:
    """Write a minimal valid grayscale PNG, optionally with a pHYs chunk (pixels per metre)."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + bytes(width) for _ in range(height))
    body = chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
    if per_metre is not None:
        body += chunk(b"pHYs", struct.pack(">IIB", per_metre[0], per_metre[1], 1))
    body += chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + body)


@pytest.fixture(scope="session")
def protocol() -> dict:
    with open(REPO_ROOT / "configs" / "protocol.toml", "rb") as f:
        return tomllib.load(f)


@pytest.fixture(scope="session")
def sd302_root(protocol) -> Path:
    root = Path(protocol["dataset"]["root"])
    if not (root / "participants.csv").exists():
        pytest.skip(f"SD302 not available at {root}")
    return root
