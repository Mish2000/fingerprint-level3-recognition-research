"""Keyed-hash ranking (R5).

Every random choice in the data layer ranks its candidates by
SHA-256(master_seed | purpose | item) and takes a prefix. The result depends
only on the seed, the purpose string and the candidate keys -- not on the
input order, the platform or the Python version -- and adding a new purpose
never changes an existing sample.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from typing import TypeVar

T = TypeVar("T")

_SEPARATOR = "\x1f"


def keyed_hash(master_seed: str, purpose: str, item: str) -> str:
    return hashlib.sha256(_SEPARATOR.join((master_seed, purpose, item)).encode("utf-8")).hexdigest()


def keyed_rank(items: Iterable[T], master_seed: str, purpose: str, key: Callable[[T], str]) -> list[T]:
    """Return the items ordered by keyed hash; ties (practically impossible) fall back to the key."""
    return sorted(items, key=lambda item: (keyed_hash(master_seed, purpose, key(item)), key(item)))
