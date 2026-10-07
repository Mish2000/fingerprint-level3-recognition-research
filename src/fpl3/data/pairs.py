"""Genuine and impostor pairs of a probe set against a reference set (B1-B3).

Genuine: same subject and same finger position (FRGP).
Impostor: same finger position, different subjects.
Excluded images never enter a pair.
"""

from __future__ import annotations

from collections import defaultdict

from .sampling import keyed_rank

PAIR_FIELDS = (
    "pair_id",
    "scenario",
    "kind",
    "frgp",
    "probe_image_id",
    "reference_image_id",
    "probe_subject",
    "reference_subject",
)


def _usable(manifest: list[dict], set_code: str) -> list[dict]:
    return [row for row in manifest if row["set"] == set_code and not int(row["excluded"])]


def pair_record(scenario: str, kind: str, probe: dict, reference: dict) -> dict:
    return {
        "pair_id": f"{scenario}:{probe['image_id']}:{reference['image_id']}",
        "scenario": scenario,
        "kind": kind,
        "frgp": int(probe["frgp"]),
        "probe_image_id": probe["image_id"],
        "reference_image_id": reference["image_id"],
        "probe_subject": probe["subject"],
        "reference_subject": reference["subject"],
    }


def _order(pair: dict) -> tuple:
    return (pair["frgp"], pair["probe_image_id"], pair["reference_image_id"])


def genuine_pairs(manifest: list[dict], scenario: str, probe_set: str, reference_set: str) -> list[dict]:
    references = {(row["subject"], int(row["frgp"])): row for row in _usable(manifest, reference_set)}
    pairs = [
        pair_record(scenario, "genuine", probe, references[(probe["subject"], int(probe["frgp"]))])
        for probe in _usable(manifest, probe_set)
        if (probe["subject"], int(probe["frgp"])) in references
    ]
    return sorted(pairs, key=_order)


def all_impostor_pairs(manifest: list[dict], scenario: str, probe_set: str, reference_set: str) -> list[dict]:
    """Every impostor pair: each probe against each reference of the same finger position from
    another subject (B3 full design, decision record 2026-10-07)."""
    references: dict[int, list[dict]] = defaultdict(list)
    for row in _usable(manifest, reference_set):
        references[int(row["frgp"])].append(row)
    pairs = [
        pair_record(scenario, "impostor", probe, reference)
        for probe in _usable(manifest, probe_set)
        for reference in references.get(int(probe["frgp"]), [])
        if probe["subject"] != reference["subject"]
    ]
    return sorted(pairs, key=_order)


def impostor_pairs(
    manifest: list[dict],
    scenario: str,
    probe_set: str,
    reference_set: str,
    per_finger: int,
    master_seed: str,
) -> list[dict]:
    """Sample `per_finger` impostor pairs for every finger position of the probe set."""
    probes: dict[int, list[dict]] = defaultdict(list)
    references: dict[int, list[dict]] = defaultdict(list)
    for row in _usable(manifest, probe_set):
        probes[int(row["frgp"])].append(row)
    for row in _usable(manifest, reference_set):
        references[int(row["frgp"])].append(row)

    pairs = []
    for frgp in sorted(probes):
        candidates = [
            (probe, reference)
            for probe in probes[frgp]
            for reference in references.get(frgp, [])
            if probe["subject"] != reference["subject"]
        ]
        if len(candidates) < per_finger:
            raise ValueError(f"{scenario} finger {frgp:02d}: only {len(candidates)} impostor candidates")
        chosen = keyed_rank(
            candidates,
            master_seed,
            f"impostors:{scenario}:frgp{frgp:02d}",
            key=lambda candidate: f"{candidate[0]['image_id']}|{candidate[1]['image_id']}",
        )[:per_finger]
        pairs.extend(pair_record(scenario, "impostor", probe, reference) for probe, reference in chosen)
    return sorted(pairs, key=_order)
