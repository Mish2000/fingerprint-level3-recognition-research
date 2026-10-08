"""Evaluation of a full run (E2-E4): EER, TAR at FAR and DET curves, with 95 % confidence intervals
from a subject bootstrap and paired differences between algorithms on identical pairs.

    python -m fpl3.experiments.evaluate [--run runs/full] [--replicates 1000]

Designs: 'full' (every pair of the run, all subjects) and 'test' (C1: both subjects in the test
half, the pairs on which the research's own methods are compared). Writes <run>/eval/report.json
and <run>/eval/det-<design>-<scenario>.svg.
"""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np

from ..data.samples import development_subjects
from ..data.sampling import keyed_hash
from ..eval import verification as v
from ..eval.det import det_svg
from .common import REPO_ROOT, load_toml, read_images, write_json
from .full_run import NAMES

TARGETS = (0.01, 0.001, 0.0001)  # E2, decision record 2026-10-07
SCENARIO_TITLES = {"UxV": "Rolled vs rolled (UxV)", "RxV": "Plain vs rolled (RxV)"}
DESIGN_TITLES = {"full": "all 200 subjects", "test": "the 100 test subjects (C1)"}


def subject(image_id: str) -> str:
    return image_id.split("_")[1]


def read_scores(path: Path) -> tuple[list[tuple], np.ndarray]:
    """Pairs as (pair_id, scenario, kind, probe subject, reference subject), and their scores
    with -inf for failures (E1)."""
    pairs, scores = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pairs.append((row["pair_id"], row["scenario"], row["kind"],
                          subject(row["probe_image_id"]), subject(row["reference_image_id"])))
            scores.append(float(row["score"]) if row["status"] == "ok" and row["score"] != "" else v.FAILED)
    return pairs, np.array(scores)


def score_set(pairs: list[tuple], scores: np.ndarray, rows: np.ndarray, index: dict[str, int]) -> v.Scores:
    kinds = np.array([pairs[i][2] for i in rows])
    genuine, impostor = rows[kinds == "genuine"], rows[kinds == "impostor"]
    return v.Scores(
        genuine=scores[genuine],
        genuine_subject=np.array([index[pairs[i][3]] for i in genuine]),
        impostor=scores[impostor],
        impostor_probe=np.array([index[pairs[i][3]] for i in impostor]),
        impostor_reference=np.array([index[pairs[i][4]] for i in impostor]),
        n_subjects=len(index),
    )


def point(r: v.Rates, target: float) -> dict:
    at = v.tar_at_far(r, target)
    return {"tar": at["tar"], "frr": at["frr"], "far": at["far"], "accept_if_score_above": float(at["accept_if_score_above"]),
            "true_accepts": round(at["tar"] * r.genuine_total), "genuine": round(r.genuine_total),
            "false_accepts": round(at["far"] * r.impostor_total), "impostor": round(r.impostor_total)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, default=REPO_ROOT / "runs" / "full")
    parser.add_argument("--replicates", type=int, default=1000)
    args = parser.parse_args(argv)

    protocol = load_toml("protocol.toml")
    master_seed = protocol["randomness"]["master_seed"]
    development, _ = development_subjects(read_images(), master_seed, protocol["split"]["dev_fraction"])
    algorithms = [a for a in NAMES if (args.run / "scores" / f"{a}.csv").exists()]
    data = {a: read_scores(args.run / "scores" / f"{a}.csv") for a in algorithms}
    pairs = data[algorithms[0]][0]
    if any(data[a][0] != pairs for a in algorithms):
        raise SystemExit("the algorithms' score files do not list identical pairs")

    out_dir = args.run / "eval"
    out_dir.mkdir(exist_ok=True)
    report = {"replicates": args.replicates, "confidence": 0.95, "targets": list(TARGETS),
              "far_rule": "lowest threshold at which the share of accepted impostors does not exceed the target (E2)",
              "bootstrap": "subjects drawn with replacement; a genuine pair weighs its subject's count, an impostor "
                           "pair the product of its two subjects' counts (E4)", "designs": {}}
    for design in ("full", "test"):
        keep = np.array([True if design == "full" else (p[3] not in development and p[4] not in development) for p in pairs])
        subjects = sorted({s for p, k in zip(pairs, keep) if k for s in p[3:5]})
        index = {s: i for i, s in enumerate(subjects)}
        entry = report["designs"][design] = {"subjects": len(subjects), "scenarios": {}}
        for scenario in ("UxV", "RxV"):
            rows = np.nonzero(keep & np.array([p[1] == scenario for p in pairs]))[0]
            sets = {a: score_set(pairs, data[a][1], rows, index) for a in algorithms}
            seed = int(keyed_hash(master_seed, f"bootstrap:{design}", "")[:16], 16)
            boot = v.subject_bootstrap(sets, TARGETS, args.replicates, seed)
            result = {"genuine": len(sets[algorithms[0]].genuine), "impostor": len(sets[algorithms[0]].impostor),
                      "algorithms": {}, "paired": {}}
            curves, markers = {}, {}
            for a in algorithms:
                r = v.rates(sets[a], v.Grid(sets[a]))
                result["algorithms"][a] = {
                    "eer": {"value": v.eer(r), "ci": v.interval(boot[a]["eer"])},
                    "at_far": {f"{t * 100:g}%": {**point(r, t), "tar_ci": v.interval(boot[a][f"tar_at_{t!r}"])} for t in TARGETS},
                }
                curves[NAMES[a]] = (r.far, r.frr)
                markers[NAMES[a]] = [(t, v.tar_at_far(r, t)["far"], v.tar_at_far(r, t)["frr"]) for t in TARGETS]
            for a, b in combinations(algorithms, 2):
                result["paired"][f"{b} minus {a}"] = {
                    "eer": v.interval(boot[b]["eer"] - boot[a]["eer"]),
                    **{f"tar_at_{t * 100:g}%": v.interval(boot[b][f"tar_at_{t!r}"] - boot[a][f"tar_at_{t!r}"]) for t in TARGETS},
                }
            entry["scenarios"][scenario] = result
            subtitle = f"{DESIGN_TITLES[design]}: {result['genuine']:,} genuine and {result['impostor']:,} impostor pairs"
            (out_dir / f"det-{design}-{scenario}.svg").write_text(
                det_svg(f"DET, {SCENARIO_TITLES[scenario]}", subtitle, curves, markers), encoding="utf-8", newline="\n")
            print(f"{design} {scenario}: {result['genuine']} genuine, {result['impostor']} impostor")
            for a in algorithms:
                x = result["algorithms"][a]
                cells = "  ".join(f"FAR {k}: TAR {100 * y['tar']:.1f}% [{100 * y['tar_ci'][0]:.1f}, {100 * y['tar_ci'][1]:.1f}]"
                                  for k, y in x["at_far"].items())
                print(f"  {NAMES[a]:<12} EER {100 * x['eer']['value']:.2f}% [{100 * x['eer']['ci'][0]:.2f}, "
                      f"{100 * x['eer']['ci'][1]:.2f}]  {cells}")
            for name, d in result["paired"].items():
                print(f"  {name}: " + "  ".join(f"{k} [{100 * lo:+.1f}, {100 * hi:+.1f}]" for k, (lo, hi) in d.items()))
    write_json(out_dir / "report.json", report)


if __name__ == "__main__":
    main()
