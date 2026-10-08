"""Score fusion for our method (docs/methods/learned-pore-descriptor.md, step 0 and later steps).

Linear logistic fusion with equal class weights, fitted on development pairs only (R7) and
separately per comparison type (R6); scores enter as log(1 + score), failures as the channel's
lowest development score (E1). Evaluated on the test pairs (C1) with the subject bootstrap (E4).

    python -m fpl3.method.fusion [--channels opencv-sift pore-sift] [--name reference]
"""

from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np

from ..data.sampling import keyed_hash
from ..data.samples import development_subjects
from ..eval import verification as v
from ..eval.det import det_svg
from ..experiments.common import REPO_ROOT, load_toml, read_images, write_json
from ..experiments.evaluate import TARGETS, point, read_scores, score_set
from ..experiments.full_run import NAMES
from .data import split_development

SCORE_DIRS = {"opencv-sift": REPO_ROOT / "runs" / "full" / "scores", "pore-sift": REPO_ROOT / "runs" / "full" / "scores"}


def fit_logistic(features: np.ndarray, labels: np.ndarray, iterations: int = 100) -> np.ndarray:
    """Weights (last = bias) of a logistic model with both classes weighted equally (Newton steps)."""
    x = np.column_stack([features, np.ones(len(features))])
    positive = labels == 1
    sample = np.where(positive, 0.5 / positive.sum(), 0.5 / (~positive).sum())
    w = np.zeros(x.shape[1])
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-(x @ w)))
        gradient = x.T @ (sample * (p - labels)) + 1e-9 * w
        hessian = (x * (sample * p * (1 - p))[:, None]).T @ x + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(hessian, gradient)
        w -= step
        if np.abs(step).max() < 1e-12:
            break
    return w


def channel_features(scores: dict[str, np.ndarray], rows: np.ndarray, floors: dict[str, float]) -> np.ndarray:
    return np.column_stack([np.log1p(np.where(np.isfinite(s[rows]), s[rows], floors[c])) for c, s in scores.items()])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--channels", nargs="+", default=["opencv-sift", "pore-sift"])
    parser.add_argument("--name", default="reference")
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--fit", choices=("development", "validation"), default="development",
                        help="'validation': only pairs of the development subjects that trained nothing (fpl3.method.data)")
    args = parser.parse_args(argv)

    protocol = load_toml("protocol.toml")
    master_seed = protocol["randomness"]["master_seed"]
    development, _ = development_subjects(read_images(), master_seed, protocol["split"]["dev_fraction"])
    fit_subjects = set(split_development(protocol, read_images())[1]) if args.fit == "validation" else development
    data = {c: read_scores(SCORE_DIRS.get(c, REPO_ROOT / "runs" / "method" / "scores") / f"{c}.csv") for c in args.channels}
    pairs = data[args.channels[0]][0]
    if any(data[c][0] != pairs for c in args.channels):
        raise SystemExit("the channels' score files do not list identical pairs")
    scores = {c: data[c][1] for c in args.channels}
    half = np.array(["dev" if p[3] in fit_subjects and p[4] in fit_subjects else
                     "test" if p[3] not in development and p[4] not in development else "other" for p in pairs])
    scenario = np.array([p[1] for p in pairs])
    genuine = np.array([p[2] == "genuine" for p in pairs])

    out_dir = REPO_ROOT / "runs" / "method" / "fusion" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    subjects = sorted({s for p, h in zip(pairs, half) if h == "test" for s in p[3:5]})
    index = {s: i for i, s in enumerate(subjects)}
    report = {"channels": args.channels, "fit": f"{args.fit} pairs", "evaluated": "test pairs (C1)", "scenarios": {}}
    for sc in ("UxV", "RxV"):
        dev_rows = np.nonzero((half == "dev") & (scenario == sc))[0]
        test_rows = np.nonzero((half == "test") & (scenario == sc))[0]
        floors = {c: float(np.min(s[dev_rows][np.isfinite(s[dev_rows])])) for c, s in scores.items()}
        w = fit_logistic(channel_features(scores, dev_rows, floors), genuine[dev_rows].astype(float))
        fused = np.full(len(pairs), np.nan)
        fused[test_rows] = channel_features(scores, test_rows, floors) @ w[:-1]
        sets = {c: score_set(pairs, scores[c], test_rows, index) for c in args.channels}
        sets["fusion"] = score_set(pairs, fused, test_rows, index)
        seed = int(keyed_hash(master_seed, "bootstrap:test", "")[:16], 16)  # the same draws as fpl3.experiments.evaluate
        boot = v.subject_bootstrap(sets, TARGETS, args.replicates, seed)
        result = {"weights": {c: float(x) for c, x in zip(args.channels, w[:-1])}, "algorithms": {}, "paired": {}}
        curves, markers = {}, {}
        for name, s in sets.items():
            r = v.rates(s, v.Grid(s))
            result["algorithms"][name] = {
                "eer": {"value": v.eer(r), "ci": v.interval(boot[name]["eer"])},
                "at_far": {f"{t * 100:g}%": {**point(r, t), "tar_ci": v.interval(boot[name][f"tar_at_{t!r}"])} for t in TARGETS},
            }
            label = NAMES.get(name, "Fusion")
            curves[label] = (r.far, r.frr)
            markers[label] = [(t, v.tar_at_far(r, t)["far"], v.tar_at_far(r, t)["frr"]) for t in TARGETS]
        for a, b in combinations(list(sets), 2):  # every pair, on the same subject draws (E4)
            result["paired"][f"{b} minus {a}"] = {f"tar_at_{t * 100:g}%": v.interval(boot[b][f"tar_at_{t!r}"] - boot[a][f"tar_at_{t!r}"])
                                                  for t in TARGETS}
        report["scenarios"][sc] = result
        title = {"UxV": "Rolled vs rolled (UxV)", "RxV": "Plain vs rolled (RxV)"}[sc]
        subtitle = f"test subjects: {len(sets['fusion'].genuine):,} genuine and {len(sets['fusion'].impostor):,} impostor pairs"
        (out_dir / f"det-test-{sc}.svg").write_text(det_svg(f"DET with fusion, {title}", subtitle, curves, markers),
                                                     encoding="utf-8", newline="\n")
        print(f"{sc}: weights {result['weights']}")
        for name, x in result["algorithms"].items():
            print(f"  {name:<12} EER {100 * x['eer']['value']:.2f}%  " + "  ".join(
                f"FAR {k}: TAR {100 * y['tar']:.1f}% [{100 * y['tar_ci'][0]:.1f}, {100 * y['tar_ci'][1]:.1f}]" for k, y in x["at_far"].items()))
        for name, d in result["paired"].items():
            print(f"  {name}: " + "  ".join(f"{k} [{100 * lo:+.1f}, {100 * hi:+.1f}]" for k, (lo, hi) in d.items()))
    write_json(out_dir / "report.json", report)


if __name__ == "__main__":
    main()
