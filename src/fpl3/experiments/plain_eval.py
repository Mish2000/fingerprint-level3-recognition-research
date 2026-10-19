"""Evaluation of the plain vs plain run (decision record 2026-10-19): fusion, operational thresholds and
TAR / FAR / FRR on the test pairs, per scenario (R6).

    python -m fpl3.experiments.plain_eval [--replicates 1000]

Run after fpl3.experiments.plain_run and `fpl3.method.descriptor score --name repeat --pairs plain`.
Channels: OpenCV SIFT and Pore SIFT (runs/plain/scores) and the learned pore descriptor, second
training. Fusions as in fpl3.method.fusion: linear logistic on log(1 + score), equal class weights,
per scenario. Fusion weights and operational thresholds come from development pairs (R7): all of
them for OpenCV SIFT, Pore SIFT and their fusion; only the validation subjects' pairs for anything
that includes the learned descriptor, which was trained on the other development subjects. Test
pairs give TAR at FAR from their own scores with subject-bootstrap intervals (E2, E4), and the
operational check of the development thresholds (E3). Writes runs/plain/eval/report.json and
det-test-<scenario>.svg.
"""

from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np

from ..data.sampling import keyed_hash
from ..data.samples import development_subjects
from ..eval import verification as v
from ..eval.det import det_svg
from ..method.data import split_development
from ..method.fusion import channel_features, fit_logistic
from .common import REPO_ROOT, load_toml, read_images, write_json
from .evaluate import TARGETS, point, read_scores, score_set

RUN = REPO_ROOT / "runs" / "plain"
CHANNELS = {
    "opencv-sift": RUN / "scores" / "opencv-sift.csv",
    "pore-sift": RUN / "scores" / "pore-sift.csv",
    "learned-pore-repeat": REPO_ROOT / "runs" / "method" / "scores" / "learned-pore-repeat-plain.csv",
}
LEARNED = "learned-pore-repeat"
METHODS = {
    "opencv-sift": ("opencv-sift",),
    "pore-sift": ("pore-sift",),
    "learned-pore-repeat": ("learned-pore-repeat",),
    "fusion-existing": ("opencv-sift", "pore-sift"),
    "fusion-learned-opencv": ("learned-pore-repeat", "opencv-sift"),
    "fusion-learned-pore": ("learned-pore-repeat", "pore-sift"),
    "fusion-all": ("opencv-sift", "pore-sift", "learned-pore-repeat"),
}
LABELS = {
    "opencv-sift": "OpenCV SIFT", "pore-sift": "Pore SIFT", "learned-pore-repeat": "Learned pore descriptor",
    "fusion-existing": "OpenCV SIFT + Pore SIFT", "fusion-learned-opencv": "Learned + OpenCV SIFT",
    "fusion-learned-pore": "Learned + Pore SIFT", "fusion-all": "All three",
}
SCENARIOS = {"RxJ": "Plain vs plain, R probes (RxJ)", "QxJ": "Plain vs plain, Q probes (QxJ)"}


def operational(fit: v.Scores, test: v.Scores, target: float) -> dict:
    """The development threshold at `target` FAR (E2 rule), applied to the test pairs (E3)."""
    threshold = float(v.tar_at_far(v.rates(fit, v.Grid(fit)), target)["accept_if_score_above"])
    true_accepts, false_accepts = int((test.genuine > threshold).sum()), int((test.impostor > threshold).sum())
    return {"accept_if_score_above": threshold, "true_accepts": true_accepts, "genuine": len(test.genuine),
            "false_accepts": false_accepts, "impostor": len(test.impostor), "tar": true_accepts / len(test.genuine),
            "frr": 1 - true_accepts / len(test.genuine), "far": false_accepts / len(test.impostor)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replicates", type=int, default=1000)
    args = parser.parse_args(argv)

    protocol, images = load_toml("protocol.toml"), read_images()
    master_seed = protocol["randomness"]["master_seed"]
    development, _ = development_subjects(images, master_seed, protocol["split"]["dev_fraction"])
    validation = set(split_development(protocol, images)[1])
    data = {c: read_scores(path) for c, path in CHANNELS.items()}
    pairs = data["opencv-sift"][0]
    if any(data[c][0] != pairs for c in CHANNELS):
        raise SystemExit("the channels' score files do not list identical pairs")
    scores = {c: data[c][1] for c in CHANNELS}
    scenario = np.array([p[1] for p in pairs])
    genuine = np.array([p[2] == "genuine" for p in pairs])
    both = lambda group: np.array([p[3] in group and p[4] in group for p in pairs])  # noqa: E731
    halves = {"development": both(development), "validation": both(validation),
              "test": np.array([p[3] not in development and p[4] not in development for p in pairs])}
    index = {s: i for i, s in enumerate(sorted({s for p in pairs for s in p[3:5]}))}

    out_dir = RUN / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"replicates": args.replicates, "confidence": 0.95, "targets": list(TARGETS), "channels": {c: str(p.relative_to(REPO_ROOT))
              for c, p in CHANNELS.items()}, "fit": "development pairs; validation pairs for anything with the learned descriptor (R7)",
              "scenarios": {}}
    for sc, title in SCENARIOS.items():
        result = {"methods": {}, "paired": {}}
        test_rows = np.nonzero(halves["test"] & (scenario == sc))[0]
        method_scores, fit_rows_of = {}, {}
        for name, channels in METHODS.items():
            fit_name = "validation" if LEARNED in channels else "development"
            fit_rows = np.nonzero(halves[fit_name] & (scenario == sc))[0]
            entry = {"fit": fit_name, "fit_pairs": {"genuine": int(genuine[fit_rows].sum()), "impostor": int((~genuine[fit_rows]).sum())}}
            if len(channels) == 1:
                fused = scores[channels[0]]
            else:
                chosen = {c: scores[c] for c in channels}
                floors = {c: float(np.min(s[fit_rows][np.isfinite(s[fit_rows])])) for c, s in chosen.items()}
                w = fit_logistic(channel_features(chosen, fit_rows, floors), genuine[fit_rows].astype(float))
                fused = np.full(len(pairs), np.nan)
                rows = np.concatenate([fit_rows, test_rows])
                fused[rows] = channel_features(chosen, rows, floors) @ w[:-1]
                entry["weights"] = {c: float(x) for c, x in zip(channels, w[:-1])}
            method_scores[name], fit_rows_of[name] = fused, fit_rows
            result["methods"][name] = entry
        sets = {name: score_set(pairs, s, test_rows, index) for name, s in method_scores.items()}
        boot = v.subject_bootstrap(sets, TARGETS, args.replicates, int(keyed_hash(master_seed, "bootstrap:plain-test", "")[:16], 16))
        curves, markers = {}, {}
        for name, s in sets.items():
            r = v.rates(s, v.Grid(s))
            fit = score_set(pairs, method_scores[name], fit_rows_of[name], index)
            result["methods"][name].update({
                "eer": {"value": v.eer(r), "ci": v.interval(boot[name]["eer"])},
                "at_far": {f"{t * 100:g}%": {**point(r, t), "tar_ci": v.interval(boot[name][f"tar_at_{t!r}"])} for t in TARGETS},
                "operational": {f"{t * 100:g}%": operational(fit, s, t) for t in TARGETS},
            })
            curves[LABELS[name]] = (r.far, r.frr)
            markers[LABELS[name]] = [(t, v.tar_at_far(r, t)["far"], v.tar_at_far(r, t)["frr"]) for t in TARGETS]
        for a, b in combinations(list(sets), 2):  # every pair, on the same subject draws (E4)
            result["paired"][f"{b} minus {a}"] = {f"tar_at_{t * 100:g}%": v.interval(boot[b][f"tar_at_{t!r}"] - boot[a][f"tar_at_{t!r}"])
                                                  for t in TARGETS}
        result["genuine"], result["impostor"] = len(sets["opencv-sift"].genuine), len(sets["opencv-sift"].impostor)
        report["scenarios"][sc] = result
        subtitle = f"test subjects: {result['genuine']:,} genuine and {result['impostor']:,} impostor pairs"
        (out_dir / f"det-test-{sc}.svg").write_text(det_svg(f"DET, {title}", subtitle, curves, markers), encoding="utf-8", newline="\n")
        print(f"{sc}: {subtitle}")
        for name, x in result["methods"].items():
            cells = "  ".join(f"FAR {k}: TAR {100 * y['tar']:.1f}% ({y['true_accepts']}/{y['genuine']}) "
                              f"[{100 * y['tar_ci'][0]:.1f}, {100 * y['tar_ci'][1]:.1f}]" for k, y in x["at_far"].items())
            print(f"  {LABELS[name]:<26} {cells}")
    write_json(out_dir / "report.json", report)


if __name__ == "__main__":
    main()
