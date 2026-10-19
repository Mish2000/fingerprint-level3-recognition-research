"""Label audit of Q and J against V (A5, decision record 2026-10-19), on the GPU.

The learned pore descriptor's second training scores every usable Q and J image against its
candidate V images; the rules are in label_audit. Runs in the fingerprint-level3-gpu environment
once the plain run's Pore SIFT features exist (test Q and J pores):

    python -m fpl3.method.audit      # -> runs/plain/audit/audit.csv and audit.json

New descriptors are cached next to the model, so the plain run's later scoring describes nothing twice.
"""

from __future__ import annotations

import time
from collections import Counter

import torch

from ..data.build import write_csv
from ..experiments.common import REPO_ROOT, write_json
from .data import dataset_root, protocol_and_images
from .descriptor import DEVICE, DescriptionNet, batched_scores, describe_images, run_paths
from .label_audit import CROSS_SHARE, FIELDS, candidate_references, classify, cross_subject, threshold

AUDIT_DIR = REPO_ROOT / "runs" / "plain" / "audit"
MODEL = "repeat"  # the second training, chosen by the validation subjects (2026-10-08)


def main() -> None:
    protocol, images = protocol_and_images()
    usable = [r for r in images if not int(r["excluded"])]
    probes = [r for r in usable if r["set"] in ("Q", "J")]
    references = [r for r in usable if r["set"] == "V"]
    model_dir = run_paths(MODEL)[0]
    net = DescriptionNet(None).to(DEVICE)
    net.load_state_dict(torch.load(model_dir / "model.pt", map_location=DEVICE))
    folder = model_dir / "descriptors"
    new = sum(not (folder / f"{r['image_id']}.npy").exists() for r in probes + references)
    start = time.time()
    descs = describe_images(net, probes + references, dataset_root(protocol), start, folder)
    describe_s = time.time() - start
    torch.cuda.empty_cache()

    scored, comparisons, began = {}, 0, time.time()
    with torch.no_grad():
        for n, row in enumerate(probes, start=1):
            refs = candidate_references(row, references)
            scores = batched_scores(descs[row["image_id"]].float(), [descs[v["image_id"]].float() for v in refs])
            scored[row["image_id"]] = list(zip(refs, scores))
            comparisons += len(refs)
            if n % 200 == 0:
                print(f"audited {n} of {len(probes)} images, {comparisons / (time.time() - began):.0f} comparisons/s", flush=True)
    compare_s = time.time() - began

    cross = [s for row in probes for s in cross_subject(row, scored[row["image_id"]])]
    t = threshold(cross)
    results = [classify(row, scored[row["image_id"]], t) for row in probes]
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(AUDIT_DIR / "audit.csv", FIELDS, results)
    summary = {
        "model": MODEL,
        "rule": "label_audit (A5)",
        "threshold": t,
        "cross_share": CROSS_SHARE,
        "cross_subject_comparisons": len(cross),
        "cross_subject_above_threshold": sum(1 for s in cross if s > t),
        "images": dict(sorted(Counter(r["set"] for r in probes).items())),
        "categories": {s: dict(sorted(Counter(r["category"] for r in results if r["set"] == s).items())) for s in ("Q", "J")},
        "comparisons": comparisons,
        "timing": {"images_described": new, "describe_s": round(describe_s, 1), "compare_s": round(compare_s, 1),
                   "comparisons_per_s": round(comparisons / compare_s, 1), "device": torch.cuda.get_device_name(0)},
    }
    write_json(AUDIT_DIR / "audit.json", summary)
    print(summary["categories"], f"T = {t}", flush=True)


if __name__ == "__main__":
    main()
