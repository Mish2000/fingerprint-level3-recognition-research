"""Pore SIFT worker (docs/algorithms/pore-sift.md).

Runs as a script inside the separate fingerprint-level3-pore-sift environment
(Python 3.10, torch 1.13.0 CPU, opencv-contrib-python 3.4.18.65); it does not
import the fpl3 package. Upstream code is imported unchanged:

- detector: Fingerprint Pore Detection survey (util.utils.loadModel, entireImage.apply_nms)
- descriptors and matching: Dahia & Pamplona Segundo (utils.sift_descriptors, matching.basic)

    python pore_sift_worker.py extract --job job.json
    python pore_sift_worker.py compare --job job.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import types

import numpy as np

# Survey detector: scripts/detect.sh (--features 40) and out_of_the_box_detect.py.
FEATURES = 40
NUMBER_LAYERS = 8
WINDOW = 17
NMS_PROBABILITY = 0.65
NMS_BOX = 17
NMS_IOU = 0.2
NMS_WINDOW = 17
# Dahia & Pamplona Segundo: ratio threshold of every documented command; matching mode 'basic' is the code default.
RATIO = 0.7


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def import_upstream(survey_dir: str, dahia_dir: str, threads: int):
    # Dahia's utils.py imports TensorFlow at module level; none of the functions used here touch it,
    # and TensorFlow 1.10 cannot be installed for this Python. An empty module satisfies the import.
    sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))
    sys.path[:0] = [survey_dir, dahia_dir]
    import cv2
    import torch
    import torchvision

    torch.set_num_threads(threads)
    cv2.setNumThreads(threads)
    import entireImage  # survey
    import matching  # Dahia & Pamplona Segundo
    import utils as dahia_utils  # Dahia & Pamplona Segundo
    from util.utils import loadModel  # survey

    return types.SimpleNamespace(
        cv2=cv2, torch=torch, torchvision=torchvision, entireImage=entireImage,
        matching=matching, dahia_utils=dahia_utils, loadModel=loadModel,
    )


def load_detector(up, survey_dir: str, expected_sha256: str):
    model_path = os.path.join(survey_dir, "out_of_the_box_detect", "models", str(FEATURES))
    actual = sha256(model_path)
    if actual != expected_sha256:
        raise SystemExit(f"model checksum mismatch: {actual}")
    # The file must be a plain state dict before upstream's loadModel unpickles it.
    state = up.torch.load(model_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise SystemExit("model file is not a plain state dict")
    model = up.loadModel(
        modelPath=model_path, device=up.torch.device("cpu"), NUMBERLAYERS=NUMBER_LAYERS, NUMBERFEATURES=FEATURES,
        MAXPOOLING=False, WINDOWSIZE=WINDOW, residual=False, gabriel=False, su=False,
    )
    model.eval()
    return model


def read_coordinates(path: str) -> list[tuple[int, int]]:
    points = []
    with open(path) as f:
        for line in f:
            if line.strip():
                row, col = (int(v) for v in line.split(","))
                points.append((row, col))
    return points


def extract(job: dict) -> None:
    up = import_upstream(job["survey_dir"], job["dahia_dir"], job.get("threads", 1))
    model = load_detector(up, job["survey_dir"], job["model_sha256"])
    to_tensor = up.torchvision.transforms.Compose([up.torchvision.transforms.ToTensor()])
    scratch = tempfile.mkdtemp(prefix="pore_sift_")
    pore_dir = os.path.join(scratch, "pore") + os.sep
    coord_dir = os.path.join(scratch, "coord") + os.sep
    os.makedirs(pore_dir)
    os.makedirs(coord_dir)
    timings = []
    try:
        for index, image in enumerate(job["images"], start=1):
            t0 = time.perf_counter()
            pixels = up.cv2.imread(image["path"], up.cv2.IMREAD_GRAYSCALE)
            # As out_of_the_box_detect.inference; no_grad only avoids storing autograd buffers.
            with up.torch.no_grad():
                pred = model(to_tensor(pixels).unsqueeze(dim=0).float()).detach().cpu()
            t1 = time.perf_counter()
            up.entireImage.apply_nms(pred, NMS_PROBABILITY, NMS_BOX, NMS_IOU, pore_dir, index, coord_dir, NMS_WINDOW)
            points = read_coordinates(os.path.join(coord_dir, f"{index}.txt"))
            t2 = time.perf_counter()
            # OpenCV 3.4.18 rejects integer coordinates in KeyPoint.convert (the pinned 3.4.0 accepted them);
            # float32 holds the same whole-number values exactly.
            descriptors = up.dahia_utils.sift_descriptors(
                up.dahia_utils.load_image(image["path"]), np.asarray(points, np.float32).reshape(-1, 2)
            )
            t3 = time.perf_counter()
            np.savez(
                os.path.join(job["features_dir"], f"{image['image_id']}.npz"),
                points=np.asarray(points, np.int32).reshape(-1, 2),
                descriptors=np.asarray(descriptors, np.float32).reshape(-1, 128),
            )
            for name in (f"{index}.png", f"{index}.txt"):
                for folder in (pore_dir, coord_dir):
                    if os.path.exists(folder + name):
                        os.remove(folder + name)
            timings.append({
                "image_id": image["image_id"], "detect_s": t1 - t0, "nms_s": t2 - t1, "describe_s": t3 - t2,
                "extract_s": t3 - t0, "pores": len(points),
                "bytes": os.path.getsize(os.path.join(job["features_dir"], f"{image['image_id']}.npz")),
            })
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    with open(job["out_path"], "w") as f:
        json.dump(timings, f)


def compare(job: dict) -> None:
    up = import_upstream(job["survey_dir"], job["dahia_dir"], job.get("threads", 1))
    cache: dict[str, tuple] = {}

    def load(image_id: str):
        if image_id not in cache:
            if len(cache) >= 8:
                cache.pop(next(iter(cache)))
            with np.load(os.path.join(job["features_dir"], f"{image_id}.npz")) as data:
                cache[image_id] = (data["points"], data["descriptors"])
        return cache[image_id]

    results = []
    for pair in job["pairs"]:
        t0 = time.perf_counter()
        points1, descriptors1 = load(pair["probe_image_id"])
        points2, descriptors2 = load(pair["reference_image_id"])
        t1 = time.perf_counter()
        score = up.matching.basic(descriptors1, descriptors2, points1, points2, thr=RATIO)
        t2 = time.perf_counter()
        results.append({"pair_id": pair["pair_id"], "status": "ok", "reason": "", "score": int(score),
                        "load_s": t1 - t0, "compare_s": t2 - t1})
    with open(job["out_path"], "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("extract", "compare"))
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    with open(args.job) as f:
        job_spec = json.load(f)
    {"extract": extract, "compare": compare}[args.command](job_spec)
