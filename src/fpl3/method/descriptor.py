"""Learned pore descriptor (docs/methods/learned-pore-descriptor.md, steps 3-4), on the GPU.

Re-implements Dahia & Pamplona Segundo's description model (third_party/dahia-pipeline:
models/description.py, train.py, utils.py) in PyTorch: six 3 x 3 convolutions with ReLU then batch
normalisation, dropout, an 8 x 8 convolution to a 128-d L2-normalised descriptor, triplet
semi-hard loss (margin 1, squared distances), plain SGD at learning rate 0.1, batches of 256
patches with two views per pore identity, their online augmentation, and early stopping on the
validation EER of whole-print matching with their `basic` score (ratio 0.7).

Runs in the fingerprint-level3-gpu environment:
    python -m fpl3.method.descriptor train
    python -m fpl3.method.descriptor score      # every full-run pair -> runs/method/scores/learned-pore.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..data.build import write_csv
from ..data.sampling import keyed_hash
from ..experiments.common import REPO_ROOT, read_pairs
from .data import DESCRIPTOR_DATA, METHOD_DIR, dataset_root, fingers, protocol_and_images, split_development
from .pores import pore_file

PATCH = 32
MODEL_DIR = METHOD_DIR / "descriptor"
SCORES = METHOD_DIR / "scores" / "learned-pore.csv"
RATIO = 0.7  # validate.matching: "SIFT's original criterion with distance ratio check threshold of 0.7"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class DescriptionNet(nn.Module):
    """models/description.py: conv(3x3, no bias) -> ReLU -> batch norm, six times; dropout; conv 8x8 -> batch norm; L2."""

    def __init__(self, dropout: float | None = 0.3):
        super().__init__()
        layers, channels = [], 1
        for filters, stride in zip((32, 32, 64, 64, 128, 128), (1, 1, 2, 1, 2, 1)):
            layers += [nn.Conv2d(channels, filters, 3, stride, padding=1, bias=False), nn.ReLU(inplace=True), nn.BatchNorm2d(filters)]
            channels = filters
        self.features = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.last = nn.Sequential(nn.Conv2d(128, 128, 8, bias=False), nn.BatchNorm2d(128))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:  # N x 1 x 32 x 32, values in [0, 1]
        return F.normalize(self.last(self.dropout(self.features(patches))).flatten(1), dim=1)


def triplet_semihard_loss(labels: torch.Tensor, embeddings: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """tf.contrib.losses.metric_learning.triplet_semihard_loss: for every anchor-positive pair, the closest
    negative farther than the positive, else the farthest negative; squared distances."""
    d = (2 - 2 * embeddings @ embeddings.T).clamp(min=0)
    same = labels[:, None] == labels[None, :]
    negative = ~same
    farther = negative[:, None, :] & (d[:, None, :] > d[:, :, None])  # anchor, positive, negative
    outside = torch.where(farther, d[:, None, :], torch.full_like(d[:, None, :], math.inf)).amin(-1)
    inside = torch.where(negative, d, torch.full_like(d, -math.inf)).amax(-1, keepdim=True)
    semi_hard = torch.where(farther.any(-1), outside, inside.expand_as(outside))
    positives = same & ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    return ((margin + d - semi_hard).clamp(min=0) * positives).sum() / positives.sum()


def augment(patches: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """utils._transform_mini_batch: contrast N(1, 0.05), brightness N(0, 0.05), translation N(0, 1) px, then
    rotation N(0, 7.5) degrees about the centre; bilinear, zero padding."""
    n = len(patches)
    normal = lambda mean, std: mean + std * torch.randn(n, generator=generator, device=patches.device)
    out = normal(1, 0.05)[:, None, None, None] * patches + normal(0, 0.05)[:, None, None, None]
    dx, dy, theta = normal(0, 1), normal(0, 1), normal(0, 7.5) * math.pi / 180
    centre = PATCH // 2
    cos, sin = torch.cos(theta), torch.sin(theta)
    # forward map: translate by (dx, dy), then rotate about the centre; sample with its inverse
    ys, xs = torch.meshgrid(torch.arange(PATCH, device=patches.device, dtype=torch.float32),
                            torch.arange(PATCH, device=patches.device, dtype=torch.float32), indexing="ij")
    u, v = xs[None] - centre, ys[None] - centre
    src_x = cos[:, None, None] * u - sin[:, None, None] * v + centre - dx[:, None, None]
    src_y = sin[:, None, None] * u + cos[:, None, None] * v + centre - dy[:, None, None]
    grid = torch.stack([(2 * src_x + 1) / PATCH - 1, (2 * src_y + 1) / PATCH - 1], -1)
    return F.grid_sample(out, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def basic_scores(descs1: torch.Tensor, descs2: torch.Tensor, ratio: float = RATIO) -> int:
    """matching.basic / utils.find_correspondences: mutual nearest descriptors whose squared distance passes
    the ratio check in both directions."""
    if len(descs1) == 0 or len(descs2) == 0:
        return 0
    d = (descs1 * descs1).sum(1)[:, None] - 2 * descs1 @ descs2.T + (descs2 * descs2).sum(1)[None, :]
    rows = torch.arange(len(descs1), device=d.device)
    if len(descs1) == 1 or len(descs2) == 1:
        j = d.argmin(1)
        return int((d.argmin(0)[j] == rows).sum())
    v1, i1 = d.topk(2, dim=1, largest=False)
    v2, i2 = d.topk(2, dim=0, largest=False)
    j = i1[:, 0]
    keep = (i2[0, j] == rows) & (v1[:, 0] < ratio * v1[:, 1]) & (v1[:, 0] < ratio * v2[1, j])
    return int(keep.sum())


def image_patches(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """utils.trained_descriptors: img[r - 16 : r + 16, c - 16 : c + 16] at every pore whose window fits."""
    h = PATCH // 2
    ok = (points[:, 0] >= h) & (points[:, 0] < image.shape[0] - h) & (points[:, 1] >= h) & (points[:, 1] < image.shape[1] - h)
    return np.stack([image[r - h : r + h, c - h : c + h] for r, c in points[ok]]) if ok.any() else np.zeros((0, PATCH, PATCH), np.uint8)


@torch.no_grad()
def describe(net: nn.Module, patches_u8: torch.Tensor, batch: int = 16384) -> torch.Tensor:
    net.eval()
    out = [net(patches_u8[i : i + batch].float().div(255).unsqueeze(1)) for i in range(0, len(patches_u8), batch)]
    return torch.cat(out) if out else torch.zeros((0, 128), device=DEVICE)


def load_image_patches(rows: list[dict], root: Path) -> list[torch.Tensor]:
    out = []
    for row in rows:
        image = cv2.imread(str(root / row["relpath"]), cv2.IMREAD_GRAYSCALE)
        with np.load(pore_file(row["image_id"])) as f:
            out.append(torch.from_numpy(image_patches(image, f["points"].astype(int))).to(DEVICE))
    return out


def eer(genuine: np.ndarray, impostor: np.ndarray) -> float:
    from ..eval import verification as v

    s = v.Scores(genuine.astype(float), np.zeros(len(genuine), int), impostor.astype(float),
                 np.zeros(len(impostor), int), np.zeros(len(impostor), int), 1)
    return v.eer(v.rates(s, v.Grid(s)))


def validation_set(protocol: dict, images: list[dict], validation: list[str]):
    """UxV pairs among validation subjects, same finger position (genuine and impostor)."""
    groups = fingers(images, validation, sets="UV")
    rows_u = [views["U"] for key, views in sorted(groups.items()) if "U" in views and "V" in views]
    rows_v = [views["V"] for key, views in sorted(groups.items()) if "U" in views and "V" in views]
    pairs = [(i, j, rows_u[i]["subject"] == rows_v[j]["subject"]) for i in range(len(rows_u)) for j in range(len(rows_v))
             if rows_u[i]["frgp"] == rows_v[j]["frgp"]]
    root = dataset_root(protocol)
    return load_image_patches(rows_u, root), load_image_patches(rows_v, root), pairs


def validation_eer(net, val) -> float:
    patches_u, patches_v, pairs = val
    descs_u, descs_v = [describe(net, p) for p in patches_u], [describe(net, p) for p in patches_v]
    scores = np.array([basic_scores(descs_u[i], descs_v[j]) for i, j, _ in pairs])
    genuine = np.array([g for _, _, g in pairs])
    return eer(scores[genuine], scores[~genuine])


def train(args) -> None:
    protocol, images = protocol_and_images()
    _, validation = split_development(protocol, images)
    seed = int(keyed_hash(protocol["randomness"]["master_seed"], "descriptor-training", "")[:8], 16)
    torch.manual_seed(seed)
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    patches = torch.from_numpy(np.load(DESCRIPTOR_DATA / "train_patches.npy")).to(DEVICE)
    labels = torch.from_numpy(np.load(DESCRIPTOR_DATA / "train_labels.npy")).to(DEVICE)
    order = torch.argsort(labels)
    labels, patches = labels[order], patches[order]
    starts = torch.searchsorted(labels, torch.arange(int(labels.max()) + 1, device=DEVICE))
    counts = torch.bincount(labels)
    identities = len(counts)
    print(f"{len(patches):,} patches, {identities:,} pore identities, device {torch.cuda.get_device_name(0)}", flush=True)
    val = validation_set(protocol, images, validation)
    print(f"validation: {len(val[2]):,} UxV pairs of {len(validation)} subjects", flush=True)

    net = DescriptionNet(args.dropout).to(DEVICE)
    optimiser = torch.optim.SGD(net.parameters(), lr=args.learning_rate)
    per_batch = args.batch_size // 2  # two views per identity (balanced batches)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log, best, faults, start = [], math.inf, 0, time.time()
    permutation, cursor = torch.randperm(identities, generator=generator, device=DEVICE), 0
    for step in range(1, args.steps + 1):
        if cursor + per_batch > identities:
            permutation, cursor = torch.randperm(identities, generator=generator, device=DEVICE), 0
        chosen = permutation[cursor : cursor + per_batch]
        cursor += per_batch
        first = torch.floor(torch.rand(per_batch, generator=generator, device=DEVICE) * counts[chosen]).long()
        second = (first + 1 + torch.floor(torch.rand(per_batch, generator=generator, device=DEVICE) * (counts[chosen] - 1)).long()) % counts[chosen]
        index = torch.cat([starts[chosen] + first, starts[chosen] + second])
        batch = augment(patches[index].float().div(255).unsqueeze(1), generator)
        net.train()
        loss = triplet_semihard_loss(torch.cat([chosen, chosen]), net(batch))
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        if step % 100 == 0:
            log.append({"step": step, "loss": float(loss.detach())})
        if step % args.validate_every == 0:
            value = validation_eer(net, val)
            log.append({"step": step, "validation_eer": value, "seconds": round(time.time() - start)})
            print(f"step {step}: loss {float(loss.detach()):.4f}, validation EER {100 * value:.2f}%, {time.time() - start:.0f}s", flush=True)
            if value < best:
                best, faults = value, 0
                torch.save(net.state_dict(), MODEL_DIR / "model.pt")
            else:
                faults += 1
                if faults >= args.tolerance:
                    print("early stop", flush=True)
                    break
    (MODEL_DIR / "training.json").write_text(json.dumps({"best_validation_eer": best, "args": vars(args), "log": log}, indent=1),
                                            encoding="utf-8")
    print(f"best validation EER {100 * best:.2f}%", flush=True)


def score(args) -> None:
    """Scores for every pair of the full run, in its pair order, so the evaluation and fusion can pair them."""
    protocol, images = protocol_and_images()
    root = dataset_root(protocol)
    net = DescriptionNet(None).to(DEVICE)
    net.load_state_dict(torch.load(MODEL_DIR / "model.pt", map_location=DEVICE))
    with open(REPO_ROOT / "runs" / "full" / "scores" / "pore-sift.csv", newline="", encoding="utf-8") as f:
        pairs = [{k: row[k] for k in ("pair_id", "scenario", "kind", "frgp", "probe_image_id", "reference_image_id", "compact")}
                 for row in csv.DictReader(f)]
    rows = {r["image_id"]: r for r in images}
    needed = sorted({p["probe_image_id"] for p in pairs} | {p["reference_image_id"] for p in pairs})
    start, descs = time.time(), {}
    for k in range(0, len(needed), 64):
        chunk = needed[k : k + 64]
        for image_id, patches in zip(chunk, load_image_patches([rows[i] for i in chunk], root)):
            descs[image_id] = describe(net, patches).half()
        print(f"described {min(k + 64, len(needed))} of {len(needed)} images, {time.time() - start:.0f}s", flush=True)
    out = []
    for n, p in enumerate(pairs):
        s = basic_scores(descs[p["probe_image_id"]].float(), descs[p["reference_image_id"]].float())
        out.append({**p, "status": "ok", "reason": "", "score": s})
        if (n + 1) % 50000 == 0:
            print(f"scored {n + 1:,} of {len(pairs):,} pairs, {time.time() - start:.0f}s", flush=True)
    SCORES.parent.mkdir(parents=True, exist_ok=True)
    write_csv(SCORES, ("pair_id", "scenario", "kind", "frgp", "probe_image_id", "reference_image_id", "compact", "status", "reason", "score"), out)
    print(f"wrote {SCORES} in {time.time() - start:.0f}s", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("train", "score"))
    parser.add_argument("--learning-rate", type=float, default=1e-1)  # train.py defaults
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--tolerance", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.3)  # README: --dropout 0.3
    parser.add_argument("--validate-every", type=int, default=1000)
    args = parser.parse_args(argv)
    {"train": train, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
