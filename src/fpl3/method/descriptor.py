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
    python -m fpl3.method.descriptor train --name repeat   # another seed: runs/method/repeat, learned-pore-repeat.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby
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


def run_paths(name: str) -> tuple[Path, Path, str]:
    """Model folder, score file and training-seed purpose (R5) of a training run; 'descriptor' is the first run."""
    if name == "descriptor":
        return METHOD_DIR / "descriptor", METHOD_DIR / "scores" / "learned-pore.csv", "descriptor-training"
    return METHOD_DIR / name, METHOD_DIR / "scores" / f"learned-pore-{name}.csv", f"descriptor-training:{name}"

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


def basic_score(descs1: torch.Tensor, descs2: torch.Tensor, ratio: float = RATIO) -> torch.Tensor:
    """matching.basic / utils.find_correspondences: mutual nearest descriptors whose squared distance passes
    the ratio check in both directions. Returns a 0-d tensor, so many pairs can be queued before one sync."""
    if len(descs1) == 0 or len(descs2) == 0:
        return torch.zeros((), dtype=torch.long, device=descs1.device)
    d = (descs1 * descs1).sum(1)[:, None] - 2 * descs1 @ descs2.T + (descs2 * descs2).sum(1)[None, :]
    rows = torch.arange(len(descs1), device=d.device)
    if len(descs1) == 1 or len(descs2) == 1:
        j = d.argmin(1)
        return (d.argmin(0)[j] == rows).sum()
    v1, i1 = d.topk(2, dim=1, largest=False)
    v2, i2 = d.topk(2, dim=0, largest=False)
    j = i1[:, 0]
    return ((i2[0, j] == rows) & (v1[:, 0] < ratio * v1[:, 1]) & (v1[:, 0] < ratio * v2[1, j])).sum()


def basic_scores(descs1: torch.Tensor, descs2: torch.Tensor, ratio: float = RATIO) -> int:
    return int(basic_score(descs1, descs2, ratio))


def many_scores(pairs: list[tuple[torch.Tensor, torch.Tensor]], flush: int = 2000) -> list[int]:
    """basic_score for many pairs, syncing with the GPU once per `flush` pairs."""
    out, queued = [], []
    for d1, d2 in pairs:
        queued.append(basic_score(d1.float(), d2.float()))
        if len(queued) == flush:
            out += torch.stack(queued).tolist()
            queued = []
    return out + (torch.stack(queued).tolist() if queued else [])


def image_patches(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """utils.trained_descriptors: img[r - 16 : r + 16, c - 16 : c + 16] at every pore whose window fits."""
    h = PATCH // 2
    ok = (points[:, 0] >= h) & (points[:, 0] < image.shape[0] - h) & (points[:, 1] >= h) & (points[:, 1] < image.shape[1] - h)
    windows = np.lib.stride_tricks.sliding_window_view(image, (PATCH, PATCH))
    return np.ascontiguousarray(windows[points[ok, 0] - h, points[ok, 1] - h])


@torch.no_grad()
def describe(net: nn.Module, patches_u8: torch.Tensor, batch: int = 2048) -> torch.Tensor:
    """Batches stay small: 16,384 patches at once filled the 16 GB card and the driver spilled into RAM."""
    net.eval()
    out = [net(patches_u8[i : i + batch].float().div(255).unsqueeze(1)) for i in range(0, len(patches_u8), batch)]
    return torch.cat(out) if out else torch.zeros((0, 128), device=DEVICE)


def read_patches(row: dict, root: Path) -> np.ndarray:
    image = cv2.imread(str(root / row["relpath"]), cv2.IMREAD_GRAYSCALE)
    with np.load(pore_file(row["image_id"])) as f:
        return image_patches(image, f["points"].astype(int))


def load_image_patches(rows: list[dict], root: Path) -> list[torch.Tensor]:
    with ThreadPoolExecutor(8) as pool:  # image decoding releases the GIL, so the GPU is not left waiting
        return [torch.from_numpy(p).to(DEVICE) for p in pool.map(lambda row: read_patches(row, root), rows)]


def batched_scores(probe: torch.Tensor, references: list[torch.Tensor], ratio: float = RATIO,
                   budget: int = 120_000_000) -> list[int]:
    """basic_score of one probe against many references, a group at a time: the references are padded
    into one tensor and every distance matrix of the group comes from one batched matrix product."""
    scores = [0] * len(references)
    n = len(probe)
    usual = [k for k, r in enumerate(references) if len(r) >= 2]
    for k, r in enumerate(references):
        if n and 0 < len(r) and (n == 1 or len(r) == 1):
            scores[k] = int(basic_score(probe, r, ratio))
    if n < 2 or not usual:
        return scores
    p2 = (probe * probe).sum(1)
    rows = torch.arange(n, device=probe.device)
    start = 0
    while start < len(usual):
        group, widest = [usual[start]], len(references[usual[start]])
        start += 1
        while start < len(usual):
            wider = max(widest, len(references[usual[start]]))
            if (len(group) + 1) * n * wider > budget:
                break
            group.append(usual[start])
            widest, start = wider, start + 1
        padded = torch.zeros((len(group), widest, probe.shape[1]), dtype=probe.dtype, device=probe.device)
        valid = torch.zeros((len(group), widest), dtype=torch.bool, device=probe.device)
        for g, k in enumerate(group):
            padded[g, : len(references[k])] = references[k]
            valid[g, : len(references[k])] = True
        d = torch.matmul(probe.unsqueeze(0), padded.transpose(1, 2))  # group x probe x reference, contiguous
        d.mul_(-2).add_(p2[None, :, None]).add_((padded * padded).sum(-1)[:, None, :])  # in place: one matrix in memory
        d.masked_fill_(~valid[:, None, :], math.inf)
        # nearest and second nearest by min reductions (faster than topk): rows first, then columns
        row_best, j = d.min(dim=2)  # each probe descriptor's nearest reference descriptor
        d.scatter_(2, j.unsqueeze(2), math.inf)
        row_second = d.min(dim=2).values
        d.scatter_(2, j.unsqueeze(2), row_best.unsqueeze(2))  # restore
        best_row = d.min(dim=1).indices  # each reference descriptor's nearest probe descriptor
        d.scatter_(1, best_row.unsqueeze(1), math.inf)
        column_second = d.min(dim=1).values
        keep = ((torch.gather(best_row, 1, j) == rows[None, :]) & (row_best < ratio * row_second)
                & (row_best < ratio * torch.gather(column_second, 1, j)))
        for g, s in zip(group, keep.sum(1).tolist()):
            scores[g] = s
        del d, row_best, j, row_second, best_row, column_second, keep
    return scores


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
    with torch.no_grad():
        descs_u, descs_v = [describe(net, p) for p in patches_u], [describe(net, p) for p in patches_v]
        scores = np.array(many_scores([(descs_u[i], descs_v[j]) for i, j, _ in pairs]))
    del descs_u, descs_v
    torch.cuda.empty_cache()
    genuine = np.array([g for _, _, g in pairs])
    return eer(scores[genuine], scores[~genuine])


def train(args) -> None:
    protocol, images = protocol_and_images()
    _, validation = split_development(protocol, images)
    model_dir, _, purpose = run_paths(args.name)
    seed = int(keyed_hash(protocol["randomness"]["master_seed"], purpose, "")[:8], 16)
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
    model_dir.mkdir(parents=True, exist_ok=True)
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
                torch.save(net.state_dict(), model_dir / "model.pt")
            else:
                faults += 1
                if faults >= args.tolerance:
                    print("early stop", flush=True)
                    break
    (model_dir / "training.json").write_text(json.dumps({"best_validation_eer": best, "args": vars(args), "log": log}, indent=1),
                                            encoding="utf-8")
    print(f"best validation EER {100 * best:.2f}%", flush=True)


def describe_images(net: nn.Module, rows: list[dict], root: Path, start: float, folder: Path) -> dict[str, torch.Tensor]:
    """Descriptors of every image, one float16 .npy each in `folder`, so scoring never describes twice."""
    folder.mkdir(parents=True, exist_ok=True)
    out, missing = {}, [r for r in rows if not (folder / f"{r['image_id']}.npy").exists()]
    for k in range(0, len(missing), 128):
        chunk = missing[k : k + 128]
        for row, patches in zip(chunk, load_image_patches(chunk, root)):
            np.save(folder / f"{row['image_id']}.npy", describe(net, patches).half().cpu().numpy())
        print(f"described {min(k + 128, len(missing))} of {len(missing)} images, {time.time() - start:.0f}s", flush=True)
    for row in rows:
        out[row["image_id"]] = torch.from_numpy(np.load(folder / f"{row['image_id']}.npy")).to(DEVICE)
    return out


def score(args) -> None:
    """Scores for every pair of the full run, in its pair order, so the evaluation and fusion can pair them.
    Pairs are grouped by probe; each probe is compared with its references a group at a time."""
    protocol, images = protocol_and_images()
    root = dataset_root(protocol)
    net = DescriptionNet(None).to(DEVICE)
    model_dir, scores_path, _ = run_paths(args.name)
    net.load_state_dict(torch.load(model_dir / "model.pt", map_location=DEVICE))
    with open(REPO_ROOT / "runs" / "full" / "scores" / "pore-sift.csv", newline="", encoding="utf-8") as f:
        pairs = [{k: row[k] for k in ("pair_id", "scenario", "kind", "frgp", "probe_image_id", "reference_image_id", "compact")}
                 for row in csv.DictReader(f)]
    rows = {r["image_id"]: r for r in images}
    needed = sorted({p["probe_image_id"] for p in pairs} | {p["reference_image_id"] for p in pairs})
    start = time.time()
    descs = describe_images(net, [rows[i] for i in needed], root, start, model_dir / "descriptors")
    torch.cuda.empty_cache()
    out, began, done = [], time.time(), 0
    with torch.no_grad():
        for n, (probe, group) in enumerate(groupby(pairs, key=lambda p: p["probe_image_id"]), start=1):
            group = list(group)
            scores = batched_scores(descs[probe].float(), [descs[p["reference_image_id"]].float() for p in group])
            out += [{**p, "status": "ok", "reason": "", "score": s} for p, s in zip(group, scores)]
            done += len(group)
            if n % 100 == 0:
                rate = done / (time.time() - began)
                finish = time.localtime(time.time() + (len(pairs) - done) / rate)
                print(f"scored {done:,} of {len(pairs):,} pairs, {rate:.0f} pairs/s, expected finish {time.strftime('%H:%M', finish)}",
                      flush=True)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(scores_path, ("pair_id", "scenario", "kind", "frgp", "probe_image_id", "reference_image_id", "compact", "status", "reason", "score"), out)
    print(f"wrote {scores_path} in {time.time() - start:.0f}s", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("train", "score"))
    parser.add_argument("--name", default="descriptor", help="training run: model folder, score file and seed purpose")
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
