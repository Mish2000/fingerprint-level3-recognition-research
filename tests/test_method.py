import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from fpl3.method import annotate  # noqa: E402


def test_moving_least_squares_reproduces_an_affine_map():
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 1000, (60, 2))
    a = np.array([[0.98, -0.12, 30.0], [0.11, 1.02, -12.0]])  # x/y affine
    dst = (src[:, ::-1] @ a[:, :2].T + a[:, 2])[:, ::-1]
    queries = rng.uniform(0, 1000, (20, 2))
    expected = (queries[:, ::-1] @ a[:, :2].T + a[:, 2])[:, ::-1]
    assert np.allclose(annotate.mls_affine(src, dst, queries, a, 120.0), expected, atol=1e-3)


def test_block_matching_finds_a_known_shift():
    rng = np.random.default_rng(1)
    reference = cv2.GaussianBlur(rng.integers(0, 256, (200, 200)).astype(np.uint8), (0, 0), 2)
    other = np.roll(reference, (3, -2), axis=(0, 1))
    centres = np.array([[100.0, 100.0], [80.0, 120.0]])
    kept, found = annotate.block_match(reference, other, centres, centres, 32, 5, 0.6)
    assert list(kept) == [0, 1] and np.array_equal(found, centres + [3, -2])


torch = pytest.importorskip("torch")
from fpl3.method import descriptor  # noqa: E402


def reference_semihard(labels, emb, margin=1.0):
    d = np.clip(2 - 2 * emb @ emb.T, 0, None)
    total, count = 0.0, 0
    for a in range(len(labels)):
        negatives = [n for n in range(len(labels)) if labels[n] != labels[a]]
        for p in range(len(labels)):
            if p == a or labels[p] != labels[a]:
                continue
            farther = [d[a, n] for n in negatives if d[a, n] > d[a, p]]
            neg = min(farther) if farther else max(d[a, n] for n in negatives)
            total += max(margin + d[a, p] - neg, 0.0)
            count += 1
    return total / count


def test_triplet_semihard_loss_matches_the_tensorflow_definition():
    rng = np.random.default_rng(2)
    emb = rng.normal(size=(12, 8))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    labels = np.repeat(np.arange(6), 2)
    got = descriptor.triplet_semihard_loss(torch.tensor(labels), torch.tensor(emb, dtype=torch.float64))
    assert float(got) == pytest.approx(reference_semihard(labels, emb), rel=1e-9)


def test_gpu_basic_score_matches_dahias_correspondences():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(300, 128)).astype(np.float32), rng.normal(size=(250, 128)).astype(np.float32)
    b[:100] = a[:100] + rng.normal(scale=0.3, size=(100, 128)).astype(np.float32)
    rows, _ = annotate_mutual_ratio(a, b, 0.7)
    assert descriptor.basic_scores(torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)) == len(rows)


def test_grouped_scores_equal_pair_by_pair_scores():
    rng = np.random.default_rng(4)
    probe = torch.tensor(rng.normal(size=(150, 16)))
    references = [torch.tensor(rng.normal(size=(m, 16))) for m in (90, 0, 1, 200, 2, 140)]
    references[3][:60] = probe[:60] + 0.2 * torch.tensor(rng.normal(size=(60, 16)))
    expected = [descriptor.basic_scores(probe, r) for r in references]
    assert descriptor.batched_scores(probe, references, budget=150 * 200 * 2) == expected
    assert descriptor.batched_scores(probe[:1], references) == [descriptor.basic_scores(probe[:1], r) for r in references]


def annotate_mutual_ratio(d1, d2, thr):
    """Dahia's find_correspondences with a ratio check, written out as in utils.py."""
    dist = (d1 * d1).sum(1)[:, None] - 2 * d1.astype(np.float64) @ d2.T + (d2 * d2).sum(1)[None, :]
    corrs2 = np.argsort(dist.T, axis=1)[:, :2]
    corrs1 = np.argsort(dist, axis=1)[:, :2]
    pairs = []
    for i, (j, _) in enumerate(corrs2):
        if corrs1[j, 0] == i and dist[j, i] < dist[corrs2[i, 1], i] * thr and dist[j, i] < dist[j, corrs1[j, 1]] * thr:
            pairs.append((j, i))
    return pairs, None


def test_augmentation_keeps_the_patch_and_is_reproducible():
    rows = torch.arange(32, dtype=torch.float32)[:, None].expand(32, 32)
    patches = (0.5 + 0.4 * torch.sin(2 * torch.pi * rows / 18)).expand(4, 1, 32, 32).contiguous()  # ridge-like
    first = descriptor.augment(patches, torch.Generator().manual_seed(0))
    second = descriptor.augment(patches, torch.Generator().manual_seed(0))
    assert first.shape == patches.shape and torch.equal(first, second)
    assert torch.corrcoef(torch.stack([first.flatten(), patches.flatten()]))[0, 1] > 0.5
