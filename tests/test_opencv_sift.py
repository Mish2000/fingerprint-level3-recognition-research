import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from fpl3.algorithms import opencv_sift  # noqa: E402


def textured_image(seed: int, size: int = 300) -> np.ndarray:
    noise = np.random.default_rng(seed).integers(0, 256, (size, size)).astype(np.uint8)
    return cv2.GaussianBlur(noise, (0, 0), 2)


def features(image: np.ndarray):
    keypoints, descriptors = cv2.SIFT_create().detectAndCompute(image, None)
    return np.float32([k.pt for k in keypoints]).reshape(-1, 2), descriptors


def test_documented_tutorial_values():
    assert (opencv_sift.FLANN_INDEX_KDTREE, opencv_sift.FLANN_TREES, opencv_sift.FLANN_CHECKS) == (1, 5, 50)
    assert (opencv_sift.RATIO, opencv_sift.MIN_MATCH_COUNT, opencv_sift.RANSAC_REPROJ_THRESHOLD) == (0.7, 10, 5.0)


def test_seeded_comparison_is_reproducible_and_separates_same_from_different():
    image = textured_image(0)
    shifted = np.roll(image, (7, 5), axis=(0, 1))
    other = textured_image(1)
    a, b, c = features(image), features(shifted), features(other)
    seed = opencv_sift.rng_seed("test-seed")
    first = opencv_sift.compare(a, b, seed)
    assert first == opencv_sift.compare(a, b, seed)
    assert first["status"] == "ok" and first["score"] > opencv_sift.compare(a, c, seed)["score"]


def test_too_few_features_is_a_failure():
    empty = (np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32))
    one = (np.zeros((1, 2), np.float32), np.ones((1, 128), np.float32))
    assert opencv_sift.compare(empty, one)["status"] == "failure"
    assert opencv_sift.compare(one, one)["status"] == "failure"
