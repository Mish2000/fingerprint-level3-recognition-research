import numpy as np
import pytest

pytest.importorskip("cv2")
skimage_data = pytest.importorskip("skimage.data")

from fpl3.algorithms import skimage_harris  # noqa: E402


def crop(row: int, col: int) -> np.ndarray:
    return skimage_data.camera()[row : row + 160, col : col + 160]


def test_documented_example_values():
    assert (skimage_harris.THRESHOLD_REL, skimage_harris.MIN_DISTANCE, skimage_harris.SUBPIX_WINDOW) == (0.001, 5, 9)
    assert (skimage_harris.WINDOW_EXT, skimage_harris.WEIGHT_SIGMA) == (5, 3)
    assert (skimage_harris.MIN_SAMPLES, skimage_harris.RESIDUAL_THRESHOLD, skimage_harris.MAX_TRIALS) == (3, 2, 100)


def test_every_probe_window_is_complete_and_located():
    features = skimage_harris.extract_image(crop(100, 100))
    assert len(features["points"]) > 0
    assert features["windows"].shape == (len(features["points"]), 11, 11)
    assert features["corner_windows"].shape == (len(features["corners"]), 11, 11)
    assert not np.isnan(features["points"]).any()


def test_seeded_comparison_is_reproducible_and_separates_same_from_different():
    probe = skimage_harris.extract_image(crop(100, 100))
    same, other = skimage_harris.extract_image(crop(103, 102)), skimage_harris.extract_image(crop(300, 250))
    seed = skimage_harris.rng_seed("test-seed")
    first = skimage_harris.compare(probe, same, seed)
    assert first == skimage_harris.compare(probe, same, seed)
    assert first["status"] == "ok" and first["score"] > skimage_harris.compare(probe, other, seed)["score"]


def test_no_corners_is_a_failure():
    flat = skimage_harris.extract_image(np.full((60, 60), 128, np.uint8))
    assert skimage_harris.compare(flat, skimage_harris.extract_image(crop(100, 100)))["status"] == "failure"
