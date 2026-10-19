"""Tests for the feature extraction in src/baseline_hog_svm.py (no image files needed to run the tests)."""

import numpy as np

from src.baseline_hog_svm import COLOR_BINS, extract_features


def _fake_image(h=200, w=300, seed=0):
    #Use numpy's 'random number generator'
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_hog_length_for_128px():
    # 7x7 blocks x 4 cells x 9 bins = 1764 values
    assert extract_features(_fake_image(), size=128).shape == (1764,)


def test_same_image_gives_same_features():
    img = _fake_image()
    assert np.array_equal(extract_features(img), extract_features(img))


def test_image_size_does_not_change_feature_length():
    small = extract_features(_fake_image(100, 150))
    large = extract_features(_fake_image(900, 1200))
    assert small.shape == large.shape


def test_color_histogram_adds_features_and_sums_to_one():
    base = extract_features(_fake_image(), use_color=False)
    with_color = extract_features(_fake_image(), use_color=True)
    n_color = int(np.prod(COLOR_BINS))
    assert with_color.shape[0] == base.shape[0] + n_color
    assert abs(with_color[-n_color:].sum() - 1.0) < 1e-4