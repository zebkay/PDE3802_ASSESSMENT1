"""Tests for src/train_cnn.py helpers. They use weights=None, so nothing is downloaded."""

import numpy as np
from tensorflow import keras

from src.train_cnn import build_model, compute_class_weights, freeze_backbone, unfreeze_top


def test_model_output_shape_and_softmax():
    model, _ = build_model("small", num_classes=9, img_size=96, weights="none", dropout=0.2)
    out = model(np.random.rand(2, 96, 96, 3).astype("float32") * 255, training=False).numpy()
    assert out.shape == (2, 9)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-4)


def test_freeze_then_unfreeze_keeps_batchnorm_frozen():
    _, base = build_model("small", num_classes=3, img_size=96, weights="none", dropout=0.2)
    freeze_backbone(base)
    assert not any(layer.trainable for layer in base.layers)
    unfreeze_top(base, 0.3)
    bn = [layer for layer in base.layers if isinstance(layer, keras.layers.BatchNormalization)]
    assert bn and not any(layer.trainable for layer in bn)
    assert any(layer.trainable for layer in base.layers)  # some layers are now trainable


def test_class_weights_favour_small_classes():
    labels = np.array([0] * 8 + [1] * 2)
    w = compute_class_weights(labels)
    assert w[1] > w[0]
    assert abs(w[0] - 10 / (2 * 8)) < 1e-9