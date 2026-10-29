import numpy as np
import pytest
import torch
import torch.nn as nn

from src.predict import parse_args, predict, preprocess


class FixedModel(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.tensor([logits], dtype=torch.float32)

    def forward(self, x):
        return self.logits


LABELS = ["mug", "bottle", "background"]


def test_parse_image():
    args = parse_args(["--image", "mug.jpg"])
    assert args.image.name == "mug.jpg"
    assert args.camera is None


def test_parse_camera():
    assert parse_args(["--camera", "0"]).camera == 0


def test_parse_requires_one_source():
    with pytest.raises(SystemExit):
        parse_args([])


def test_parse_rejects_two_sources():
    with pytest.raises(SystemExit):
        parse_args(["--image", "a.jpg", "--camera", "0"])


def test_preprocess_shape_and_type():
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    tensor = preprocess(image)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32


def test_predict_confident():
    model = FixedModel([10.0, 0.0, 0.0])
    label, conf = predict(model, torch.zeros(1, 3, 224, 224), LABELS, threshold=0.5)
    assert label == "mug"
    assert conf > 0.9


def test_predict_uncertain_below_threshold():
    model = FixedModel([1.0, 1.0, 1.0])
    label, _ = predict(model, torch.zeros(1, 3, 224, 224), LABELS, threshold=0.5)
    assert label == "uncertain"


def test_predict_background_is_no_object():
    model = FixedModel([0.0, 0.0, 10.0])
    label, _ = predict(model, torch.zeros(1, 3, 224, 224), LABELS, threshold=0.5)
    assert label == "no object"