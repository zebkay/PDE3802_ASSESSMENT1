"""Tests for src/metrics.py. 
Contains automated checks that src/metrics.py calculates correctly, by trying it on tiny made-up examples where the right answer is known.
Run from the repo root:  python -m pytest tests"""

import json
import pytest
from src.metrics import build_comparison, evaluate_predictions

CLASSES = ["mug", "bottle", "book"]


def test_perfect_predictions(tmp_path):
    y = ["mug", "bottle", "book", "mug"]
    s = evaluate_predictions(y, y, CLASSES, tmp_path, "perfect")
    assert s["accuracy"] == 1.0
    assert s["macro_f1"] == 1.0
    assert s["top_confusions"] == []


def test_known_confusion(tmp_path):
    # 2 mugs, 2 bottles; one mug predicted as bottle
    y_true = ["mug", "mug", "bottle", "bottle"]
    y_pred = ["mug", "bottle", "bottle", "bottle"]
    s = evaluate_predictions(y_true, y_pred, CLASSES, tmp_path, "m")
    assert s["accuracy"] == pytest.approx(0.75)
    assert s["per_class"]["mug"]["recall"] == pytest.approx(0.5)
    assert s["per_class"]["bottle"]["precision"] == pytest.approx(2 / 3)
    assert s["top_confusions"][0] == {"true": "mug", "predicted": "bottle", "count": 1}


def test_class_absent_from_split_is_not_penalised(tmp_path):
    # 'book' never appears in y_true, so macro-F1 averages over mug and bottle only
    y = ["mug", "bottle"]
    s = evaluate_predictions(y, y, CLASSES, tmp_path, "m")
    assert s["macro_f1"] == 1.0
    assert s["classes_missing_from_split"] == ["book"]


def test_files_written_and_mistakes_sorted_by_confidence(tmp_path):
    evaluate_predictions(
        ["mug", "mug", "book"], ["bottle", "book", "book"], CLASSES, tmp_path, "m",
        paths=["a.jpg", "b.jpg", "c.jpg"], confidences=[0.6, 0.9, 0.99],
    )
    out = tmp_path / "m" / "test"
    for name in ("metrics.json", "per_class_report.csv", "misclassified.csv",
                 "confusion_matrix.png", "confusion_matrix_normalised.png"):
        assert (out / name).exists(), name
    lines = (out / "misclassified.csv").read_text().strip().splitlines()
    assert lines[1].startswith("b.jpg")  # 0.9 before 0.6


def test_unknown_label_raises(tmp_path):
    with pytest.raises(ValueError):
        evaluate_predictions(["mug"], ["spoon"], CLASSES, tmp_path, "m")


def test_length_mismatch_raises(tmp_path):
    with pytest.raises(ValueError):
        evaluate_predictions(["mug", "book"], ["mug"], CLASSES, tmp_path, "m")


def test_comparison_table(tmp_path):
    y = ["mug", "bottle", "book"]
    evaluate_predictions(y, y, CLASSES, tmp_path, "model_a", extra={"inference_ms": 5.0})
    evaluate_predictions(y, ["mug", "mug", "book"], CLASSES, tmp_path, "model_b")
    build_comparison(tmp_path, "test")
    md = (tmp_path / "comparison_test.md").read_text()
    assert "model_a" in md and "model_b" in md and "inference_ms" in md
    assert json.loads((tmp_path / "model_a" / "test" / "metrics.json").read_text())["accuracy"] == 1.0