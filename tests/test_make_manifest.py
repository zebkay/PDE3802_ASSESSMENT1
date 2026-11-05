"""Tests for the split rules in src/make_manifest.py. No image files needed."""

from pathlib import Path

from src.data_utils import Sample
from src.make_manifest import assign_by_source

CFG = {"online_collectors": ["oi", "pl"], "val_batches": [4], "test_batches": [5],
       "test_collectors": ["ak"], "train_collectors": ["bs"]}


def S(cls, obj, collector):
    return Sample(Path(f"{cls}_{obj}_0001.jpg"), cls, obj, collector, 1)


def test_online_batches_go_to_train_val_or_test():
    samples = [S("mug", "oi01", "oi"), S("mug", "oi04", "oi"), S("mug", "oi05", "oi")]
    a, errors, _ = assign_by_source(samples, CFG)
    assert not errors
    assert a["mug_oi01"] == "train" and a["mug_oi04"] == "val" and a["mug_oi05"] == "test"


def test_own_photos_go_to_test_own_and_extra_own_photos_to_train():
    a, _, _ = assign_by_source([S("mug", "ak01", "ak"), S("mug", "bs01", "bs")], CFG)
    assert a["mug_ak01"] == "test_own" and a["mug_bs01"] == "train"


def test_one_object_gets_exactly_one_split_even_with_many_images():
    samples = [S("mug", "oi04", "oi")] * 3
    a, _, _ = assign_by_source(samples, CFG)
    assert list(a.values()) == ["val"]


def test_unlisted_collector_is_an_error():
    _, errors, _ = assign_by_source([S("mug", "zz01", "zz")], CFG)
    assert errors and "zz" in errors[0]


def test_collector_in_two_lists_is_an_error():
    cfg = dict(CFG, train_collectors=["ak"])
    _, errors, _ = assign_by_source([S("mug", "ak01", "ak")], cfg)
    assert any("more than one" in e for e in errors)


def test_batch_in_both_val_and_test_is_an_error():
    cfg = dict(CFG, test_batches=[4])
    _, errors, _ = assign_by_source([S("mug", "oi01", "oi")], cfg)
    assert any("both val_batches and test_batches" in e for e in errors)


def test_notes_when_test_sets_are_missing():
    _, _, notes = assign_by_source([S("mug", "oi01", "oi")], CFG)
    assert any("No online test images" in n for n in notes)
    assert any("No own-photo test images" in n for n in notes)