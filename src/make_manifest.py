"""Build data/manifest.csv with a leakage-safe split.

Splits: train, val and test come from online batches; test_own is our own photos.
Each object (or scene for 'background') lands in one split only.

Run from the repo root:
    python -m src.make_manifest
    python -m src.make_manifest --force   # build even if the checker reports errors
"""

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

from .check_dataset import collect_samples, print_section
from .data_utils import load_config

SPLITS = ("train", "val", "test", "test_own")


def assign_splits(object_keys, fractions, rng):
    """Map each object key to 'train', 'val' or 'test'."""
    objs = sorted(object_keys)
    rng.shuffle(objs)
    n = len(objs)
    n_test = max(1, round(n * fractions["test"]))
    n_val = max(1, round(n * fractions["val"]))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError(f"{n} objects is too few to fill train, val and test")
    out = {}
    for i, key in enumerate(objs):
        out[key] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return out

def assign_by_source(samples, cfg):
    """Decide each object's split from WHO collected it (not by random shuffling).

    online_collectors (oi, rf, pl): split by BATCH. val_batches -> val, test_batches -> test,
    the rest -> train.
    test_collectors  (your own photos): everything -> test_own.
    train_collectors (optional extra own photos): everything -> train.
    Returns (assignment {object_key: split}, errors, notes).
    """
    online = set(cfg.get("online_collectors") or [])
    test_c = set(cfg.get("test_collectors") or [])
    train_c = set(cfg.get("train_collectors") or [])
    val_batches = set(cfg.get("val_batches") or [])
    test_batches = set(cfg.get("test_batches") or [])

    errors, notes = [], []
    for code in sorted((online & test_c) | (online & train_c) | (test_c & train_c)):
        errors.append(f"collector code '{code}' is listed in more than one of "
                      "online_collectors / test_collectors / train_collectors")
    for batch in sorted(val_batches & test_batches):
        errors.append(f"online batch {batch} is listed in both val_batches and test_batches")

    assignment, unknown = {}, set()
    for s in samples:
        if s.object_key in assignment:
            continue
        if s.collector in test_c:
            assignment[s.object_key] = "test_own"
        elif s.collector in train_c:
            assignment[s.object_key] = "train"
        elif s.collector in online:
            batch = int(s.object_id[len(s.collector):])
            if batch in val_batches:
                assignment[s.object_key] = "val"
            elif batch in test_batches:
                assignment[s.object_key] = "test"
            else:
                assignment[s.object_key] = "train"
        else:
            unknown.add(s.collector)
    for code in sorted(unknown):
        errors.append(f"collector '{code}' has images but is not in online_collectors, "
                      "test_collectors or train_collectors in config.yaml")

    if "test" not in assignment.values():
        notes.append("No online test images (check test_batches). --final will refuse until they exist.")
    if "test_own" not in assignment.values():
        notes.append("No own-photo test images yet (test_collectors is empty, or those collectors "
                     "have no files). --final will refuse until they exist.")
    notes.append(f"Online batches {sorted(val_batches)} -> val; {sorted(test_batches)} -> test; "
                 "other online batches -> train.")
    return assignment, errors, notes

def main():
    parser = argparse.ArgumentParser(description="Create the dataset manifest and split.")
    parser.add_argument("--config", default="configuration.yaml")
    parser.add_argument("--force", action="store_true", help="ignore checker errors")
    args = parser.parse_args()

    cfg = load_config(args.config)
    fractions = cfg["split"]
    if abs(sum(fractions.values()) - 1.0) > 1e-6:
        sys.exit("split fractions in configuration.yaml must sum to 1.0")

    samples, errors, _ = collect_samples(cfg)
    if errors and not args.force:
        print_section("ERRORS (fix these, or use --force)", errors)
        sys.exit(1)
    if not samples:
        sys.exit("No valid images found.")

    assignment, split_errors, notes = assign_by_source(samples, cfg)
    if split_errors:
        sys.exit("\n".join(split_errors))
    for note in notes:
        print("Note:", note)

    rows = []
    for s in sorted(samples, key=lambda s: str(s.path)):
        rows.append({
            "path": s.path.as_posix(),
            "class": s.cls,
            "object_id": s.object_id,
            "collector": s.collector,
            "index": s.index,
            "split": assignment[s.object_key],
        })

    # Sanity check: no object may appear in more than one split
    seen = defaultdict(set)
    for s, r in zip(sorted(samples, key=lambda s: str(s.path)), rows):
        seen[s.object_key].add(r["split"])
    assert all(len(v) == 1 for v in seen.values()), "an object landed in several splits"

    manifest_path = Path(cfg["manifest_path"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Summary: images (objects) per class and split
    counts = defaultdict(lambda: defaultdict(lambda: [0, set()]))
    for s, r in zip(sorted(samples, key=lambda s: str(s.path)), rows):
        cell = counts[s.cls][r["split"]]
        cell[0] += 1
        cell[1].add(s.object_key)

    summary_path = Path(cfg["summary_path"])
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "split", "objects", "images"])
        for cls in cfg["classes"]:
            for sp in SPLITS:
                imgs, objs = counts[cls][sp] if cls in counts else (0, set())
                writer.writerow([cls, sp, len(objs), imgs])

    print(f"Wrote {manifest_path} ({len(rows)} images) and {summary_path}\n")
    print(f"{'class':<12}" + "".join(f"{sp:>16}" for sp in SPLITS))
    print("-" * 76)
    for cls in cfg["classes"]:
        if cls not in counts:
            continue
        cells = []
        for sp in SPLITS:
            imgs, objs = counts[cls][sp]
            cells.append(f"{imgs} ({len(objs)} obj)")
        print(f"{cls:<12}" + "".join(f"{c:>16}" for c in cells))
    print("\nFormat: images (distinct objects/scenes).")


if __name__ == "__main__":
    main()
