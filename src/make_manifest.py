"""Build data/manifest.csv with a leakage-safe train/val/test split.

The split is done per physical object (per scene for 'background'), arranged by
class, so no object appears in more than one split.

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

SPLITS = ("train", "val", "test")


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


def main():
    """Runs the checker first and refuses to continue if errors. Then splits each class, double-checks that no objects landed in two splits, writes data/manifest.csv (one row per image with its split) and data/split_summary.csv (counts)"""
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

    by_class = defaultdict(set)
    for s in samples:
        by_class[s.cls].add(s.object_key)

    assignment = {}
    for cls in cfg["classes"]:
        objs = by_class.get(cls)
        if not objs:
            print(f"Note: class '{cls}' has no images, skipped.")
            continue
        rng = random.Random(f"{cfg['seed']}-{cls}")  # per-class seed: adding a class won't reshuffle others
        try:
            assignment.update(assign_splits(objs, fractions, rng))
        except ValueError as exc:
            sys.exit(f"class '{cls}': {exc}. Collect more distinct objects.")

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
    print(f"{'class':<12}" + "".join(f"{sp:>14}" for sp in SPLITS))
    print("-" * 54)
    for cls in cfg["classes"]:
        if cls not in counts:
            continue
        cells = []
        for sp in SPLITS:
            imgs, objs = counts[cls][sp]
            cells.append(f"{imgs} ({len(objs)} obj)")
        print(f"{cls:<12}" + "".join(f"{c:>14}" for c in cells))
    print("\nFormat: images (distinct objects/scenes).")


if __name__ == "__main__":
    main()
