"""Validate data/raw against the naming convention and basic quality rules.

Run from the repo root:
    python -m src.check_dataset
    python -m src.check_dataset --config configuration.yaml

Exit code is 1 if there are errors (warnings do not fail the check).
"""

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

from .data_utils import Sample, iter_image_files, load_config, parse_filename

GPS_IFD = 0x8825  # EXIF tag that holds GPS location data
MAX_LINES = 40    # cap printed lines per section


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_samples(cfg: dict):
    """Scan raw_dir. Returns (samples, errors, warnings)."""
    raw_dir = Path(cfg["raw_dir"])
    classes = set(cfg["classes"])
    collectors = set((cfg.get("collectors") or {}).keys())
    min_side = cfg.get("min_side_px", 224)
    samples, errors, warnings = [], [], []
    hashes = defaultdict(list)

    if not raw_dir.is_dir():
        errors.append(f"raw_dir '{raw_dir}' does not exist")
        return samples, errors, warnings

    for path in iter_image_files(raw_dir):
        rel = path.relative_to(raw_dir)
        parsed = parse_filename(path.name)
        if parsed is None:
            errors.append(
                f"{rel}: name must be <class>_<objectID>_<index>.jpg "
                f"(all lowercase, e.g. mug_ak01_0001.jpg)"
            )
            continue

        folder = rel.parts[0] if len(rel.parts) > 1 else ""
        if parsed["cls"] not in classes:
            errors.append(f"{rel}: class '{parsed['cls']}' is not in configuration.yaml")
            continue
        if folder != parsed["cls"]:
            errors.append(f"{rel}: file is in folder '{folder}' but its name says '{parsed['cls']}'")
            continue
        if collectors and parsed["collector"] not in collectors:
            errors.append(f"{rel}: unknown collector code '{parsed['collector']}'")
            continue

        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                w, h = im.size
                if min(w, h) < min_side:
                    warnings.append(f"{rel}: small image ({w}x{h}), shortest side < {min_side}px")
                try:
                    if im.getexif().get_ifd(GPS_IFD):
                        warnings.append(f"{rel}: contains GPS location data; strip EXIF before sharing")
                except Exception:
                    pass
        except Exception as exc:
            errors.append(f"{rel}: cannot be read as an image ({exc})")
            continue

        sample = Sample(path, parsed["cls"], parsed["object_id"], parsed["collector"], parsed["index"])
        samples.append(sample)
        hashes[md5sum(path)].append(sample)

    # Exact duplicate files
    for group in hashes.values():
        if len(group) < 2:
            continue
        names = ", ".join(s.path.name for s in group)
        if len({s.object_key for s in group}) > 1:
            errors.append(f"identical image appears under different objects (leakage risk): {names}")
        else:
            warnings.append(f"identical duplicate images within one object: {names}")

    # Per-class quality targets
    per_class = defaultdict(lambda: defaultdict(int))
    for s in samples:
        per_class[s.cls][s.object_key] += 1

    for cls in cfg["classes"]:
        objs = per_class.get(cls, {})
        if not objs:
            warnings.append(f"class '{cls}': no images yet")
            continue
        if len(objs) < cfg.get("min_objects_per_class", 5):
            warnings.append(
                f"class '{cls}': only {len(objs)} distinct objects "
                f"(target {cfg.get('min_objects_per_class', 5)})"
            )
        target = cfg.get("min_images_per_object", 20)
        short = {k: n for k, n in objs.items() if n < target}
        if short:
            worst = ", ".join(f"{k.split('_', 1)[1]}={n}" for k, n in sorted(short.items(), key=lambda kv: kv[1])[:5])
            warnings.append(
                f"class '{cls}': {len(short)} of {len(objs)} objects have fewer than {target} images "
                f"(lowest: {worst})"
            )
    return samples, errors, warnings


def print_summary(samples, cfg):
    per_class = defaultdict(lambda: defaultdict(int))
    collectors = defaultdict(set)
    for s in samples:
        per_class[s.cls][s.object_key] += 1
        collectors[s.cls].add(s.collector)

    print(f"{'class':<12}{'images':>8}{'objects':>9}{'min img/obj':>13}  collectors")
    print("-" * 56)
    total = 0
    for cls in cfg["classes"]:
        objs = per_class.get(cls, {})
        n_img = sum(objs.values())
        total += n_img
        lo = min(objs.values()) if objs else 0
        print(f"{cls:<12}{n_img:>8}{len(objs):>9}{lo:>13}  {','.join(sorted(collectors[cls]))}")
    print("-" * 56)
    print(f"{'total':<12}{total:>8}")


def print_section(title, lines):
    if not lines:
        return
    print(f"\n{title} ({len(lines)})")
    for line in lines[:MAX_LINES]:
        print(f"  - {line}")
    if len(lines) > MAX_LINES:
        print(f"  ... and {len(lines) - MAX_LINES} more")


def main():
    parser = argparse.ArgumentParser(description="Check the raw dataset against the naming convention.")
    parser.add_argument("--config", default="configuration.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    samples, errors, warnings = collect_samples(cfg)
    print_summary(samples, cfg)
    print_section("ERRORS (must fix)", errors)
    print_section("WARNINGS (review)", warnings)
    print("\nResult:", "FAILED" if errors else "OK (no errors)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()