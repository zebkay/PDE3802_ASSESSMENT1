"""Copy one object's photos into data/raw/<class>/ with the correct filenames.

Originals are left untouched. Each copy is saved as JPG, turned the right way up,
reduced to at most 1280 px on its longest side, and has its metadata (including
GPS location) removed.

Run from the repo root, once per physical object or scene:
    python scripts/rename_photos.py --src "C:\\path\\to\\photos" --cls mug --obj tu01
"""

import argparse
import re
import sys
from pathlib import Path

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
MAX_SIDE = 1280


def main():
    parser = argparse.ArgumentParser(description="Copy and name photos for one object")
    parser.add_argument("--src", type=Path, required=True, help="folder holding this object's photos")
    parser.add_argument("--cls", required=True, help="class name, e.g. mug")
    parser.add_argument("--obj", required=True, help="object ID, e.g. tu01")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    if not re.fullmatch(r"[a-z]{2,3}\d{2}", args.obj):
        sys.exit("Error: --obj must be 2-3 lowercase letters then 2 digits, e.g. tu01")
    if not args.src.is_dir():
        sys.exit(f"Error: '{args.src}' is not a folder")
    out_dir = args.raw_dir / args.cls
    if not out_dir.is_dir():
        sys.exit(f"Error: '{out_dir}' does not exist. Check --cls and run from the repo root.")

    existing = [int(p.stem.rsplit("_", 1)[1]) for p in out_dir.glob(f"{args.cls}_{args.obj}_*.jpg")]
    index = max(existing, default=0)

    files = sorted(p for p in args.src.iterdir() if p.is_file())
    written = 0
    for path in files:
        if path.suffix.lower() not in IMAGE_EXTS:
            print(f"Skipped {path.name}: not JPG or PNG (convert HEIC to JPG first)")
            continue
        img = cv2.imread(str(path))
        if img is None:
            print(f"Skipped {path.name}: could not be read")
            continue
        scale = MAX_SIDE / max(img.shape[:2])
        if scale < 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        index += 1
        cv2.imwrite(str(out_dir / f"{args.cls}_{args.obj}_{index:04d}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        written += 1
    print(f"Wrote {written} images to {out_dir}")


if __name__ == "__main__":
    main()
