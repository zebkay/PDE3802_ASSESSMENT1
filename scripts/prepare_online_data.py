"""Turn the downloaded Open Images + Roboflow data into cropped, correctly named images.

What it does
  1. Open Images (downloaded from fiftyOne): finds boxes for the 7 classes, drops flagged boxes
     (occluded / truncated / group-of / depiction / inside), drops small boxes, keeps the
     largest few per photo, caps how many come from one photographer, and picks up to
     TARGET_PER_CLASS["oi"] crops per class.
  2. Roboflow (exported from COCO): does the same for the 'notebook' class only.
  3. Crops each box (with padding), resizes if very large, and saves it to
     data/raw/<class>/<class>_<code><batch>_<index>.jpg   e.g. mug_oi03_0012.jpg
        code   oi = Open Images, rf = Roboflow  (these are not be to used as personal collector codes)
        batch  01-05, chosen by hashing the photographer/source, so one photographer's photos
               never end up in two batches (batches become the train/val units later)
  4. Writes data/sources.csv (source, author, licence, URL, box) and contact sheets in
     data/review/ to eyeball the crops for wrong labels.

Run from the project folder (the one containing config.yaml):
    python scripts/prepare_online_data.py --dry-run     # counts only, writes nothing
    python scripts/prepare_online_data.py               # does the real run

Re-running deletes the previous oi/rf crops it created.
"""

import argparse
import csv
import json
import random
import re
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

OPENIMAGES_DIR = Path.home() / "fiftyone" / "open-images-v7" / "train"
ROBOFLOW_DIR = Path(r"\Users\redre\OneDrive\Documents\CSSE\YEAR 3\Office Items.v26i.coco")

OUT_DIR = Path("data/raw")
SOURCES_CSV = Path("data/sources.csv")
REVIEW_DIR = Path("data/review")

OI_CLASS_MAP = {  # Open Images class name -> your class
    "Mug": "mug", "Bottle": "bottle", "Book": "book",
    "Computer keyboard": "keyboard", "Computer mouse": "mouse",
    "Mobile phone": "phone", "Stapler": "stapler",
}
RF_CLASS_MAP = {"notebook": "notebook", "stapler": "stapler"}

TARGET_PER_CLASS = {"oi": 150, "rf": 150}  # crops per class to keep, per source
MAX_PER_IMAGE = 2                          # crops per class from one photo
MAX_PER_GROUP = {"oi": 5, "rf": 8}         # crops per class from one photographer / filename prefix
MIN_SIDE = 100                             # skip boxes smaller than this (original pixels)
MAX_ASPECT = 3.0                           # skip very long thin boxes (longer side / shorter side)
PAD = 0.10                                 # padding added around each box
MAX_SIDE_OUT = 640                         # shrink crops larger than this
N_BATCHES = 5
SEED = 42
JPEG_QUALITY = 92

OI_BAD_FLAGS = ["IsGroupOf", "IsDepiction", "IsInside"]


# ---------------------------------------------------------------- image I/O
def imread_any(path):
    """cv2.imread that also works with unusual characters in Windows paths."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_any(path, img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError(f"could not encode {path}")
    buf.tofile(str(path))


def batch_of(group_key):
    """Deterministic batch number 1..N_BATCHES from a photographer / source key."""
    return zlib.crc32(group_key.encode("utf-8")) % N_BATCHES + 1


# ---------------------------------------------------------------- Open Images
def load_image_meta(path, wanted_ids):
    """Stream the (huge) image_ids.csv and keep only rows for the images we downloaded."""
    meta = {}
    if not path.exists():
        print(f"Note: {path} not found; author/licence info will be blank")
        return meta
    print("Reading Open Images metadata (large file, can take a minute or two)...")
    with open(path, encoding="utf-8", newline="") as f:
        header = next(csv.reader([f.readline()]))
        for line in f:
            if line.split(",", 1)[0] in wanted_ids:
                try:
                    row = next(csv.reader([line]))
                except Exception:
                    continue
                meta[row[0]] = dict(zip(header, row))
    return meta


def open_images_candidates(root, stats):
    image_dir = root / "data"
    ids = {p.stem for p in image_dir.glob("*.jpg")}
    if not ids:
        raise SystemExit(f"No images found in {image_dir}")
    print(f"Open Images: {len(ids)} downloaded images")

    wanted = {}  # label id -> your class
    with open(root / "metadata" / "classes.csv", encoding="utf-8") as f:
        for line in f:
            mid, name = line.strip().split(",", 1)
            if name in OI_CLASS_MAP:
                wanted[mid] = OI_CLASS_MAP[name]
    meta = load_image_meta(root / "metadata" / "image_ids.csv", ids)

    cands = []
    print("Reading box annotations (large file, can take a minute)...")
    with open(root / "labels" / "detections.csv", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        ix = {name: i for i, name in enumerate(header)}
        for line in f:
            parts = line.split(",")
            image_id = parts[0]
            if image_id not in ids:
                continue
            label = parts[ix["LabelName"]]
            if label not in wanted:
                continue
            cls = wanted[label]
            stats["oi_boxes_seen"] += 1
            bad = [flag for flag in OI_BAD_FLAGS if parts[ix[flag]] == "1"]
            if bad:
                stats["oi_dropped_" + bad[0]] += 1
                continue
            m = meta.get(image_id, {})
            author = m.get("Author", "") or m.get("AuthorProfileURL", "")
            cands.append({
                "dataset": "oi", "cls": cls, "path": image_dir / f"{image_id}.jpg",
                "source_id": image_id,
                "norm": (float(parts[ix["XMin"]]), float(parts[ix["XMax"]]),
                         float(parts[ix["YMin"]]), float(parts[ix["YMax"]])),
                "group": author or image_id, "author": author,
                "licence": m.get("License", ""),
                "url": m.get("OriginalLandingURL", "") or m.get("OriginalURL", ""),
            })
    return cands


# ---------------------------------------------------------------- Roboflow
def roboflow_candidates(root, stats):
    cands = []
    for split in ("train", "valid", "test"):
        ann_path = root / split / "_annotations.coco.json"
        if not ann_path.exists():
            print(f"Note: {ann_path} not found, skipped")
            continue
        coco = json.load(open(ann_path, encoding="utf-8"))
        cat_names = {c["id"]: c["name"] for c in coco["categories"]}
        images = {im["id"]: im for im in coco["images"]}
        for a in coco["annotations"]:
            name = cat_names[a["category_id"]]
            if name not in RF_CLASS_MAP:
                continue
            stats["rf_boxes_seen"] += 1
            im = images[a["image_id"]]
            x, y, w, h = a["bbox"]
            file_name = im["file_name"]
            stem = file_name.split(".rf.")[0]
            group = stem.split("_")[0].lower() or stem.lower()
            cands.append({
                "dataset": "rf", "cls": RF_CLASS_MAP[name], "path": root / split / file_name,
                "source_id": file_name, "px": (x, y, x + w, y + h),
                "group": group, "author": "Workspace1 (Roboflow Universe)",
                "licence": "CC BY 4.0",
                "url": "https://universe.roboflow.com/workspace1-1gbmx/office-items-plugt",
            })
    return cands


# ---------------------------------------------------------------- selection
def to_pixel_boxes(cands, stats):
    """Convert normalised Open Images boxes to pixels, drop small ones, keep the largest few per photo."""
    dims = {}
    out = []
    for c in cands:
        if "px" in c:
            box = c["px"]
        else:
            if c["path"] not in dims:
                img = imread_any(c["path"]) if c["path"].exists() else None
                dims[c["path"]] = None if img is None else img.shape[:2]
            if dims[c["path"]] is None:
                stats["unreadable_image"] += 1
                continue
            h, w = dims[c["path"]]
            xmin, xmax, ymin, ymax = c["norm"]
            box = (xmin * w, ymin * h, xmax * w, ymax * h)
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        if min(x2 - x1, y2 - y1) < MIN_SIDE:
            stats[c["dataset"] + "_dropped_small"] += 1
            continue
        if max(x2 - x1, y2 - y1) > MAX_ASPECT * min(x2 - x1, y2 - y1):
            stats[c["dataset"] + "_dropped_aspect"] += 1
            continue
        c = dict(c, box=(x1, y1, x2, y2), area=(x2 - x1) * (y2 - y1))
        out.append(c)

    per_image = defaultdict(list)
    for c in out:
        per_image[(c["path"], c["cls"])].append(c)
    kept = []
    for items in per_image.values():
        items.sort(key=lambda c: -c["area"])
        kept += items[:MAX_PER_IMAGE]
        stats["dropped_per_image_cap"] += max(0, len(items) - MAX_PER_IMAGE)
    return kept


def select(cands, dataset, rng):
    by_class = defaultdict(list)
    for c in cands:
        if c["dataset"] == dataset:
            by_class[c["cls"]].append(c)
    chosen = []
    for cls in sorted(by_class):
        items = sorted(by_class[cls], key=lambda c: (str(c["path"]), c["box"]))
        rng.shuffle(items)
        per_group, picked = Counter(), []
        for c in items:
            if len(picked) >= TARGET_PER_CLASS[dataset]:
                break
            if per_group[c["group"]] >= MAX_PER_GROUP[dataset]:
                continue
            per_group[c["group"]] += 1
            picked.append(c)
        chosen += picked
    return chosen


# ---------------------------------------------------------------- output
def crop_box(img, box):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    px, py = int((x2 - x1) * PAD), int((y2 - y1) * PAD)
    x1, y1, x2, y2 = max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)
    out = img[y1:y2, x1:x2]
    longest = max(out.shape[:2])
    if longest > MAX_SIDE_OUT:
        scale = MAX_SIDE_OUT / longest
        out = cv2.resize(out, (int(out.shape[1] * scale), int(out.shape[0] * scale)),
                         interpolation=cv2.INTER_AREA)
    return out


def clear_previous(out_dir):
    pattern = re.compile(r"^[a-z]+_(oi|rf)\d{2}_\d{4}\.jpg$")
    removed = 0
    for p in out_dir.glob("*/*.jpg"):
        if pattern.match(p.name):
            p.unlink()
            removed += 1
    if removed:
        print(f"Removed {removed} crops from a previous run")


def write_outputs(chosen, out_dir, dry_run):
    counters = defaultdict(int)
    rows, written = [], defaultdict(list)
    cache_path, cache_img = None, None
    for c in sorted(chosen, key=lambda c: (str(c["path"]), c["box"])):
        batch = batch_of(c["group"])
        key = (c["cls"], c["dataset"], batch)
        counters[key] += 1
        name = f"{c['cls']}_{c['dataset']}{batch:02d}_{counters[key]:04d}.jpg"
        dest = out_dir / c["cls"] / name
        rows.append({
            "filename": name, "class": c["cls"],
            "source_dataset": "Open Images V7" if c["dataset"] == "oi" else "Roboflow Universe",
            "source_id": c["source_id"], "author": c["author"], "licence": c["licence"],
            "url": c["url"], "batch": batch,
            "box_x1_y1_x2_y2": " ".join(str(v) for v in c["box"]),
        })
        written[c["cls"]].append(dest)
        if dry_run:
            continue
        if cache_path != c["path"]:
            cache_path, cache_img = c["path"], imread_any(c["path"])
        if cache_img is None:
            print(f"Warning: could not read {c['path']}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        imwrite_any(dest, crop_box(cache_img, c["box"]))
    return rows, written


def write_sources(rows):
    SOURCES_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(SOURCES_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["filename"]))


def contact_sheets(written):
    """Grids of thumbnails (labelled with the batch/index) for quick visual checking."""
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    thumb, cols, rows_per_sheet = 140, 8, 5
    per_sheet = cols * rows_per_sheet
    for cls, paths in written.items():
        paths = sorted(paths)
        for s in range(0, len(paths), per_sheet):
            canvas = np.full((rows_per_sheet * thumb, cols * thumb, 3), 40, np.uint8)
            for i, p in enumerate(paths[s:s + per_sheet]):
                img = imread_any(p)
                if img is None:
                    continue
                img = cv2.resize(img, (thumb - 4, thumb - 4), interpolation=cv2.INTER_AREA)
                label = p.stem.split("_", 1)[1]
                cv2.putText(img, label, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
                cv2.putText(img, label, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                r, c = divmod(i, cols)
                canvas[r * thumb + 2:r * thumb + 2 + thumb - 4, c * thumb + 2:c * thumb + 2 + thumb - 4] = img
            imwrite_any(REVIEW_DIR / f"{cls}_sheet{s // per_sheet + 1:02d}.jpg", canvas)


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description="Crop and name online training images.")
    parser.add_argument("--dry-run", action="store_true", help="print counts only, write nothing")
    parser.add_argument("--openimages-dir", default=str(OPENIMAGES_DIR))
    parser.add_argument("--roboflow-dir", default=str(ROBOFLOW_DIR))
    args = parser.parse_args()

    stats = Counter()
    rng = random.Random(SEED)
    all_cands = []

    oi_root = Path(args.openimages_dir)
    if oi_root.exists():
        all_cands += open_images_candidates(oi_root, stats)
    else:
        print(f"Warning: Open Images folder not found: {oi_root}")

    rf_root = Path(args.roboflow_dir)
    if rf_root.exists():
        all_cands += roboflow_candidates(rf_root, stats)
    else:
        print(f"Warning: Roboflow folder not found: {rf_root} (edit ROBOFLOW_DIR at the top of this file)")

    if not all_cands:
        raise SystemExit("No candidate boxes found.")

    print("Measuring boxes...")
    candidates = to_pixel_boxes(all_cands, stats)
    chosen = select(candidates, "oi", rng) + select(candidates, "rf", rng)

    if not args.dry_run:
        clear_previous(OUT_DIR)
    rows, written = write_outputs(chosen, OUT_DIR, args.dry_run)

    # ---- report
    print("\nWhy boxes were dropped:")
    for k, v in sorted(stats.items()):
        print(f"  {k:<28}{v:>8}")
    print(f"\n{'class':<12}{'selected':>9}   per batch (01..{N_BATCHES:02d})")
    print("-" * 52)
    for cls in sorted({c['cls'] for c in chosen}):
        items = [c for c in chosen if c["cls"] == cls]
        per_batch = Counter(batch_of(c["group"]) for c in items)
        spread = " ".join(f"{per_batch.get(b, 0):>3}" for b in range(1, N_BATCHES + 1))
        print(f"{cls:<12}{len(items):>9}   {spread}")
    print(f"{'total':<12}{len(chosen):>9}")

    if args.dry_run:
        print("\nDry run: nothing was written. Run again without --dry-run to create the images.")
        return
    write_sources(rows)
    contact_sheets(written)
    print(f"\nWrote crops to {OUT_DIR}, {SOURCES_CSV} and contact sheets in {REVIEW_DIR}")
    print("Next: look through the contact sheets, delete bad crops, then run: python -m src.check_dataset")


if __name__ == "__main__":
    main()
