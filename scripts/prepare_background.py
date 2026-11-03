"""Prepare the 'background' class from the Places365 validation images (256x256) dataset.

Steps:

  Step 1  (no flag)     Picks candidate images from office-like scene categories and copies
                        them into data/review/bg_candidates/ . Open that folder in File
                        Explorer (View > Large icons) and DELETE every image that:
                          - shows any target object, even a small or partly hidden one
                            (mug, bottle, book, notebook, keyboard, mouse, phone, stapler)
                          - shows a person or a face
                          - is a poor image (watermarked, tiny content, not an office-type scene)
  More    (--more)      If images kept fewer than TARGET, offers NEW candidates (never the ones
                        you already saw) in data/review/bg_candidates_more1/ (then _more2, ...).
                        It gives more candidates to the categories where the most were kept.
                        Review that folder the same way.
  Step 2  (--finalize)  Takes the images you kept from ALL bg_candidates* folders (up to TARGET),
                        names them background_pl<batch>_<index>.jpg , copies them to
                        data/raw/background/ and records them in data/sources.csv.

Run from the project folder (the one containing config.yaml):
    python scripts/prepare_background.py
    python scripts/prepare_background.py --more            # optional: --count 300
    python scripts/prepare_background.py --finalize

Before step 1, extract both downloaded .tar files into PLACES_DIR (see README / hand-off notes).
"""

import argparse
import csv
import difflib
import json
import random
import re
import shutil
import zlib
from collections import Counter
from pathlib import Path


# A folder that contains (anywhere inside it) the extracted val_256 images and the label files
PLACES_DIR = Path(r"\Users\redre\OneDrive\Documents\CSSE\YEAR 3")

REVIEW_DIR = Path("data/review")
CANDIDATE_DIR = REVIEW_DIR / "bg_candidates"        # round 1
OFFERED_JSON = REVIEW_DIR / "bg_offered.json"        # how many images per category were offered so far
OUT_DIR = Path("data/raw/background")
SOURCES_CSV = Path("data/sources.csv")

# Scene categories likely to look like where the robot works. Computer rooms, kitchens,
# libraries and bookshops are left out on purpose: they are full of target objects.
CATEGORIES = [
    "office", "office_cubicles", "home_office", "conference_room", "server_room",
    "corridor", "lobby", "waiting_room", "classroom", "reception",
]
PER_CATEGORY = 30   # candidates offered per category in round 1
TARGET = 150        # images to keep in the dataset
N_BATCHES = 5
SEED = 42

SOURCE_NAME = "Places365 (validation set, 256x256)"
SOURCE_URL = "http://places2.csail.mit.edu/"
SOURCE_LICENCE = "Places365 terms (see places2.csail.mit.edu); credit Zhou et al., TPAMI"
FIELDS = ["filename", "class", "source_dataset", "source_id", "author", "licence",
          "url", "batch", "box_x1_y1_x2_y2"]
NAME_RE = re.compile(r"^background_pl\d{2}_\d{4}\.jpg$")


def batch_of(key):
    return zlib.crc32(key.encode("utf-8")) % N_BATCHES + 1


def candidate_dirs():
    """Round 1 folder plus any bg_candidates_more* folders."""
    return sorted(p for p in REVIEW_DIR.glob("bg_candidates*") if p.is_dir())


# ---------------------------------------------------------------- finding the Places files
def locate_files(root):
    if not root.exists():
        raise SystemExit(f"PLACES_DIR not found: {root}\nEdit PLACES_DIR at the top of this file.")
    labels = next(root.rglob("places365_val.txt"), None)
    cats = next(root.rglob("categories_places365.txt"), None)
    images = {p.name: p for p in root.rglob("Places365_val_*.jpg")}
    problems = []
    if not images:
        problems.append("No Places365_val_*.jpg images found. Extract val_256.tar first: tar -xf val_256.tar")
    if labels is None:
        problems.append("places365_val.txt not found. Extract filelist_places365-standard.tar first.")
    if cats is None:
        problems.append(
            "categories_places365.txt not found. Download it with:\n"
            "  curl.exe -L -o categories_places365.txt "
            "https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt"
        )
    if problems:
        raise SystemExit("\n".join(problems))
    return labels, cats, images


def read_categories(path):
    """Returns {short_name: index}, e.g. {'office': 280}."""
    out = {}
    for line in open(path, encoding="utf-8"):
        parts = line.split()
        if len(parts) != 2:
            continue
        name = parts[0]
        short = name.split("/", 2)[2] if name.startswith("/") and name.count("/") >= 2 else name
        out[short] = int(parts[1])
    return out


def shuffled_by_category(verbose=False):
    """Each category's image list in a FIXED shuffled order (same seed every time).

    Round 1 uses the first PER_CATEGORY of each list, later rounds continue further down the
    same lists, so a new round never repeats an image you already reviewed.
    """
    labels_path, cats_path, images = locate_files(PLACES_DIR)
    cat_index = read_categories(cats_path)
    by_idx = {}
    for line in open(labels_path, encoding="utf-8"):
        parts = line.split()
        if len(parts) == 2:
            by_idx.setdefault(int(parts[1]), []).append(parts[0])

    rng = random.Random(SEED)
    ordered = {}
    for cat in CATEGORIES:
        if cat not in cat_index:
            if verbose:
                close = difflib.get_close_matches(cat, cat_index, n=3)
                print(f"{cat:<20}  NOT FOUND (similar: {close})")
            continue
        files = sorted(f for f in by_idx.get(cat_index[cat], []) if f in images)
        rng.shuffle(files)
        ordered[cat] = [(f, images[f]) for f in files]
    return ordered


# ---------------------------------------------------------------- step 1
def step_candidates(force):
    if CANDIDATE_DIR.exists() and any(CANDIDATE_DIR.glob("*.jpg")) and not force:
        raise SystemExit(
            f"{CANDIDATE_DIR} already has images. Re-running would undo your review.\n"
            "To get MORE candidates without losing your review, run:\n"
            "  python scripts/prepare_background.py --more\n"
            "(--force wipes the folder and starts again; only use it if you really want that.)"
        )
    ordered = shuffled_by_category(verbose=True)
    if CANDIDATE_DIR.exists():
        shutil.rmtree(CANDIDATE_DIR)
    CANDIDATE_DIR.mkdir(parents=True)

    offered, total = {}, 0
    print(f"{'category':<20}{'copied':>7}")
    for cat, files in ordered.items():
        n = min(PER_CATEGORY, len(files))
        for name, src in files[:n]:
            shutil.copy2(src, CANDIDATE_DIR / f"{cat.replace('/', '-')}__{name}")
        offered[cat] = n
        total += n
        print(f"{cat:<20}{n:>7}")
    OFFERED_JSON.write_text(json.dumps(offered, indent=2), encoding="utf-8")
    print(f"\n{total} candidates copied to {CANDIDATE_DIR}")
    print("NOW: open that folder in File Explorer (View > Large icons) and delete every image that")
    print("  - contains any target object (even small / half hidden): mug, bottle, book, notebook,")
    print("    keyboard, mouse, phone, stapler")
    print("  - contains a person or face, or is a poor image")
    print(f"Aim to keep at least {TARGET}. Then run: python scripts/prepare_background.py --finalize")


# ---------------------------------------------------------------- more candidates
def count_kept_per_category():
    kept = Counter()
    for d in candidate_dirs():
        for p in d.glob("*.jpg"):
            kept[p.name.split("__", 1)[0]] += 1
    return kept


def step_more(count):
    if not CANDIDATE_DIR.exists():
        raise SystemExit("No round-1 candidates yet. Run step 1 first (no flags).")
    ordered = shuffled_by_category(verbose=True)

    # How many have been offered per category so far. Round 1 of the original script
    # always offered PER_CATEGORY, so that is the fallback if the record file is missing.
    if OFFERED_JSON.exists():
        offered = json.loads(OFFERED_JSON.read_text(encoding="utf-8"))
    else:
        offered = {cat: min(PER_CATEGORY, len(files)) for cat, files in ordered.items()}

    kept = count_kept_per_category()
    total_kept = sum(kept.values())
    if total_kept >= TARGET:
        print(f"You already have {total_kept} kept images (target {TARGET}). Run --finalize.")
        return

    yields, available = {}, {}
    for cat, files in ordered.items():
        off = offered.get(cat, 0)
        yields[cat] = kept.get(cat.replace('/', '-'), 0) / off if off else 0.0
        available[cat] = max(0, len(files) - off)

    # Give more candidates to categories where you kept more (floor of 5% so none is zero)
    weights = {cat: max(yields[cat], 0.05) for cat in ordered if available[cat] > 0}
    if not weights:
        raise SystemExit("No unseen images left in these categories. Add categories to CATEGORIES.")
    wsum = sum(weights.values())
    alloc = {cat: min(available[cat], round(count * w / wsum)) for cat, w in weights.items()}

    k = 1
    while (REVIEW_DIR / f"bg_candidates_more{k}").exists():
        k += 1
    out = REVIEW_DIR / f"bg_candidates_more{k}"
    out.mkdir(parents=True)

    print(f"\nKept so far: {total_kept} (target {TARGET})")
    print(f"{'category':<20}{'offered':>8}{'kept':>6}{'keep-rate':>11}{'new':>6}")
    new_total = 0
    for cat in ordered:
        off, n = offered.get(cat, 0), alloc.get(cat, 0)
        for name, src in ordered[cat][off:off + n]:
            shutil.copy2(src, out / f"{cat.replace('/', '-')}__{name}")
        offered[cat] = off + n
        new_total += n
        print(f"{cat:<20}{off:>8}{kept.get(cat.replace('/', '-'), 0):>6}{yields[cat]:>10.0%}{n:>6}")
    OFFERED_JSON.write_text(json.dumps(offered, indent=2), encoding="utf-8")

    overall = total_kept / max(1, sum(v for v in (offered.get(c, 0) - alloc.get(c, 0) for c in offered)))
    print(f"\n{new_total} NEW candidates copied to {out}  (expected keep-rate about {overall:.0%})")
    print("Review that folder the same way (delete anything with a target object, a person, or poor quality).")
    print("Then run: python scripts/prepare_background.py --finalize   (or --more again if still short)")


# ---------------------------------------------------------------- step 2
def update_sources(new_rows):
    kept = []
    if SOURCES_CSV.exists():
        with open(SOURCES_CSV, newline="", encoding="utf-8") as f:
            kept = [r for r in csv.DictReader(f) if not NAME_RE.match(r.get("filename", ""))]
    rows = kept + new_rows
    SOURCES_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(SOURCES_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["filename"]))


def step_finalize():
    kept = sorted(p for d in candidate_dirs() for p in d.glob("*.jpg"))
    if not kept:
        raise SystemExit(f"No images in {CANDIDATE_DIR}. Run step 1 first.")
    print(f"Found {len(kept)} kept images in {len(candidate_dirs())} folder(s).")
    rng = random.Random(SEED)
    rng.shuffle(kept)
    if len(kept) > TARGET:
        print(f"Using a random {TARGET} of them.")
        kept = kept[:TARGET]
    elif len(kept) < TARGET:
        print(f"Note: only {len(kept)} images kept (target {TARGET}). "
              "Run --more for extra candidates, or continue if you accept this number.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in OUT_DIR.glob("*.jpg"):
        if NAME_RE.match(p.name):
            p.unlink()

    counters, rows, per_cat = Counter(), [], Counter()
    for p in sorted(kept):
        category, original = p.name.split("__", 1)
        batch = batch_of(original)
        counters[batch] += 1
        name = f"background_pl{batch:02d}_{counters[batch]:04d}.jpg"
        shutil.copy2(p, OUT_DIR / name)
        per_cat[category] += 1
        rows.append({"filename": name, "class": "background", "source_dataset": SOURCE_NAME,
                     "source_id": original, "author": f"scene category: {category}",
                     "licence": SOURCE_LICENCE, "url": SOURCE_URL, "batch": batch,
                     "box_x1_y1_x2_y2": ""})
    update_sources(rows)

    print(f"\nWrote {len(rows)} images to {OUT_DIR} and updated {SOURCES_CSV}")
    print("Per batch:    " + "  ".join(f"{b:02d}:{counters.get(b, 0)}" for b in range(1, N_BATCHES + 1)))
    print("Per category: " + ", ".join(f"{c}={n}" for c, n in sorted(per_cat.items())))
    print("Next: python -m src.check_dataset")


def main():
    parser = argparse.ArgumentParser(description="Prepare the background class from Places365.")
    parser.add_argument("--more", action="store_true", help="offer NEW candidates (keeps your earlier review)")
    parser.add_argument("--count", type=int, default=300, help="with --more: how many new candidates")
    parser.add_argument("--finalize", action="store_true", help="build the dataset from the images you kept")
    parser.add_argument("--force", action="store_true", help="step 1: wipe and restart round 1")
    args = parser.parse_args()
    if args.finalize:
        step_finalize()
    elif args.more:
        step_more(args.count)
    else:
        step_candidates(args.force)


if __name__ == "__main__":
    main()