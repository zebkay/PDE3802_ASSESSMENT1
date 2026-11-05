"""Make a JPG copy of every HEIC photo in a folder and its subfolders.

The HEIC originals are left in place. Needs: python -m pip install pillow-heif

Run from the repo root:
    python scripts/convert_heic.py "C:\\Users\\redre\\Downloads\\topup"
"""

import sys
from pathlib import Path

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()

if len(sys.argv) != 2 or not Path(sys.argv[1]).is_dir():
    sys.exit('Usage: python scripts/convert_heic.py "path\\to\\folder"')

files = [p for p in Path(sys.argv[1]).rglob("*") if p.suffix.lower() in {".heic", ".heif"}]
converted = 0
for path in files:
    out = path.with_suffix(".jpg")
    if out.exists():
        print(f"Skipped {path.name}: {out.name} already exists")
        continue
    with Image.open(path) as img:
        ImageOps.exif_transpose(img).convert("RGB").save(out, "JPEG", quality=95)
    converted += 1
print(f"Converted {converted} of {len(files)} HEIC files")