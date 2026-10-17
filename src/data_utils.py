"""configuration loading and filename parsing.

Filename convention:  <class>_<objectID>_<index>.jpg
    class     lowercase letters only, e.g. mug
    objectID  2-3 letter collector code + 2 digits, e.g. ak01
              (for the 'background' class this identifies the scene)
    index     4 digits starting at 0001
Example: mug_ab01_0001.jpg
"""

import re
from dataclasses import dataclass
from pathlib import Path
import yaml

# Add a pattern that defines a valid filename
FILENAME_RE = re.compile(
    r"^(?P<cls>[a-z]+)_(?P<obj>(?P<collector>[a-z]{2,3})\d{2})_(?P<idx>\d{4})\.(?:jpg|jpeg)$"
)

# Anything with these extensions is treated as an image and must follow the convention.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


# Creating a class where the data cannot be changed after creation
@dataclass(frozen=True) # 'frozen=True' makes the objects attributes read-only after creation
class Sample: # A record for one image
    path: Path
    cls: str
    object_id: str
    collector: str
    index: int

    @property
    def object_key(self) -> str:
        """Unique per physical object: the same ID in two classes is two different objects."""
        return f"{self.cls}_{self.object_id}"

# Read the 'configuration.yaml' file
def load_config(path: str = "configuration.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Split filename into its parts or return None if it breaks the convention
def parse_filename(name: str):
    """Return a dict of the filename parts, or None if it breaks the convention."""
    m = FILENAME_RE.match(name)
    if m is None:
        return None
    return {
        "cls": m.group("cls"),
        "object_id": m.group("obj"),
        "collector": m.group("collector"),
        "index": int(m.group("idx")),
    }


# List every image under data/raw
def iter_image_files(raw_dir):
    """Yield every image-like file under raw_dir, in a stable order."""
    for p in sorted(Path(raw_dir).rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p
