import json
from collections import Counter

EXPORT = r"\Users\redre\OneDrive\Documents\CSSE\YEAR 3\Office Items.v26i.coco"

for split in ("train", "valid", "test"):
    d = json.load(open(f"{EXPORT}/{split}/_annotations.coco.json"))
    names = {c["id"]: c["name"] for c in d["categories"]}
    counts = Counter(names[a["category_id"]] for a in d["annotations"])
    print(split, len(d["images"]), "images:", dict(counts))
