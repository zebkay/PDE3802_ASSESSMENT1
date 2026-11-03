from pathlib import Path

root = Path.home() / "fiftyone" / "open-images-v7" / "train"
print("images:", len(list((root / "data").glob("*.jpg"))))

wanted = ["Mug", "Bottle", "Book", "Computer keyboard", "Computer mouse",
          "Mobile phone", "Stapler", "Notebook"]
print("\nclasses found:")
for line in open(root / "metadata" / "classes.csv", encoding="utf-8"):
    mid, name = line.strip().split(",", 1)
    if name in wanted:
        print(" ", mid, name)

print("\nfirst 3 lines of detections.csv:")
with open(root / "labels" / "detections.csv", encoding="utf-8") as f:
    for _ in range(3):
        print(" ", f.readline().strip())