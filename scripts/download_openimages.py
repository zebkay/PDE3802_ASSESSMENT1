"""Top-up download of Open Images V7 samples for the classes that came up short."""
import fiftyone.zoo as foz

CLASSES = ["Book", "Stapler", "Mobile phone", "Computer keyboard", "Bottle"]
PER_CLASS = 500

for cls in CLASSES:
    foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections"],
        classes=[cls],
        max_samples=PER_CLASS,
        only_matching=True,
        shuffle=True,
        seed=42,
        dataset_name=f"oi_{cls.lower().replace(' ', '_')}",
        drop_existing_dataset=True,
    )
    print(f"Done: {cls}")