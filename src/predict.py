"""Classify one office item from an image, a folder of images, or a live camera.

Run from the repo root:
    python -m src.predict --image path/to/photo.jpg
    python -m src.predict --folder path/to/folder
    python -m src.predict --camera 0
Add --weights models/<name>.keras to choose a model.

Camera mode classifies only the area inside the green box, because the model was
trained on tight crops of single objects. Press 's' to save a snapshot, 'q' to quit.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

CONFIG_PATH = Path("configuration.yaml")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
GUIDE_BOX_FRACTION = 0.6  # guide box side, as a fraction of the frame's shorter side
SNAPSHOT_DIR = Path("results/snapshots")


def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(weights_path, class_names):
    weights_path = Path(weights_path)
    if not weights_path.exists():
        sys.exit(f"Error: model '{weights_path}' not found. Train one with "
                 "'python -m src.train_cnn --variant small', or choose one with --weights.")
    from tensorflow import keras  # imported here so the error above appears quickly

    model = keras.models.load_model(weights_path, compile=False)
    if model.output_shape[-1] != len(class_names):
        sys.exit(f"Error: model has {model.output_shape[-1]} outputs but "
                 f"configuration.yaml lists {len(class_names)} classes.")
    return model, class_names


def preprocess(bgr_image, size=224):
    """Match training: RGB, 0-255, 224x224. The model scales its own input, so no /255."""
    resized = cv2.resize(bgr_image, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32)[None]  # add batch dim


def predict(model, tensor, labels, threshold=0.5):
    probs = model(tensor, training=False).numpy()[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    label = labels[idx]
    if conf < threshold:
        label = "uncertain"
    elif label == "background":
        label = "no object"
    return label, conf


def run_image(path, model, labels, threshold):
    image = cv2.imread(str(path))
    if image is None:
        sys.exit(f"Error: could not read '{path}' (missing or not an image)")
    label, conf = predict(model, preprocess(image), labels, threshold)
    print(f"{path.name}: {label} ({conf:.2f})")
    return label, conf


def run_folder(folder, model, labels, threshold, out_csv=Path("results/predictions.csv")):
    if not folder.is_dir():
        sys.exit(f"Error: '{folder}' is not a folder")
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        sys.exit(f"Error: no images ({', '.join(IMAGE_EXTS)}) in '{folder}'")

    rows = []
    for path in files:
        image = cv2.imread(str(path))
        if image is None:
            print(f"{path.name}: skipped (unreadable)")
            continue
        label, conf = predict(model, preprocess(image), labels, threshold)
        print(f"{path.name}: {label} ({conf:.2f})")
        rows.append((path.name, label, f"{conf:.4f}"))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "label", "confidence"])
        writer.writerows(rows)
    print(f"Saved {len(rows)} predictions to {out_csv}")


def guide_box(frame, fraction=GUIDE_BOX_FRACTION):
    """Return (x1, y1, x2, y2) of a centred square."""
    h, w = frame.shape[:2]
    side = int(min(h, w) * fraction)
    x1, y1 = (w - side) // 2, (h - side) // 2
    return x1, y1, x1 + side, y1 + side


def run_camera(index, model, labels, threshold):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(f"Error: camera {index} did not open - try --camera 1 "
                 "or check camera permissions")
    print("Place the item inside the box. Press 's' to save a snapshot, 'q' to quit")
    prev = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Error: could not read a frame from the camera")
                break
            x1, y1, x2, y2 = guide_box(frame)
            label, conf = predict(model, preprocess(frame[y1:y2, x1:x2]), labels, threshold)

            now = time.time()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now

            colour = (0, 0, 255) if label == "uncertain" else (0, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(frame, f"{label} ({conf:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Office Item Classifier", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
                out = SNAPSHOT_DIR / f"{int(now)}_{label.replace(' ', '_')}.jpg"
                cv2.imwrite(str(out), frame)
                print(f"Saved {out}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Office item classifier")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="path to one image")
    source.add_argument("--folder", type=Path, help="folder of images")
    source.add_argument("--camera", type=int, help="camera index, e.g. 0")
    parser.add_argument("--weights", type=Path, default=Path("models/mobilenetv3_large_topup.keras"))
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="below this confidence, output 'uncertain'")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_config()
    model, labels = load_model(args.weights, config["classes"])

    if args.image is not None:
        run_image(args.image, model, labels, args.threshold)
    elif args.folder is not None:
        run_folder(args.folder, model, labels, args.threshold)
    else:
        run_camera(args.camera, model, labels, args.threshold)


if __name__ == "__main__":
    main()
