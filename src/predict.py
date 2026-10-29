
import argparse
import csv
import sys
import time
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

CONFIG_PATH = Path("configuration.yaml")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_config(path=CONFIG_PATH):
     with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(weights_path, class_names):
     
    weights_path = Path(weights_path)
    if weights_path.exists():
        model = mobilenet_v3_small(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, len(class_names))
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state)
        labels = class_names
    else:
        print(f"[warning] {weights_path} not found - using ImageNet weights "
              "(labels will be ImageNet classes, not ours)")
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = mobilenet_v3_small(weights=weights)
        labels = weights.meta["categories"]
    model.eval()
    return model, labels


def preprocess(bgr_image, size=224):
    
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))          # HWC -> CHW
    return torch.from_numpy(x).unsqueeze(0)  # add batch dim


def predict(model, tensor, labels, threshold=0.5):
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    conf, idx = torch.max(probs, dim=0)
    conf = float(conf)
    label = labels[int(idx)]
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


def run_camera(index, model, labels, threshold):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(f"Error: camera {index} did not open - try --camera 1 "
                 "or check camera permissions")
    print("Press 'q' to quit")
    prev = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Error: could not read a frame from the camera")
                break
            label, conf = predict(model, preprocess(frame), labels, threshold)

            now = time.time()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now

            colour = (0, 0, 255) if label == "uncertain" else (0, 200, 0)
            cv2.putText(frame, f"{label} ({conf:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Office Item Classifier", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Office item classifier")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="path to one image")
    source.add_argument("--folder", type=Path, help="folder of images")
    source.add_argument("--camera", type=int, help="camera index, e.g. 0")
    parser.add_argument("--weights", type=Path, default=Path("models/best.pt"))
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