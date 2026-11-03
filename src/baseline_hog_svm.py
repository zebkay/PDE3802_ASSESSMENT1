"""Classical baseline: HOG features + SVM classifier.

Reads data/manifest.csv (made by src.make_manifest), extracts HOG features,
tunes the SVM on the VALIDATION split, and reports results through src.metrics.

Run from the repo root:
    python -m src.baseline_hog_svm            # tune on train/val, report val results
    python -m src.baseline_hog_svm --final    # also evaluate on TEST once and save the model

Rule: use --final only when you are done experimenting. The test split should be
scored once per model, otherwise you are tuning to the test set.
"""

import argparse
import csv
import time
from functools import lru_cache
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data_utils import load_config
from .metrics import evaluate_predictions, print_summary

MODEL_NAME = "hog_svm"
COLOR_BINS = (8, 4, 4)  # HSV histogram bins (H, S, V) when --color-hist is used


#FEATURES
@lru_cache(maxsize=None)
def _hog(size):
    """HOG descriptor for a size x size window (cached: creating it is slow)."""
    # args: window, block, block stride, cell, orientation bins
    return cv2.HOGDescriptor((size, size), (32, 32), (16, 16), (16, 16), 9)


def extract_features(img_bgr, size=128, use_color=False):
    """Turn one BGR image (NumPy array) into a 1-D feature vector.

    HOG captures edge/shape structure on the grayscale image. Optionally an HSV
    colour histogram is appended, because HOG ignores colour.
    Note: the image is resized to size x size, which squashes the aspect ratio.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    parts = [_hog(size).compute(gray).ravel()]
    if use_color:
        small = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, list(COLOR_BINS),
                            [0, 180, 0, 256, 0, 256]).ravel()
        parts.append(hist / (hist.sum() + 1e-9))
    return np.concatenate(parts).astype(np.float32)


#DATA LOADING
def read_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_features(rows, size, use_color):
    """Read the images listed in rows and return (X, y, paths). Unreadable files are skipped."""
    X, y, paths = [], [], []
    for r in rows:
        img = cv2.imread(r["path"], cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: could not read {r['path']}, skipped")
            continue
        X.append(extract_features(img, size, use_color))
        y.append(r["class"])
        paths.append(r["path"])
    return np.array(X), y, paths


#MODEL
def candidate_params():
    """Hyperparameter combinations to try (kept small so it runs in seconds)."""
    grid = [{"kernel": "linear", "C": c} for c in (0.1, 1, 10)]
    grid += [{"kernel": "rbf", "C": c, "gamma": "scale"} for c in (1, 10, 100)]
    return grid


def build_model(params, seed, probability=False):
    # class_weight='balanced' stops bigger classes from dominating if the class counts differ
    svm = SVC(class_weight="balanced", probability=probability, random_state=seed, **params)
    return make_pipeline(StandardScaler(), svm)


def tune(X_train, y_train, X_val, y_val, seed):
    """Fit every candidate on train, score on val (macro-F1). Returns (best_params, all_results)."""
    results = []
    for params in candidate_params():
        model = build_model(params, seed)
        model.fit(X_train, y_train)
        score = f1_score(y_val, model.predict(X_val), average="macro", zero_division=0)
        results.append({"params": params, "val_macro_f1": float(score)})
        print(f"  {str(params):<50} val macro-F1 = {score:.3f}")
    best = max(results, key=lambda r: r["val_macro_f1"])
    return best["params"], results


def predict_with_confidence(model, X):
    """Predicted class names plus the (Platt-scaled, approximate) probability of that class."""
    pred = model.predict(X)
    proba = model.predict_proba(X)
    col = {c: i for i, c in enumerate(model.classes_)}
    conf = [float(proba[i, col[p]]) for i, p in enumerate(pred)]
    return list(pred), conf


def time_inference(model, rows, size, use_color, n=50):
    """Average ms per image for feature extraction + prediction (excludes file decoding)."""
    imgs = []
    for r in rows:
        img = cv2.imread(r["path"], cv2.IMREAD_COLOR)
        if img is not None:
            imgs.append(img)
        if len(imgs) == n:
            break
    if not imgs:
        return None
    start = time.perf_counter()
    for img in imgs:
        model.predict(extract_features(img, size, use_color)[None, :])
    return (time.perf_counter() - start) / len(imgs) * 1000


#MAIN
def main():
    parser = argparse.ArgumentParser(description="HOG + SVM baseline")
    parser.add_argument("--config", default="configuration.yaml")
    parser.add_argument("--manifest", default=None, help="defaults to manifest_path in config")
    parser.add_argument("--img-size", type=int, default=128, help="multiple of 16, at least 32")
    parser.add_argument("--color-hist", action="store_true", help="append an HSV colour histogram")
    parser.add_argument("--final", action="store_true", help="also score the test split and save the model")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    if args.img_size < 32 or args.img_size % 16:
        parser.error("--img-size must be a multiple of 16 and at least 32")

    cfg = load_config(args.config)
    classes = cfg["classes"]
    seed = cfg["seed"]
    rows = read_manifest(args.manifest or cfg["manifest_path"])
    by_split = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    for s, r in by_split.items():
        print(f"{s}: {len(r)} images")
    if not by_split["train"] or not by_split["val"]:
        raise SystemExit("Need non-empty train and val splits. Run python -m src.make_manifest first.")

    print("\nExtracting features...")
    X_train, y_train, _ = load_features(by_split["train"], args.img_size, args.color_hist)
    X_val, y_val, val_paths = load_features(by_split["val"], args.img_size, args.color_hist)
    print(f"Feature length: {X_train.shape[1]}")
    missing = sorted(set(classes) - set(y_train))
    if missing:
        print(f"Warning: no training images for {missing}; they can never be predicted")

    print("\nTuning on the validation split:")
    best_params, tuning = tune(X_train, y_train, X_val, y_val, seed)
    print(f"Best: {best_params}")

    out_dir = Path(args.results_dir) / MODEL_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "tuning.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["params", "val_macro_f1"])
        for t in tuning:
            w.writerow([t["params"], f"{t['val_macro_f1']:.4f}"])

    # Refit the chosen setting (with probabilities, for confidence scores) on TRAIN only.
    model = build_model(best_params, seed, probability=True)
    model.fit(X_train, y_train)
    extra = {"n_features": int(X_train.shape[1])}
    ms = time_inference(model, by_split["val"], args.img_size, args.color_hist)
    if ms is not None:
        extra["inference_ms"] = round(ms, 2)

    pred, conf = predict_with_confidence(model, X_val)
    print()
    print_summary(evaluate_predictions(y_val, pred, classes, args.results_dir, MODEL_NAME,
                                       split="val", paths=val_paths, confidences=conf, extra=extra))

    if args.final:
        if not by_split["test"]:
            raise SystemExit("--final needs a non-empty test split.")
        X_test, y_test, test_paths = load_features(by_split["test"], args.img_size, args.color_hist)
        pred, conf = predict_with_confidence(model, X_test)
        print("\n=== FINAL: test split (report these numbers) ===")
        print_summary(evaluate_predictions(y_test, pred, classes, args.results_dir, MODEL_NAME,
                                           split="test", paths=test_paths, confidences=conf, extra=extra))
        Path("models").mkdir(exist_ok=True)
        joblib.dump({"model": model, "classes": classes, "img_size": args.img_size,
                     "use_color": args.color_hist}, f"models/{MODEL_NAME}.joblib")
        print(f"\nSaved models/{MODEL_NAME}.joblib (gitignored; use 'git add -f' for the final submitted model)")
    else:
        print("\nTest split not touched. Re-run with --final when you are done experimenting.")


if __name__ == "__main__":
    main()
