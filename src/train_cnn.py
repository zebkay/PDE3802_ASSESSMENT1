"""Transfer learning with MobileNetV3 (Keras) on the office-item dataset.

Reads data/manifest.csv, trains in two phases, and reports results through src.metrics:
  Phase 1  backbone frozen, only the new classification head is trained
  Phase 2  the top part of the backbone is unfrozen and fine-tuned at a low learning rate

Run from the repo root:
    python -m src.train_cnn --variant small          # experiment: report val results
    python -m src.train_cnn --variant large
    python -m src.train_cnn --variant small --final --export-tflite

Rule: use --final only when done experimenting (the test split is scored once).

For deployment: images are fed as RGB, 0-255, resized to
img_size x img_size. OpenCV loads BGR, so convert with cv2.cvtColor(img, cv2.COLOR_BGR2RGB).
The model contains its own input scaling, so do not divide by 255.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quieter TensorFlow logs

import argparse
import csv
import time
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import tensorflow as tf
from tensorflow import keras

from .data_utils import load_config
from .metrics import evaluate_predictions, print_summary

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


#DATA
def read_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_images(rows, class_to_idx, img_size):
    """Load images into memory as RGB uint8 arrays of shape (N, img_size, img_size, 3).

    Loaded with OpenCV so EXIF rotation is applied (same as the HOG baseline).
    Returns (images, integer labels, paths).
    """
    images, labels, paths = [], [], []
    for r in rows:
        img = cv2.imread(r["path"], cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: could not read {r['path']}, skipped")
            continue
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        labels.append(class_to_idx[r["class"]])
        paths.append(r["path"])
    return np.stack(images), np.array(labels, dtype=np.int32), paths


def make_augmenter():
    """Random changes applied to training images only (never to val/test)."""
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.08),
        keras.layers.RandomZoom(0.15),
        keras.layers.RandomBrightness(0.2, value_range=(0, 255)),
        keras.layers.RandomContrast(0.2),
    ], name="augmentation")


def make_dataset(images, labels, batch_size, training, seed, augmenter=None):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        ds = ds.shuffle(len(images), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if training and augmenter is not None:
        ds = ds.map(lambda x, y: (augmenter(tf.cast(x, tf.float32), training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def compute_class_weights(labels):
    """Balanced weights {class_index: weight} so smaller classes count for more."""
    counts = Counter(int(x) for x in labels)
    n, k = len(labels), len(counts)
    return {c: n / (k * cnt) for c, cnt in counts.items()}


#MODEL
def build_model(variant, num_classes, img_size, weights, dropout):
    """MobileNetV3 backbone (+ built-in input scaling) with a new softmax head."""
    ctor = {"small": keras.applications.MobileNetV3Small,
            "large": keras.applications.MobileNetV3Large}[variant]
    base = ctor(input_shape=(img_size, img_size, 3), include_top=False,
                weights=None if weights == "none" else weights,
                pooling="avg", include_preprocessing=True)
    inputs = keras.Input((img_size, img_size, 3))
    x = base(inputs, training=False)  # training=False keeps BatchNorm in inference mode
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs), base


def freeze_backbone(base):
    base.trainable = False


def unfreeze_top(base, fraction):
    """Unfreeze the last `fraction` of backbone layers, but keep BatchNorm layers frozen."""
    base.trainable = True
    cutoff = int(len(base.layers) * (1 - fraction))
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= cutoff and not isinstance(layer, keras.layers.BatchNormalization)


def compile_model(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])


def run_phase(model, name, epochs, lr, train_ds, val_ds, class_weights, patience):
    """Train one phase. Returns a list of per-epoch dicts."""
    if epochs <= 0:
        return []
    compile_model(model, lr)
    print(f"\n--- {name}: up to {epochs} epochs, lr={lr}, "
          f"trainable params={sum(int(np.prod(w.shape)) for w in model.trainable_weights):,} ---")
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                         restore_best_weights=True)
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                     class_weight=class_weights, callbacks=[stop], verbose=2).history
    rows = []
    for i in range(len(hist["loss"])):
        rows.append({"phase": name, "epoch": i + 1, **{k: float(v[i]) for k, v in hist.items()}})
    return rows


#EVALUATION HELPERS
def predict_split(model, images, batch_size):
    probs = model.predict(make_dataset(images, np.zeros(len(images), np.int32), batch_size,
                                       training=False, seed=0), verbose=0)
    return probs.argmax(axis=1), probs.max(axis=1)


def time_inference(model, images, n=30):
    """Average ms per single image (batch of 1) on this computer's CPU/GPU, after warm-up."""
    sample = images[:n].astype(np.float32)
    for x in sample[:3]:
        model(x[None], training=False)
    start = time.perf_counter()
    for x in sample:
        model(x[None], training=False).numpy()
    return (time.perf_counter() - start) / len(sample) * 1000


def save_history(rows, out_dir):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = range(1, len(rows) + 1)
    switch = next((i for i, r in enumerate(rows) if r["phase"] != rows[0]["phase"]), None)
    for ax, metric in zip(axes, ("loss", "accuracy")):
        ax.plot(x, [r[metric] for r in rows], label="train")
        ax.plot(x, [r["val_" + metric] for r in rows], label="val")
        if switch is not None:
            ax.axvline(switch + 0.5, color="grey", linestyle="--", label="start fine-tuning")
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=200)
    plt.close(fig)


def export_tflite(model, base_path, sample):
    """Write float32 and dynamic-range-quantised TFLite files and check they match Keras."""
    expected = model(sample[None].astype(np.float32), training=False).numpy()
    for suffix, quantise in (("", False), ("_dynamic", True)):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        if quantise:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        data = converter.convert()
        path = Path(f"{base_path}{suffix}.tflite")
        path.write_bytes(data)
        interp = tf.lite.Interpreter(model_content=data)
        interp.allocate_tensors()
        interp.set_tensor(interp.get_input_details()[0]["index"], sample[None].astype(np.float32))
        interp.invoke()
        got = interp.get_tensor(interp.get_output_details()[0]["index"])
        print(f"Saved {path} ({len(data) / 1e6:.2f} MB), max |difference| vs Keras output: "
              f"{np.abs(got - expected).max():.4f}")


#MAIN
def main():
    parser = argparse.ArgumentParser(description="MobileNetV3 transfer learning (Keras)")
    parser.add_argument("--config", default="configuration.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--variant", choices=["small", "large"], default="small")
    parser.add_argument("--weights", choices=["imagenet", "none"], default="imagenet",
                        help="'none' trains from scratch (only useful for quick smoke tests)")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-head", type=int, default=10)
    parser.add_argument("--epochs-finetune", type=int, default=20)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-4)
    parser.add_argument("--unfreeze-fraction", type=float, default=0.3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--tag", default="", help="suffix for the results folder, e.g. '_run2'")
    parser.add_argument("--final", action="store_true", help="score the test split and save the model")
    parser.add_argument("--export-tflite", action="store_true", help="with --final: write .tflite files")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    classes = cfg["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    seed = cfg["seed"]
    keras.utils.set_random_seed(seed)
    model_name = f"mobilenetv3_{args.variant}{args.tag}"

    rows = read_manifest(args.manifest or cfg["manifest_path"])
    by_split = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    for s, r in by_split.items():
        print(f"{s}: {len(r)} images")
    if not by_split["train"] or not by_split["val"]:
        raise SystemExit("Need non-empty train and val splits. Run python -m src.make_manifest first.")

    print("\nLoading images...")
    X_train, y_train, _ = load_images(by_split["train"], class_to_idx, args.img_size)
    X_val, y_val, val_paths = load_images(by_split["val"], class_to_idx, args.img_size)

    train_ds = make_dataset(X_train, y_train, args.batch_size, True, seed, make_augmenter())
    val_ds = make_dataset(X_val, y_val, args.batch_size, False, seed)
    class_weights = compute_class_weights(y_train)

    model, base = build_model(args.variant, len(classes), args.img_size, args.weights, args.dropout)
    print(f"Parameters: {model.count_params():,}")

    history = []
    freeze_backbone(base)
    history += run_phase(model, "head", args.epochs_head, args.lr_head,
                         train_ds, val_ds, class_weights, args.patience)
    unfreeze_top(base, args.unfreeze_fraction)
    history += run_phase(model, "finetune", args.epochs_finetune, args.lr_finetune,
                         train_ds, val_ds, class_weights, args.patience)

    out_dir = Path(args.results_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_history(history, out_dir)

    extra = {"params": int(model.count_params()),
             "inference_ms": round(time_inference(model, X_val), 2)}
    names = np.array(classes)
    idx, conf = predict_split(model, X_val, args.batch_size)
    print()
    print_summary(evaluate_predictions(names[y_val], names[idx], classes, args.results_dir,
                                       model_name, split="val", paths=val_paths,
                                       confidences=conf, extra=extra))

    if args.final:
        if not by_split["test"]:
            raise SystemExit("--final needs a non-empty test split.")
        X_test, y_test, test_paths = load_images(by_split["test"], class_to_idx, args.img_size)
        idx, conf = predict_split(model, X_test, args.batch_size)
        print("\n=== FINAL: test split (report these numbers) ===")
        print_summary(evaluate_predictions(names[y_test], names[idx], classes, args.results_dir,
                                           model_name, split="test", paths=test_paths,
                                           confidences=conf, extra=extra))
        Path("models").mkdir(exist_ok=True)
        model.save(f"models/{model_name}.keras")
        print(f"\nSaved models/{model_name}.keras (gitignored; use 'git add -f' for the submitted model)")
        if args.export_tflite:
            export_tflite(model, f"models/{model_name}", X_test[0])
    else:
        print("\nTest split not touched. Re-run with --final when you are done experimenting.")


if __name__ == "__main__":
    main()
