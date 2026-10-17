"""Evaluation shared by every model (HOG+SVM, MobileNetV3, ...).

Any model only has to hand over true labels and predicted labels (class names).
This module then writes, to results/<model_name>/<split>/:

    metrics.json                    accuracy, macro-F1, per-class scores, top confusions
    per_class_report.csv            precision / recall / F1 / support per class
    confusion_matrix.png            raw counts
    confusion_matrix_normalised.png rows sum to 1 (recall per class)
    misclassified.csv               every wrong prediction (most confident mistakes first)

Usage from a training/eval script:
    from src.metrics import evaluate_predictions
    evaluate_predictions(y_true, y_pred, classes, "results", "hog_svm", split="test",
                         paths=paths, confidences=conf, extra={"inference_ms": 12.3})

Compare all models that have been evaluated on a split:
    python -m src.metrics --compare --split test
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

LOW_SUPPORT = 10  # classes with fewer images than this in a split give noisy scores


def _to_indices(labels, class_to_idx, what):
    """Converts class names to numbers and rejects unknown names"""
    out = []
    for lab in labels:
        if lab not in class_to_idx:
            raise ValueError(f"Unknown label '{lab}' in {what}; expected one of {list(class_to_idx)}")
        out.append(class_to_idx[lab])
    return np.array(out, dtype=int)


def _plot_confusion(matrix, classes, title, path, normalised):
    """draws a confusion matrix image"""
    k = len(classes)
    size = max(6, 0.75 * k + 2)
    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1 if normalised else None)
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    thresh = (1 if normalised else matrix.max()) / 2.0 if matrix.size else 0
    for i in range(k):
        for j in range(k):
            val = matrix[i, j]
            text = f"{val:.2f}" if normalised else f"{int(val)}"
            if normalised and val == 0:
                text = ""
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="white" if val > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def evaluate_predictions(y_true, y_pred, classes, out_dir, model_name, split="test",
                         paths=None, confidences=None, extra=None):
    """Score predictions and write all result files. Returns the summary dict.

    y_true, y_pred : sequences of class NAMES (same order as paths/confidences)
    classes        : full ordered list of class names (defines matrix order)
    paths          : optional image paths, used in misclassified.csv
    confidences    : optional confidence of each prediction (0-1)
    extra          : optional dict of additional facts to store (e.g. inference_ms, params)
    """
    classes = list(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    yt = _to_indices(y_true, class_to_idx, "y_true")
    yp = _to_indices(y_pred, class_to_idx, "y_pred")
    n = len(yt)
    if len(yp) != n:
        raise ValueError(f"y_true has {n} items but y_pred has {len(yp)}")
    if n == 0:
        raise ValueError("No predictions to evaluate")
    for name, seq in (("paths", paths), ("confidences", confidences)):
        if seq is not None and len(seq) != n:
            raise ValueError(f"{name} has {len(seq)} items, expected {n}")

    k = len(classes)
    labels = list(range(k))
    cm = confusion_matrix(yt, yp, labels=labels)
    prec, rec, f1, support = precision_recall_fscore_support(yt, yp, labels=labels, zero_division=0)

    present = support > 0  # macro average only over classes that exist in this split
    accuracy = float((yt == yp).mean())
    macro_f1 = float(f1[present].mean())
    weighted_f1 = float(np.average(f1, weights=support)) if support.sum() else 0.0

    # Top confusions (off-diagonal cells)
    pairs = [(int(cm[i, j]), i, j) for i in range(k) for j in range(k) if i != j and cm[i, j] > 0]
    pairs.sort(reverse=True)
    top_confusions = [
        {"true": classes[i], "predicted": classes[j], "count": c} for c, i, j in pairs[:5]
    ]

    summary = {
        "model": model_name,
        "split": split,
        "n_images": int(n),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": {
            classes[i]: {
                "precision": float(prec[i]), "recall": float(rec[i]),
                "f1": float(f1[i]), "support": int(support[i]),
            } for i in labels
        },
        "low_support_classes": [classes[i] for i in labels if 0 < support[i] < LOW_SUPPORT],
        "classes_missing_from_split": [classes[i] for i in labels if support[i] == 0],
        "top_confusions": top_confusions,
    }
    if confidences is not None:
        conf = np.asarray(confidences, dtype=float)
        correct = yt == yp
        summary["mean_confidence_correct"] = float(conf[correct].mean()) if correct.any() else None
        summary["mean_confidence_incorrect"] = float(conf[~correct].mean()) if (~correct).any() else None
    if extra:
        summary["extra"] = extra

    out = Path(out_dir) / model_name / split
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(out / "per_class_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "support", "precision", "recall", "f1"])
        for i in labels:
            w.writerow([classes[i], int(support[i]), f"{prec[i]:.4f}", f"{rec[i]:.4f}", f"{f1[i]:.4f}"])

    with open(out / "misclassified.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "true_class", "predicted_class", "confidence"])
        wrong = [i for i in range(n) if yt[i] != yp[i]]
        if confidences is not None:
            wrong.sort(key=lambda i: -float(confidences[i]))  # confident mistakes are most informative
        for i in wrong:
            w.writerow([
                paths[i] if paths is not None else i,
                classes[yt[i]], classes[yp[i]],
                f"{float(confidences[i]):.4f}" if confidences is not None else "",
            ])

    title = f"{model_name} ({split}, n={n}, acc={accuracy:.3f}, macro-F1={macro_f1:.3f})"
    _plot_confusion(cm, classes, title, out / "confusion_matrix.png", normalised=False)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)
    _plot_confusion(cm_norm, classes, title + " - normalised by row",
                    out / "confusion_matrix_normalised.png", normalised=True)

    return summary


def print_summary(summary):
    """Prints the results to the screen"""
    print(f"{summary['model']} on {summary['split']}: n={summary['n_images']}  "
          f"accuracy={summary['accuracy']:.3f}  macro-F1={summary['macro_f1']:.3f}")
    print(f"{'class':<12}{'support':>8}{'prec':>8}{'recall':>8}{'f1':>8}")
    for cls, m in summary["per_class"].items():
        print(f"{cls:<12}{m['support']:>8}{m['precision']:>8.3f}{m['recall']:>8.3f}{m['f1']:>8.3f}")
    if summary["low_support_classes"]:
        print(f"Note: fewer than {LOW_SUPPORT} images in this split for: "
              f"{', '.join(summary['low_support_classes'])} (scores are noisy)")
    for c in summary["top_confusions"]:
        print(f"Confusion: true {c['true']} -> predicted {c['predicted']}: {c['count']}")


def build_comparison(results_dir="results", split="test"):
    """Collect results/<model>/<split>/metrics.json into comparison.csv and comparison.md."""
    rows = []
    for mpath in sorted(Path(results_dir).glob(f"*/{split}/metrics.json")):
        with open(mpath, encoding="utf-8") as f:
            s = json.load(f)
        row = {"model": s["model"], "accuracy": s["accuracy"], "macro_f1": s["macro_f1"],
               "n_images": s["n_images"]}
        row.update(s.get("extra", {}))
        rows.append(row)
    if not rows:
        raise SystemExit(f"No metrics.json files found under {results_dir}/*/{split}/")

    columns = list(dict.fromkeys(k for r in rows for k in r))
    out_csv = Path(results_dir) / f"comparison_{split}.csv"
    out_md = Path(results_dir) / f"comparison_{split}.md"

    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else ("" if v is None else str(v))

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in rows:
            w.writerow([fmt(r.get(c)) for c in columns])
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("|" + "|".join("---" for _ in columns) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(fmt(r.get(c)) for c in columns) + " |\n")
    print(out_md.read_text(encoding="utf-8"))
    print(f"Wrote {out_csv} and {out_md}")


def main():
    parser = argparse.ArgumentParser(description="Compare evaluated models.")
    parser.add_argument("--compare", action="store_true", help="build the comparison table")
    parser.add_argument("--split", default="test")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    if args.compare:
        build_comparison(args.results_dir, args.split)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()