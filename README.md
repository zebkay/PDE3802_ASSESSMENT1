# Office Item Classifier (PDE3802 Assessment 1)

Recognises one office item per image (file or live camera frame) and outputs
the predicted class with a confidence score.

**Classes (9):** mug, bottle, book, notebook, keyboard, mouse, phone, stapler, background

The final model is MobileNetV3 Large (Keras, ImageNet transfer learning). A HOG + SVM
model is the baseline.

## Team
| Name | Student ID | Role |
|------|-----------|-------|
| Nigel Gambiza | M00951628 | software architecture |
| Kay Bugingo | M00971312 | dataset and model training |
| Janet Miti | M00956868 | image preprocessing and testing |

## 1. Installation
```bash
git clone <repo-url>
cd <repo-folder>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Tested on: Windows, Python 3.9.10, TensorFlow TODO-version.

## 2. Getting the data
The images are not stored in Git. Download the dataset zip and unzip it so the images sit in
`data/raw/<class>/`:

TODO-dataset-link

See the dataset card ([`data/dataset_card.md`](data/dataset_card.md)) for counts, sources and
licences. The split for every image is listed in `data/manifest.csv`.

## 3. Usage
Run from the repo root. The final model is `models/mobilenetv3_large_topup.keras`, which is
the default for `--weights`.

```bash
# One image
python -m src.predict --image path/to/photo.jpg

# A folder of images (also writes results/predictions.csv)
python -m src.predict --folder path/to/folder

# Live camera
python -m src.predict --camera 0

# Optional: use another model or confidence threshold
python -m src.predict --camera 0 --weights models/<name>.keras --threshold 0.5
```

In camera mode, place the item inside the box, because only that area is classified. Press
`s` to save a snapshot to `results/snapshots/`, or `q` to quit.

**Expected output:** one line per image in the form `<file>: <label> (<confidence>)`, or the
same label and confidence on the live video. `<label>` is one of:
- a class name, such as `mug`
- `no object`, when the model predicts `background`
- `uncertain`, when confidence is below the threshold (default 0.5)

## 4. Training and evaluation
```bash
python -m src.check_dataset                # validate filenames and images
python -m src.make_manifest                # build data/manifest.csv and the splits
python -m pytest tests                     # run the unit tests

python -m src.baseline_hog_svm --tag _nocolor                  # baseline, validation only
python -m src.train_cnn --variant small --tag _topup           # CNN, validation only
python -m src.train_cnn --variant large --tag _topup

# Final evaluation: use once per chosen model
python -m src.baseline_hog_svm --tag _nocolor --final
python -m src.train_cnn --variant large --tag _topup --final --export-tflite

python -m src.metrics --compare --split test_own
```

`--final` retrains the model and then scores `test` and `test_own`. A retrain gives slightly
different numbers from those below. Results are written to `results/<model>/<split>/`.

## 5. Results
All three models used the same 984 training images.

| Model | Val acc / macro-F1 | Online test acc / macro-F1 | Own photos acc / macro-F1 | Inference (ms, laptop CPU) | Parameters |
|-------|---:|---:|---:|---:|---:|
| HOG + SVM (no colour) | 0.656 / 0.646 | 0.662 / 0.669 | 0.217 / 0.149 | 1.64 | n/a |
| MobileNetV3 Small | 0.901 / 0.897 | not scored | not scored | 97.36 | 944,313 |
| **MobileNetV3 Large (final)** | **0.976 / 0.975** | **0.942 / 0.943** | **0.739 / 0.748** | **138.40** | **3,005,001** |

- **Val (253 images):** online images, used to compare models and stop training.
- **Online test (278 images):** a held-out online batch.
- **Own photos (161 images):** our own photos of objects not used for training, and the best
  estimate of real use.

Large was chosen on validation results, before either test set was scored. Inference times
are from a laptop and are not Raspberry Pi measurements.

Own-photo results for the final model:

| Class | Images | Precision | Recall | F1 |
|-------|---:|---:|---:|---:|
| mug | 25 | 0.676 | 1.000 | 0.806 |
| bottle | 24 | 1.000 | 0.375 | 0.545 |
| book | 19 | 0.750 | 0.789 | 0.769 |
| notebook | 27 | 1.000 | 0.630 | 0.773 |
| keyboard | 7 | 0.875 | 1.000 | 0.933 |
| mouse | 16 | 1.000 | 0.688 | 0.815 |
| phone | 15 | 0.483 | 0.933 | 0.636 |
| stapler | 12 | 0.556 | 0.833 | 0.667 |
| background | 16 | 0.917 | 0.688 | 0.786 |

Confusion matrices:
- `results/mobilenetv3_large_topup/test_own/confusion_matrix.png`
- `results/mobilenetv3_large_topup/test/confusion_matrix.png`

## 6. Error analysis
**Domain shift.** Accuracy falls from 0.942 on online test images to 0.739 on own photos. The
training data is mostly tight online crops, while our photos show more of the scene, with
different cameras, backgrounds and viewpoints.

**Main own-photo mistakes (42 of 161 images):**
- Bottle has the lowest recall: 15 of 24 bottles were wrong (11 as mug, 4 as stapler).
- Notebook: 10 of 27 were wrong (8 as phone, 2 as book).
- Phone and mug are over-predicted: 15 wrong predictions were phone and 12 were mug.

TODO-bottle-findings: say what the 11 bottle-to-mug photos have in common.

**Possible causes (not proven):**
- Mug, phone, book and notebook received training top-up photos, but bottle did not. 32 of the
  42 wrong predictions went to mug, phone or book, so the model may partly link phone-style
  photos to those classes. Top-up images were not removed and re-tested, so this is untested.
- Notebooks seen from above and phones can both look like flat dark rectangles.
- Each own-photo class came from one collector, so class and photographer effects cannot be
  separated.

**Confidence.** 34 of the 42 mistakes had confidence of 0.5 or more, and 12 had 0.9 or more. The
0.5 threshold therefore catches few mistakes, so a robot should not rely on it alone.

**What would help:** adding own-style photos of every class to training (especially bottle),
training on wider views, and calibrating confidence on a separate own-photo set.

**Limits of these results:**
- Own-photo accuracy has a margin of roughly ±7 points (±3 for online test). The true
  uncertainty is larger, because photos of one object are not independent.
- Keyboard has 7 own photos of one keyboard, so its score is not reliable.
- The online test batch had earlier been the validation batch for baseline tuning and a smoke
  test, so its baseline score may be slightly optimistic.

## 7. Troubleshooting
| Problem | Fix |
|---------|-----|
| Camera not opening | Try `--camera 1`, check permissions |
| `The function is not implemented` from `cv2.imshow` | `opencv-python-headless` is installed. Run `pip uninstall -y opencv-python-headless opencv-python`, then `pip install "opencv-python<5"` |
| `module 'cv2' has no attribute 'HOGDescriptor'` | OpenCV 5 is installed. Run `pip uninstall -y opencv-python`, then `pip install "opencv-python<5"` |
| `Error: model ... not found` | Train a model or pass its path with `--weights` |
| `The structure of inputs doesn't match` warning | Harmless, predictions are not affected |
| `--final needs both the test and test_own splits` | Add the own photos and run `python -m src.make_manifest` |
| HEIC photos are skipped | Run `python scripts/convert_heic.py "path/to/folder"` (needs `pip install pillow-heif`) |
| `collector ... is not in online_collectors` | Add the code to `collectors` and to a split list in `configuration.yaml` |

## 8. Project layout
```
configuration.yaml  classes, collector codes and split rules
src/        dataset checks, training, evaluation and prediction code
scripts/    one-off helpers for downloading and preparing data
data/       manifest, dataset card and sources (images downloaded separately)
models/     final trained model
results/    metrics, confusion matrices and training curves
tests/      unit tests
```
