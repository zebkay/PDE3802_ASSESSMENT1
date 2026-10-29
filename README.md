# Office Item Classifier (PDE3802 Assessment 1, Part B)

Recognises one office item per image (file or live camera frame) and outputs
the predicted class with a confidence score.

**Classes (8):** mug, bottle, book, notebook, keyboard, mouse, phone, stapler
<!-- Update if the group changes the class list -->

## Team
| Name | Student ID | Role |
|------|-----------|-------|
|  Nigel Gambiza|M00951628|software architecture|
|  Kay Bugingo|M00971312|dataset and model training| 
|  Janet Miti |M00956868|image preprocessing and testing|
## 1. Installation
```bash
git clone <repo-url>
cd pde3802-office-item-classifier
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Tested on: <!-- OS, Python version -->

## 2. Getting the data
See [`data/README.md`](data/README.md) and the dataset card
([`data/dataset_card.md`](data/dataset_card.md)).

## 3. Usage
```bash
# Classify an image file
python -m src.predict --image data/sample/mug_01.jpg

# Live camera
python -m src.predict --camera 0
```
**Expected output:** <!-- paste a real example here -->

## 4. Training and evaluation
```bash
python -m src.train
python -m src.evaluate
```

## 5. Results
| Model | Accuracy | Macro-F1 | Inference time (ms) |
|-------|----------|----------|---------------------|
| HOG + SVM (baseline) | | | |
| CNN (transfer learning) | | | |

Confusion matrix: `results/confusion_matrix.png`

## 6. Error analysis
<!-- Which classes are confused, why, and what would fix it. -->

## 7. Troubleshooting
| Problem | Fix |
|---------|-----|
| Camera not opening | Try `--camera 1`, check permissions |
| | |

## 8. Project layout
```
src/        training, evaluation and prediction code
data/       dataset card, sample images, download instructions
models/     trained model (final release only)
results/    metrics, confusion matrix, plots
docs/       figures used in the report
tests/      tests
```
