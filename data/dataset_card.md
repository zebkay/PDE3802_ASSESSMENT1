# Dataset Card: Office Items

## Summary
1,676 images of 9 classes for single-object classification: 8 office items plus `background`
("nothing to act on"). Most images are online crops, and our own photos make up the
training top-up and the `test_own` set.

## Classes and counts
| Class | Train: online | Train: top-up | Val | Test (online) | Test_own (objects) | Total |
|-------|---:|---:|---:|---:|---:|---:|
| mug | 97 | 36 | 28 | 25 | 25 (3) | 211 |
| bottle | 93 | 0 | 22 | 35 | 24 (3) | 174 |
| book | 90 | 36 | 28 | 32 | 19 (2) | 205 |
| notebook | 91 | 36 | 26 | 33 | 27 (3) | 213 |
| keyboard | 89 | 0 | 22 | 39 | 7 (1) | 157 |
| mouse | 79 | 0 | 32 | 29 | 16 (2) | 156 |
| phone | 90 | 36 | 31 | 29 | 15 (2) | 201 |
| stapler | 124 | 0 | 25 | 32 | 12 (1) | 193 |
| background | 87 | 0 | 39 | 24 | 16 (3 scenes) | 166 |
| **Total** | **840** | **144** | **253** | **278** | **161** | **1,676** |

Online images are grouped into source batches, so the batch count does not equal a number of
physical objects.

| Source | Images | Used for |
|--------|---:|---|
| Open Images V7 crops (`oi`) | 939 | mug, bottle, book, keyboard, mouse, phone, stapler |
| Roboflow Universe crops (`rf`) | 282 | notebook, stapler |
| Places365 scenes (`pl`) | 150 | background |
| Own training top-up (`tu`) | 144 | mug, book, notebook, phone |
| Own test photos (`jm`, `kb`, `ng`) | 161 | all classes |

## Collection
- **Online crops:** cut from labelled bounding boxes with 10% padding, using
  `scripts/prepare_online_data.py`. Crops longer than 640 px were reduced.
- **Background:** wide office-type scenes from the Places365 validation set, selected by hand
  to exclude target objects, people and poor-quality images.
- **Devices used:** TODO-devices
- **Locations / backgrounds / lighting:** TODO-locations
- **Image resolution and format:** own photos are JPG, with the longest side at most 1280 px.
  Places365 images are 256x256. Online crops vary, and many are under 224 px on the shortest
  side. All images are resized to 224x224 for the CNN.
- **Who collected which own test photos:**
  - Janet Miti (`jm`): book, mouse, phone
  - Kay Bugingo (`kb`): mug, bottle, background
  - Nigel Gambiza (`ng`): notebook, keyboard, stapler
- **Training top-up (`tu`):** 12 photos of each of 3 objects per class, across varied angles,
  on surfaces and held in a hand. It was added after camera checks showed failures at angles,
  and before either test set was scored.

## Splits
Splits are set by source, not by random shuffling (`src/make_manifest.py`).

| Split | Contents | Purpose |
|-------|----------|---------|
| train | Online batches 1-3 and `tu` photos | Training |
| val | Online batch 4 | Choosing models and stopping training |
| test | Online batch 5 | Held-out online score, scored once with `--final` |
| test_own | `jm`, `kb` and `ng` photos | Real-use score, scored once with `--final` |

How leakage was avoided:
- Online batches are assigned by hashing the photographer (Open Images), filename prefix
  (Roboflow) or filename (Places365), so one source stays in one batch.
- No object ID appears in more than one split, and the manifest script checks this.
- Top-up photos use physical objects different from those in `test_own`. TODO-confirm-objects
- Own test photos are never used for training.

## Labelling
- Online labels come from the source datasets' bounding-box annotations. Open Images boxes
  marked as groups, depictions or inside views were dropped, as were boxes under 100 px.
- Crops were reviewed by hand on contact sheets.
- Own photos were labelled by class when taken, and the label is stored in the filename and
  folder.
- `src/check_dataset.py` checks filenames, class folders, readable images, GPS data and exact
  duplicates. The final dataset reported no errors.

## Ethics and privacy
- Own photos contain no faces or personal information. Some top-up photos show a hand.
  TODO-confirm-no-faces
- `scripts/rename_photos.py` removes metadata, including GPS location, from own photos.
- Wide background images with people were excluded during selection.
- Online crops are tightly cropped objects, but were reviewed for label quality rather than
  specifically for people.

## Known limitations and biases
- Online crops differ from robot-camera views, so scores on own photos are much lower.
- Top-up photos cover only 4 of the 9 classes, which may bias the model towards those classes.
- Own-photo classes each came from one collector, so class and photographer are linked.
- Keyboard (7 photos, 1 object) and stapler (12 photos, 1 object) are thin in `test_own`.
- 132 of 181 stapler crops come from Roboflow image sequences of few objects, so stapler has
  less variety. Some Roboflow notebook images may show pens lying on notebooks.
- Many online crops are small, and images are squashed to square, which distorts shape.
- Background training images are wide rooms only, so close-up empty desks are under-represented.
- The background class cannot cover every unseen object.

## Licence / sources
- **Open Images V7:** 939 crops, each from a Flickr image under CC BY 2.0. Author, licence and
  URL for each are in `data/sources.csv`.
- **Roboflow Universe:** 282 crops under CC BY 4.0, from Workspace1, "office-items",
  https://universe.roboflow.com/workspace1-1gbmx/office-items-plugt
- **Places365:** 150 images. Copyright belongs to the image owners. These images are not
  listed in `data/sources.csv`. Cite: Zhou, B., Lapedriza, A., Khosla, A., Oliva, A. and
  Torralba, A. (2018) 'Places: A 10 Million Image Database for Scene Recognition', *IEEE
  Transactions on Pattern Analysis and Machine Intelligence*, 40(6), pp. 1452-1464.
  doi:10.1109/TPAMI.2017.2723009.
- **Own photographs:** taken by the team.
