
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

CONFIG_PATH = Path("configuration.yaml")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def load_config(path=CONFIG_PATH):
    """Return the config dict (class names live in config['classes'])."""
    # TODO: open the yaml file and return yaml.safe_load(...)
    raise NotImplementedError


def load_model(weights_path, num_classes):
    """Load MobileNetV3-Small.

    Until Kay's trained weights exist, use the ImageNet-pretrained model from
    torchvision so the app can be built and tested now.
    """
    # TODO: torchvision.models.mobilenet_v3_small(weights=...)
    # TODO: if weights_path exists -> replace classifier[3] with
    #       nn.Linear(in_features, num_classes) and load_state_dict(...)
    # TODO: model.eval() and return it
    raise NotImplementedError


def preprocess(bgr_image, size=224):
    """OpenCV BGR uint8 image -> normalised 1x3xHxW tensor.

    Must match the preprocessing used in training (ask Kay).
    """
    # TODO: BGR -> RGB, resize to size x size, scale to 0-1,
    #       normalise with ImageNet mean/std, HWC -> CHW, add batch dim
    raise NotImplementedError


def predict(model, tensor, class_names, threshold=0.5):
    """Return (label, confidence). Label is 'uncertain' below threshold."""
    # TODO: torch.no_grad(), softmax over logits, take the max
    raise NotImplementedError


def run_image(path, model, class_names, threshold):
    # TODO: cv2.imread (check for None -> clear error message), predict, print
    raise NotImplementedError


def run_folder(folder, model, class_names, threshold):
    # TODO: loop over files with IMAGE_EXTS, print one line per file,
    #       optionally write results to results/predictions.csv
    raise NotImplementedError


def run_camera(index, model, class_names, threshold):
    # TODO: cv2.VideoCapture(index); if not opened -> helpful error
    # TODO: loop: read frame, predict, cv2.putText label + confidence + FPS,
    #       cv2.imshow; quit on 'q'; release camera and destroy windows
    raise NotImplementedError


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
    # TODO: load config + model, then dispatch to run_image / run_folder /
    #       run_camera depending on which argument was given
    raise NotImplementedError


if __name__ == "__main__":
    main()