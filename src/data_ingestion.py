# src/data_ingestion.py

import os
import sys
import json
import yaml
import math
import shutil
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml


# ----------------------------- Logging setup -----------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

# Avoid duplicate handlers when script is re-run
if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)

    log_file_path = os.path.join(LOG_DIR, "data_ingestion.log")
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s -")
    ch.setFormatter(fmt)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)


# ----------------------------- Utils -----------------------------
def load_params(params_path: str = "params.yaml") -> dict:
    try:
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        logger.debug("Parameters loaded from %s", params_path)
        return params
    except Exception as e:
        logger.error("Failed to load params.yaml: %s", e)
        raise


def make_dir(path: str, overwrite: bool = False) -> None:
    if overwrite and os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def fetch_mnist_openml(dataset_name: str = "mnist_784") -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        X: (N, 784) uint8-like array
        y: (N,) labels as integers 0..9
    """
    try:
        logger.info("Fetching dataset '%s' from OpenML...", dataset_name)
        data = fetch_openml(dataset_name, version=1, as_frame=False)
        X = data.data  # (N, 784) float64
        y = data.target  # strings '0'..'9'
        y = y.astype(int)
        # convert to uint8 0..255 if needed later; we keep float for now
        logger.info("Fetched MNIST with shape X=%s, y=%s", X.shape, y.shape)
        return X, y
    except Exception as e:
        logger.error("Error fetching MNIST: %s", e)
        raise


def optional_subsample_per_class(
    X: np.ndarray,
    y: np.ndarray,
    max_per_class: Optional[int],
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if max_per_class is None:
        return X, y

    logger.info("Subsampling up to %d samples per class...", max_per_class)
    rng = np.random.default_rng(random_state)
    keep_indices = []

    classes = np.unique(y)
    for c in classes:
        idx = np.where(y == c)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        keep_indices.append(idx)

    keep_indices = np.concatenate(keep_indices)
    # Shuffle overall to mix classes
    rng.shuffle(keep_indices)

    X_sub = X[keep_indices]
    y_sub = y[keep_indices]
    logger.debug("After subsample: X=%s, y=%s", X_sub.shape, y_sub.shape)
    return X_sub, y_sub


def to_uint8_images(X: np.ndarray) -> np.ndarray:
    """
    X: (N, 784) in 0..255-ish floats
    Returns (N, 28, 28) uint8 images.
    """
    X = np.clip(X, 0, 255)
    X = X.reshape(-1, 28, 28)
    X_uint8 = X.astype(np.uint8)
    return X_uint8


def pad_image(img: np.ndarray, pad: int) -> np.ndarray:
    if pad <= 0:
        return img
    h, w = img.shape
    padded = np.zeros((h + 2 * pad, w + 2 * pad), dtype=img.dtype)
    padded[pad : pad + h, pad : pad + w] = img
    return padded


def save_split_as_images(
    X: np.ndarray,
    y: np.ndarray,
    out_dir: str,
    image_format: str = "png",
    padding: int = 0,
) -> pd.DataFrame:
    """
    Saves images into class-named subfolders and returns a DataFrame with (filepath, label).
    """
    make_dir(out_dir, overwrite=False)
    records = []
    counts = {int(c): 0 for c in np.unique(y)}
    for i, (img, label) in enumerate(zip(X, y)):
        img_arr = img
        if padding > 0:
            img_arr = pad_image(img_arr, padding)
        pil_img = Image.fromarray(img_arr)

        class_dir = os.path.join(out_dir, str(int(label)))
        os.makedirs(class_dir, exist_ok=True)
        idx = counts[int(label)]
        filename = f"img_{idx:05d}.{image_format.lower()}"
        fpath = os.path.join(class_dir, filename)
        pil_img.save(fpath)
        records.append((fpath, int(label)))
        counts[int(label)] += 1

    df = pd.DataFrame(records, columns=["filepath", "label"])
    return df


# ----------------------------- Main flow -----------------------------
def main():
    params = load_params("params.yaml")
    p = params.get("data_ingestion", {})

    dataset_name = p.get("dataset_name", "mnist_784")
    data_dir = p.get("data_dir", "data")
    raw_dirname = p.get("raw_dirname", "raw")
    train_dirname = p.get("train_dirname", "train")
    test_dirname = p.get("test_dirname", "test")
    test_size = float(p.get("test_size", 0.2))
    random_state = int(p.get("random_state", 42))
    stratify_flag = bool(p.get("stratify", True))
    max_per_class = p.get("max_samples_per_class", None)
    image_format = p.get("image_format", "png")
    image_padding = int(p.get("image_padding", 0))
    ensure_uint8 = bool(p.get("ensure_uint8", True))
    overwrite = bool(p.get("overwrite", False))
    shuffle_before_split = bool(p.get("shuffle_before_split", True))

    raw_root = os.path.join(data_dir, raw_dirname)
    train_root = os.path.join(raw_root, train_dirname)
    test_root = os.path.join(raw_root, test_dirname)

    # Prepare directories
    make_dir(raw_root, overwrite=overwrite)
    make_dir(train_root, overwrite=overwrite)
    make_dir(test_root, overwrite=overwrite)

    # Fetch data
    X, y = fetch_mnist_openml(dataset_name=dataset_name)

    # Optional shuffle before split (keeps class balance if stratify is on)
    if shuffle_before_split:
        rng = np.random.default_rng(random_state)
        idx = np.arange(len(y))
        rng.shuffle(idx)
        X = X[idx]
        y = y[idx]

    # Optional subsample per class
    X, y = optional_subsample_per_class(X, y, max_per_class, random_state)

    # Convert to proper image arrays
    X_img = to_uint8_images(X) if ensure_uint8 else X.reshape(-1, 28, 28)

    # Split
    stratify = y if stratify_flag else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_img, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    logger.info(
        "Split done. Train: %s, Test: %s", X_train.shape[0], X_test.shape[0]
    )

    # Save as images + CSVs
    train_df = save_split_as_images(
        X_train, y_train, train_root, image_format=image_format, padding=image_padding
    )
    test_df = save_split_as_images(
        X_test, y_test, test_root, image_format=image_format, padding=image_padding
    )

    # Save label CSVs for convenience
    train_csv = os.path.join(raw_root, "train_labels.csv")
    test_csv = os.path.join(raw_root, "test_labels.csv")
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    # Also save a small summary JSON
    summary = {
        "dataset": dataset_name,
        "num_classes": int(len(np.unique(y))),
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "image_shape": list(train_df.shape),  # dataframe shape
        "params_used": p,
    }
    with open(os.path.join(raw_root, "ingestion_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved train/test images and labels under '%s'", raw_root)
    logger.info("Data ingestion completed successfully.")


if __name__ == "__main__":
    main()
