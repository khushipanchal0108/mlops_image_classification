# src/data_preprocessing.py
"""
Preprocess MNIST images:
- Load filepaths + labels from train/test CSVs (created in ingestion stage)
- Normalize pixel values
- Reshape to (H, W, C)
- Optionally apply augmentations
- Save processed arrays into data/interim/
"""

import os
import sys
import logging
import yaml
import numpy as np
import pandas as pd
from PIL import Image

# ----------------------------- Logging setup -----------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    log_file_path = os.path.join(LOG_DIR, "data_preprocessing.log")
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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


def load_images_from_df(df: pd.DataFrame, img_shape: tuple, normalize: bool, add_channel: bool) -> np.ndarray:
    """
    Load images from filepaths into numpy array.
    """
    images = []
    for fp in df["filepath"]:
        img = Image.open(fp).convert("L")  # grayscale
        img = img.resize(img_shape)        # resize if needed
        arr = np.array(img, dtype=np.float32)
        if normalize:
            arr = arr / 255.0
        if add_channel:
            arr = np.expand_dims(arr, axis=-1)  # (H, W, 1)
        images.append(arr)
    return np.array(images)


def save_numpy_arrays(X: np.ndarray, y: np.ndarray, out_path: str):
    np.savez_compressed(out_path, X=X, y=y)
    logger.debug("Saved processed arrays to %s", out_path)


# ----------------------------- Main -----------------------------
def main():
    try:
        params = load_params("params.yaml")
        p = params.get("data_preprocessing", {})

        img_height = p.get("img_height", 28)
        img_width = p.get("img_width", 28)
        normalize = bool(p.get("normalize", True))
        add_channel_dim = bool(p.get("add_channel_dim", True))

        # Paths
        raw_root = os.path.join("data", "raw")
        interim_root = os.path.join("data", "interim")
        os.makedirs(interim_root, exist_ok=True)

        train_csv = os.path.join(raw_root, "train_labels.csv")
        test_csv = os.path.join(raw_root, "test_labels.csv")

        logger.info("Loading label CSVs...")
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)

        logger.info("Preprocessing train set...")
        X_train = load_images_from_df(train_df, (img_width, img_height), normalize, add_channel_dim)
        y_train = train_df["label"].to_numpy()

        logger.info("Preprocessing test set...")
        X_test = load_images_from_df(test_df, (img_width, img_height), normalize, add_channel_dim)
        y_test = test_df["label"].to_numpy()

        # Save processed arrays
        save_numpy_arrays(X_train, y_train, os.path.join(interim_root, "train_processed.npz"))
        save_numpy_arrays(X_test, y_test, os.path.join(interim_root, "test_processed.npz"))

        logger.info("Data preprocessing completed successfully.")

    except Exception as e:
        logger.error("Data preprocessing failed: %s", e)
        raise


if __name__ == "__main__":
    main()
