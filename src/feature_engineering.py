# src/feature_engineering.py
"""
Feature engineering for MNIST images:
- Load preprocessed .npz files (train/test)
- Flatten or keep 2D based on params
- Optionally apply PCA for dimensionality reduction
- Save processed features to data/processed/
"""

import os
import sys
import logging
import yaml
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# ----------------------------- Logging setup -----------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger("feature_engineering")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    log_file_path = os.path.join(LOG_DIR, "feature_engineering.log")
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


def load_numpy_data(file_path: str):
    try:
        data = np.load(file_path)
        X, y = data["X"], data["y"]
        logger.debug("Loaded data from %s", file_path)
        return X, y
    except Exception as e:
        logger.error("Failed to load numpy data: %s", e)
        raise


def flatten_images(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1)


def apply_pca(X_train: np.ndarray, X_test: np.ndarray, n_components: int):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    logger.debug("Applied PCA with %d components", n_components)
    return X_train_pca, X_test_pca


def save_features(X: np.ndarray, y: np.ndarray, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, X=X, y=y)
    logger.debug("Saved features to %s", out_path)


# ----------------------------- Main -----------------------------
def main():
    try:
        params = load_params("params.yaml")
        p = params.get("feature_engineering", {})

        flatten = bool(p.get("flatten", True))
        apply_dim_reduction = bool(p.get("apply_pca", False))
        n_components = int(p.get("pca_components", 50))

        interim_root = os.path.join("data", "interim")
        processed_root = os.path.join("data", "processed")
        os.makedirs(processed_root, exist_ok=True)

        train_npz = os.path.join(interim_root, "train_processed.npz")
        test_npz = os.path.join(interim_root, "test_processed.npz")

        X_train, y_train = load_numpy_data(train_npz)
        X_test, y_test = load_numpy_data(test_npz)

        if flatten:
            logger.info("Flattening images...")
            X_train = flatten_images(X_train)
            X_test = flatten_images(X_test)

        if apply_dim_reduction:
            logger.info("Applying PCA...")
            X_train, X_test = apply_pca(X_train, X_test, n_components)

        # Save engineered features
        save_features(X_train, y_train, os.path.join(processed_root, "train_features.npz"))
        save_features(X_test, y_test, os.path.join(processed_root, "test_features.npz"))

        logger.info("Feature engineering completed successfully.")

    except Exception as e:
        logger.error("Feature engineering failed: %s", e)
        raise


if __name__ == "__main__":
    main()
