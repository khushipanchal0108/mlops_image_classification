# src/model_building.py
"""
Model training for MNIST classification:
- Loads processed features (from feature_engineering.py)
- Trains classifier (default: RandomForest)
- Logs params & metrics with MLflow
- Saves model to model/model.pkl
"""

import os
import sys
import pickle
import logging
import yaml
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

# ----------------------------- Logging setup -----------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    log_file_path = os.path.join(LOG_DIR, "model_building.log")
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(fmt)
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)


# ----------------------------- Utils -----------------------------
def load_params(params_path="params.yaml") -> dict:
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


def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict):
    try:
        clf = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None),
            random_state=params.get("random_state", 42),
            n_jobs=-1,
        )
        logger.debug("Training RandomForest with %d samples", X_train.shape[0])
        clf.fit(X_train, y_train)
        logger.debug("Training completed")
        return clf
    except Exception as e:
        logger.error("Training failed: %s", e)
        raise


def save_model(model, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(model, f)
    logger.debug("Model saved at %s", file_path)


# ----------------------------- Main -----------------------------
def main():
    # Ensure mlruns directory exists to avoid permission issues
    os.makedirs("mlruns", exist_ok=True)
    mlflow.set_tracking_uri(f"file://{os.path.abspath('mlruns')}")
    mlflow.set_experiment("MNIST-Pipeline")

    try:
        params = load_params()["model_building"]
        processed_root = os.path.join("data", "processed")

        train_file = os.path.join(processed_root, "train_features.npz")
        test_file = os.path.join(processed_root, "test_features.npz")

        X_train, y_train = load_numpy_data(train_file)
        X_test, y_test = load_numpy_data(test_file)

        with mlflow.start_run(run_name="train"):
            mlflow.log_params(params)

            clf = train_model(X_train, y_train, params)
            y_pred = clf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            logger.info("Test Accuracy: %.4f", acc)
            mlflow.log_metric("test_accuracy", acc)

            # Save model locally
            model_save_path = os.path.join("model", "model.pkl")
            save_model(clf, model_save_path)

            # Log model to MLflow
            mlflow.sklearn.log_model(clf, "mnist_rf_model")
            logger.info("Model logged to MLflow")

    except Exception as e:
        logger.error("Model building failed: %s", e)
        print(f"Error: {e}")
        raise



