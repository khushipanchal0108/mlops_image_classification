# src/model_evaluation.py
"""
Model evaluation for MNIST classification:
- Loads model.pkl and test data (.npz)
- Computes metrics (accuracy, precision, recall, f1)
- Saves metrics.json, classification report, and confusion matrix
- Logs metrics locally and for DVC
"""

import os
import sys
import json
import pickle
import logging
import yaml
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# ----------------------------- Logging setup -----------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    log_file_path = os.path.join(LOG_DIR, "model_evaluation.log")
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(fmt)
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)

# ----------------------------- Utils -----------------------------
def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    return params

def load_model(file_path: str):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def load_numpy_data(file_path: str):
    data = np.load(file_path)
    return data["X"], data["y"]

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = clf.predict(X_test)

    metrics_dict = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro"),
        "recall": recall_score(y_test, y_pred, average="macro"),
        "f1": f1_score(y_test, y_pred, average="macro"),
    }

    logger.info("Evaluation Metrics: %s", metrics_dict)
    return metrics_dict, y_pred

def save_metrics(metrics: dict, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(metrics, f, indent=4)

def save_classification_report(y_test, y_pred, file_path: str):
    report = classification_report(y_test, y_pred, output_dict=True)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(report, f, indent=4)

def save_confusion_matrix(y_test, y_pred, file_path: str):
    cm = confusion_matrix(y_test, y_pred)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.save(file_path, cm)

# ----------------------------- Main -----------------------------
def main():
    try:
        params = load_params()
        clf = load_model("model/model.pkl")
        X_test, y_test = load_numpy_data("data/processed/test_features.npz")

        metrics, y_pred = evaluate_model(clf, X_test, y_test)

        # Save metrics and reports
        save_metrics(metrics, "reports/metrics.json")
        save_classification_report(y_test, y_pred, "reports/classification_report.json")
        save_confusion_matrix(y_test, y_pred, "reports/confusion_matrix.npy")

        logger.info("Evaluation completed successfully. Metrics and reports saved locally.")

    except Exception as e:
        logger.error("Evaluation failed: %s", e)
        raise

if __name__ == "__main__":
    main()
