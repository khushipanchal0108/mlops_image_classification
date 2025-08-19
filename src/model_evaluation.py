# src/model_evaluation.py
"""
Model evaluation for MNIST classification:
- Loads model.pkl and test data (.npz)
- Computes metrics (accuracy, precision, recall, f1)
- Logs to MLflow & DVC Live
- Saves metrics.json
"""

import os
import sys
import json
import pickle
import logging
import yaml
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from dvclive import Live

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


# ----------------------------- Main -----------------------------
def main():
    import mlflow
    mlflow.set_tracking_uri(f"file://{os.path.abspath('mlruns')}")
    mlflow.set_experiment("MNIST-Pipeline")

    try:
        with mlflow.start_run(run_name="eval", nested=True):
            params = load_params()
            clf = load_model("./model/model.pkl")
            X_test, y_test = load_numpy_data("./data/processed/test_features.npz")

            metrics, y_pred = evaluate_model(clf, X_test, y_test)

            # Save metrics.json
            save_metrics(metrics, "reports/metrics.json")

            # Log metrics to DVC Live
            with Live(save_dvc_exp=True) as live:
                for k, v in metrics.items():
                    live.log_metric(k, v)
                live.log_params(params)

            # Log to MLflow
            mlflow.log_metrics(metrics)
            mlflow.log_artifact("reports/metrics.json")

            # Extra: log classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            report_path = "reports/classification_report.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=4)
            mlflow.log_artifact(report_path)

            # Extra: confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            cm_path = "reports/confusion_matrix.npy"
            np.save(cm_path, cm)
            mlflow.log_artifact(cm_path)

            logger.info("Evaluation completed and logged to MLflow & DVC")

    except Exception as e:
        logger.error("Evaluation failed: %s", e)
        raise


if __name__ == "__main__":
    main()
