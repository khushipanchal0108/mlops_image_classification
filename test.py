import os
import json

def test_metrics_file_exists():
    """Check if metrics.json file is created after training."""
    assert os.path.exists("reports/metrics.json"), "❌ metrics.json file is missing!"

def test_metrics_content():
    """Check if metrics.json has required keys."""
    with open("reports/metrics.json", "r") as f:
        metrics = json.load(f)

    required_keys = ["accuracy", "precision", "recall", "f1"]
    for key in required_keys:
        assert key in metrics, f"❌ Missing key: {key} in metrics.json"

def test_accuracy_threshold():
    """Ensure accuracy is at least 0.90"""
    with open("reports/metrics.json", "r") as f:
        metrics = json.load(f)

    accuracy = metrics.get("accuracy", 0)
    assert accuracy >= 0.90, f"❌ Accuracy too low: {accuracy}"
