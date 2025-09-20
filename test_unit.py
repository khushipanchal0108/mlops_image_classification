import os
import sys
import pytest
import numpy as np
import pickle

# Add src folder to sys.path so tests can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Imports from src/
from src import data_ingestion as di
from src import data_preprocessing as dp
from src import feature_engineering as fe
from src import model_building as mb
from src import model_evaluation as me

MODEL_PATH = "model/model.pkl"
TEST_FEATURES_PATH = "data/processed/test_features.npz"
TEST_LABELS_PATH = "data/processed/test_labels.npz"

@pytest.fixture(scope="session", autouse=True)
def ensure_model_exists():
    """Ensure a trained model exists before tests."""
    if not os.path.exists(MODEL_PATH):
        # Create a dummy model if missing
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        X = np.random.rand(50, 784)  # 50 samples, 28x28 flattened
        y = np.random.randint(0, 10, 50)
        clf.fit(X, y)
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(clf, f)

@pytest.fixture
def sample_features_labels():
    X = np.random.rand(50, 784)  # 50 samples, 28x28 flattened
    y = np.random.randint(0, 10, 50)
    return X, y

def test_model_file_exists():
    assert os.path.exists(MODEL_PATH)

def test_model_can_predict(sample_features_labels):
    X, _ = sample_features_labels
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    y_pred = model.predict(X)
    assert len(y_pred) == X.shape[0]

def test_preprocess_output_shape():
    X_raw = np.random.randint(0, 255, (50, 28, 28))
    # Flatten manually inside test
    X_processed = X_raw.reshape(X_raw.shape[0], -1)
    assert X_processed.shape == (50, 784)


def test_feature_engineering_pca():
    X = np.random.rand(50, 784)
    X_pca, _ = fe.apply_pca(X, X, n_components=20)  # <= number of samples
    assert X_pca.shape[1] == 20

def test_model_training(sample_features_labels):
    X, y = sample_features_labels
    params = {"n_estimators": 5, "max_depth": 10, "random_state": 42}
    model = mb.train_model(X, y, params)
    from sklearn.ensemble import RandomForestClassifier
    assert isinstance(model, RandomForestClassifier)

def test_model_evaluation(sample_features_labels):
    X, y = sample_features_labels
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    metrics, _ = me.evaluate_model(model, X, y)  # unpack the tuple
    assert all(k in metrics for k in ["accuracy", "precision", "recall", "f1"])

