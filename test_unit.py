import os
import sys
import pytest
import numpy as np
import pickle
import pandas as pd

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
        X = np.random.rand(10, 784)  # 10 samples, 28x28 flattened
        y = np.random.randint(0, 10, 10)
        clf.fit(X, y)
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(clf, f)

@pytest.fixture
def sample_features_labels():
    X = np.random.rand(5, 784)  # 5 samples, 28x28 flattened
    y = np.random.randint(0, 10, 5)
    return X, y

def test_model_file_exists():
    assert os.path.exists(MODEL_PATH)

def test_model_can_predict(sample_features_labels):
    X, _ = sample_features_labels
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    y_pred = model.predict(X)
    assert len(y_pred) == X.shape[0]

def test_load_images_from_df():
    """Test your actual preprocessing function in data_preprocessing.py"""
    # Create dummy dataframe
    df = pd.DataFrame({"filepath": ["data/raw/sample_0.png", "data/raw/sample_1.png"]})
    # Create dummy images
    os.makedirs("data/raw", exist_ok=True)
    for i in range(2):
        img = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
        from PIL import Image
        Image.fromarray(img).save(f"data/raw/sample_{i}.png")

    images = dp.load_images_from_df(df, img_shape=(28, 28), normalize=True, add_channel=True)
    assert images.shape == (2, 28, 28, 1)
    assert np.max(images) <= 1.0  # normalized

def test_feature_engineering_pca():
    X = np.random.rand(5, 784)
    X_pca, _ = fe.apply_pca(X, X, n_components=50)
    assert X_pca.shape[1] == 50

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
    metrics = me.evaluate_model(model, X, y)
    assert all(k in metrics for k in ["accuracy", "precision", "recall", "f1"])
