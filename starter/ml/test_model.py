import numpy as np
from sklearn.ensemble import RandomForestClassifier
from starter.ml.model import train_model, compute_model_metrics, inference


def test_train_model():
    X = np.array([[1, 0], [0, 1]])
    y = np.array([0, 1])
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


def test_inference():
    X = np.array([[0, 1], [1, 0]])
    y = np.array([0, 1])
    model = train_model(X, y)
    preds = inference(model, X)
    assert np.array_equal(preds.shape, y.shape)


def test_compute_model_metrics():
    y = np.array([1, 0, 1, 1])
    preds = np.array([1, 0, 0, 1])
    precision, recall, f1 = compute_model_metrics(y, preds)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= f1 <= 1
