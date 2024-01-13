import pytest
from data_and_models_tests import load_data, train_model, evaluate_model

def test_model_accuracy():
    # Model training and evaluation test
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    assert accuracy > 0.9, f"Model accuracy is {accuracy}, expected > 0.9"
