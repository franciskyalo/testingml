# model/train.py
import mlflow
import mlflow.sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def load_data():
    iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, n_estimators=100):
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def log_to_mlflow(model, accuracy, n_estimators=100):
    with mlflow.start_run():
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")
        mlflow.set_tracking_uri(uri="http://localhost:5000")
        
        # Save the model using joblib for local deployment
        joblib.dump(model, "model/model.pkl")

def train_and_log():
    X_train, X_test, y_train, y_test = load_data()

    # Train a simple RandomForestClassifier
    model = train_model(X_train, y_train)

    # Make predictions and calculate accuracy of the model
    accuracy = evaluate_model(model, X_test, y_test)

    # Log the model and metrics using MLflow
    log_to_mlflow(model, accuracy)
