from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import joblib

def load_data():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Save the trained model using joblib
        joblib.dump(model, "model.pkl")

        # Log parameters
        mlflow.log_param("model_type", "LogisticRegression")

        return model

def evaluate_model(model, X_test, y_test):
    with mlflow.start_run():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)

        return accuracy
def load_trained_model():
    # Load the trained model from joblib
    model_path = "model.pkl"
    model = joblib.load(model_path)
    return model