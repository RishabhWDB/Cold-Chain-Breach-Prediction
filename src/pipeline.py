import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import joblib
from prefect import flow, task
from sklearn.model_selection import train_test_split
from src.train import load_features, optimize, train_model
from src.simulate_data import generate_dataset

@task
def task_generate_data():
    generate_dataset()

@task
def task_load_features():
    return load_features()

@task
def task_optimize(X_train, y_train, X_test, y_test):
    return optimize(X_train, y_train, X_test, y_test)

@task 
def task_train(X_train, y_train, X_test, y_test, params):
    return train_model(X_train, y_train, X_test, y_test, params)

@flow(name = "cold-chain-training-pipeline")
def training_pipeline():
    task_generate_data()

    X, y = task_load_features()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                         random_state = 42)

    best_params = task_optimize(X_train, y_train, X_test, y_test)

    model = task_train(X_train, y_train, X_test, y_test, best_params)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    print("Pipeline complete, model saved.")

if __name__ == "__main__":
    training_pipeline()