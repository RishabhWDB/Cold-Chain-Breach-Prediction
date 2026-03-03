import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
from lightgbm import LGBMClassifier
import mlflow
import mlflow.lightgbm
from sklearn.metrics import accuracy_score, f1_score, classification_report
from src.features import build_features

def load_features():
    df = pd.read_csv('data/raw_shipments.csv')
    df = build_features(df)

    y = df["is_breached"].astype(int)
    #print(y.dtype)
    #print(y.shape)

    df = pd.get_dummies(df, columns = ["product_type", "cooling_system"])
    X = df.drop(columns = ["is_breached"])

    return X,y


def train_model(X_train, y_train, X_test, y_test, params):
    mlflow.set_experiment("cold-chain-breach-prediction")

    with mlflow.start_run(run_name="lightgbm"):


        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)

        for key, value in params.items():
            mlflow.log_param(key, value)
            
        acc, f1, report = evaluate_model(model, X_test, y_test)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)
        mlflow.lightgbm.log_model(model, "model")

        return model
    


def evaluate_model(model, X_test, y_test):
    res = model.predict(X_test)

    acc = accuracy_score(y_test,res)
    f1 = f1_score(y_test, res)
    report = classification_report(y_test, res)

    return acc, f1, report


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    X, y = load_features()
    X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.2, random_state = 42)
    #print(y_train.dtypes)
    #print(type(y_train))

    params = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 6,
        "random_state": 42
        }
    
    model = train_model(X_train, y_train, X_test, y_test, params)

    acc, f1, report = evaluate_model(model, X_test, y_test)

    print(acc)
    print(f1)
    print(report)

