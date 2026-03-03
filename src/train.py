import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
from src.features import build_features

def load_features():
    df = pd.read_csv('data/raw_shipments.csv')

    df = build_features(df)

    y = df["is_breached"]

    df = pd.get_dummies(df, columns = ["product_type", "cooling_system"])

    X = df.drop(columns = ["is_breached"])

    return X,y

if __name__ == "__main__":
    X, y = load_features()
    print(X.shape)
    print(y.shape)
    print(X.columns.tolist())