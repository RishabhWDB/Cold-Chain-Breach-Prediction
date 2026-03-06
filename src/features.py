import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
print(sys.path[0])
import pandas as pd
from src.simulate_data import PRODUCT_BASELINES 

def aggregate_shipment(df):
    product_type = df["product_type"].iloc[0]
    cooling_system = df["cooling_system"].iloc[0]
    is_breached = df["is_breached"].iloc[0]

    inside_temp_mean = df["inside_temp"].mean()
    inside_temp_max = df["inside_temp"].max()
    inside_temp_std = df["inside_temp"].std()

    outside_temp_mean = df["outside_temp"].mean()
    outside_temp_max = df["outside_temp"].max()
    outside_temp_std = df["outside_temp"].std()

    humidity_mean = df["humidity"].mean()
    humidity_max = df["humidity"].max()
    humidity_std = df["humidity"].std()
    
    door_open_count = df["door_opens"].sum()

    threshold = PRODUCT_BASELINES[product_type]["threshold"]
    violations = df["inside_temp"] > threshold
    minutes_above_threshold = violations.sum() * 15

    max_consecutive_violations = 0
    current = 0
    for v in violations:
        if v:
            current+=1
            max_consecutive_violations = max(max_consecutive_violations, current)
        else:
            current = 0

    temp_humidity_interaction = inside_temp_mean * humidity_mean

    return{
        "product_type" : product_type,
        "cooling_system" : cooling_system,
        "is_breached" : is_breached,
        "inside_temp_mean" : inside_temp_mean,
        "inside_temp_max" : inside_temp_max,
        "inside_temp_std" : inside_temp_std,
        "outside_temp_mean" : outside_temp_mean,
        "outside_temp_max" : outside_temp_max,
        "outside_temp_std" : outside_temp_std,
        "humidity_mean" : humidity_mean,
        "humidity_max" : humidity_max,
        "humidity_std" : humidity_std,
        "door_open_count" : door_open_count,
        "minutes_above_threshold" : minutes_above_threshold,
        "max_consecutive_violations" : max_consecutive_violations,
        "temp_humidity_interaction" : temp_humidity_interaction
    }

def build_features(raw_df):
    rows = []
    for shipment_id, group in raw_df.groupby("shipment_id"):
        row = aggregate_shipment(group)
        rows.append(row)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    raw_df = pd.read_csv("data/raw_shipments.csv")
    features_df = build_features(raw_df)
    print(features_df.shape)
    print(features_df.head())