import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import pytest
from src.features import aggregate_shipment, build_features
from src.simulate_data import PRODUCT_BASELINES

EXPECTED_COLUMNS = [
    "inside_temp_mean", "inside_temp_max", "inside_temp_std",
    "outside_temp_mean", "outside_temp_max", "outside_temp_std",
    "humidity_mean", "humidity_max", "humidity_std",
    "door_open_count", "minutes_above_threshold",
    "max_consecutive_violations", "temp_humidity_interaction",
    "product_type", "cooling_system", "is_breached"
]

def make_fake_shipment(product_type = "dairy", cooling_system = "standard",
                       inside_temp = 4.0, is_breached = False):
    return pd.DataFrame({
        "shipment_id": 1,
        "product_type": product_type,
        "cooling_system": cooling_system,
        "inside_temp": [inside_temp] * 60,
        "outside_temp": [30.0] * 60,
        "humidity": [70.0] * 60,
        "door_opens": [0] * 60,
        "is_breached": is_breached
    })

def test_aggregate_columns_present():
    df = make_fake_shipment()
    result = aggregate_shipment(df)
    for col in EXPECTED_COLUMNS:
        assert col in result, f"Missing column: {col}"

def test_no_breachzero_violations():
    df = make_fake_shipment(product_type = "dairy", inside_temp = 4.0)
    result = aggregate_shipment(df)
    assert result["minutes_above_threshold"] == 0

def test_breach_has_violations():
    df = make_fake_shipment(product_type = "dairy", inside_temp = 10.0, is_breached = True)
    result = aggregate_shipment(df)
    assert result["minutes_above_threshold"] == 60 * 15

def test_build_features_shape():
    raw_df = pd.read_csv("data/raw_shipments.csv")
    features_df = build_features(raw_df)
    assert features_df.shape[1] == 16
    assert features_df.shape[0] > 0
