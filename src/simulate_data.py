import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

PRODUCT_BASELINES = {
    "fish": {"temp" : 2, "threshold" : 4},
    "dairy": {"temp" : 4, "threshold" : 7},
    "produce": {"temp" : 8, "threshold" : 12},
    "pharma": {"temp" : 5, "threshold" : 8}
}

def generate_shipment(shipment_id):
    product_type = random.choice(["fish", "dairy", "produce", "pharma"])
    cooling_system = random.choice(["standard", "premium"])


    start_time = datetime(2024,1,1,8,0) + timedelta(hours=random.randint(0,23))
    timestamps = [start_time + timedelta(minutes=15*i) for i in range(60)]


    baseline_temp = PRODUCT_BASELINES[product_type]["temp"]
    inside_temps = [baseline_temp+ np.random.normal(0,0.5) for _ in range(60)]
    outside_temps = [np.random.uniform(25,40) for _ in range(60)]
    humidities = [np.random.uniform(60,90) for _ in range(60)]
    door_opens = [1 if np.random.random() < 0.05 else 0 for _ in range(60)]

    df = pd.DataFrame({
        "shipment_id" : shipment_id,
        "timestamp" : timestamps,
        "product_type" : product_type,
        "cooling_system" : cooling_system,
        "inside_temp" : inside_temps,
        "outside_temp" : outside_temps,
        "humidity" : humidities,
        "door_opens" : door_opens

    })

    return df

if __name__ == "__main__":
    df = generate_shipment(1)
    print(df.head(10))
    print(df.shape)



