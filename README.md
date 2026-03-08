# Cold Chain Breach Prediction

An end-to-end MLOps pipeline that predicts whether a refrigerated food shipment has been temperature-compromised during transport, based on simulated IoT sensor data.

## Stack

- Data: Synthetic IoT sensor simulation with realistic noise and breach patterns
- Model: LightGBM classifier with Optuna hyperparameter tuning
- Tracking: MLflow experiment tracking and model registry
- Serving: FastAPI REST API
- Orchestration: Prefect pipeline
- Versioning: DVC for data versioning
- Testing: pytest with GitHub Actions CI
- Deployment: Docker and docker-compose

## Project Structure
```
Cold-Chain-Breach-Prediction/
├── src/
│   ├── simulate_data.py    # IoT sensor data simulation
│   ├── features.py         # Feature engineering from raw sensor logs
│   ├── train.py            # Model training with MLflow and Optuna
│   └── pipeline.py         # Prefect orchestration flow
├── api/
│   └── main.py             # FastAPI prediction API
├── tests/
│   └── test_features.py    # pytest test suite
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions CI
├── data/                   # Raw data (tracked by DVC, not git)
├── models/                 # Saved model artifacts (not in git)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
```

## Generate Data and Train
```bash
# Generate synthetic shipment data
python src/simulate_data.py

# Train with Optuna tuning and MLflow tracking
python src/train.py

# Or run the full Prefect pipeline
python src/pipeline.py
```

View MLflow experiments:
```bash
mlflow ui --port 5001
```
Open http://localhost:5001

## Run API Locally
```bash
uvicorn api.main:app --reload
```

- API: http://localhost:8000
- Swagger docs: http://localhost:8000/docs

## Run with Docker
```bash
# API only
docker build -t cold-chain-api .
docker run -p 8000:8000 cold-chain-api

# API + MLflow together
docker-compose up --build
```

## Run Tests
```bash
pytest tests/
```

Tests run automatically on every push to main via GitHub Actions.

## Data Versioning with DVC
```bash
# After regenerating data
dvc add data/raw_shipments.csv
dvc push
git add data/raw_shipments.csv.dvc
git commit -m "data: update shipments"

# To fetch data after cloning
dvc pull
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check |
| POST | /predict | Predict shipment breach |
| GET | /docs | Swagger UI |

## Problem

Each shipment is a sequence of 60 sensor readings taken every 15 minutes. Features are aggregated from the raw time series including mean, max, and standard deviation of temperature and humidity, cumulative minutes above safe threshold, longest streak of consecutive violations, and door open count. The model predicts whether a shipment is compromised before it arrives at its destination.