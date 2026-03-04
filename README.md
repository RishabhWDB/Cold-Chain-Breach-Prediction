# Cold Chain Breach Prediction

An end-to-end MLOps pipeline that predicts whether a food shipment has been temperature-compromised during transport, based on IoT sensor data (Synthetic data used is created and used here).

## Project Structure
```
Cold-Chain-Breach-Prediction/
├── src/
│   ├── simulate_data.py    # Synthetic IoT sensor data generation
│   ├── features.py         # Feature engineering from raw sensor logs
│   └── train.py            # Model training with MLflow tracking
├── api/
│   └── main.py             # FastAPI prediction API
├── models/                 # Saved model artifacts (not in git)
├── data/                   # Raw and processed data (not in git)
├── Dockerfile
└── requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
```

## Generate Data & Train Model
```bash
python src/simulate_data.py
python src/train.py
```

View experiments at `http://localhost:5000` after running `mlflow ui`.

## Run API Locally
```bash
uvicorn api.main:app --reload
```

API available at `http://localhost:8000`
Swagger docs at `http://localhost:8000/docs`

## Run with Docker
```bash
docker build -t cold-chain-api .
docker run -p 8000:8000 cold-chain-api
```

## API Endpoints

| Method | Endpoint | Description |
| GET | `/health` | Health check |
| POST | `/predict` | Predict shipment breach |