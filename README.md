# IoT Anomaly Detection Service

This project implements a small anomaly detection service for simulated factory sensor data. It is built as a REST API using **FastAPI** and exposes a simple machine learning model that assigns an anomaly score to incoming sensor measurements.

The setup is intentionally lightweight and focuses on how a predictive model can be integrated and served as part of an application, rather than on model complexity.

---

## What the service does

- Loads a historical sensor dataset (CSV)
- Trains a basic anomaly detection model (Isolation Forest)
- Exposes endpoints to:
  - Check service health
  - Inspect basic dataset statistics
  - Submit single sensor events for anomaly prediction
  - Simulate a real-time sensor data stream using Server-Sent Events (SSE)

Each sensor event represents one produced item in a manufacturing process.

---

## API overview

- `GET /health`  
  Simple health check to verify the service is running.

- `POST /predict`  
  Accepts one set of sensor readings and returns an anomaly score and a boolean flag.

- `GET /stats`  
  Returns basic statistics about the dataset used to train the model.

- `GET /stream-simulate`  
  Streams simulated sensor events in real time, including anomaly scores.

Swagger UI is available at:  
`http://localhost:8000/docs`

Note: the streaming endpoint is not interactive in Swagger and should be tested via a browser or `curl`.

## Running the project

1. Create and activate the virtual environment
   ```bash
   uv venv
   ```
   ```bash
   .venv/Scripts/activate
   ```
2. Install dependencies:
   ```bash
    uv sync
   ```
3. Run the project:
      ```bash
   uv run -m src.main
   ```
4. Open the API docs:
   ```bash
   http://localhost:8000/docs
   ```