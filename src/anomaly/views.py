from typing import Dict

import numpy as np
from fastapi import APIRouter, HTTPException
import pandas as pd
from .models import AnomalyResponse, SensorInput
from src.data.model import model, scaler, FEATURE_COLUMNS, df
import asyncio
from fastapi.responses import StreamingResponse
import json

router = APIRouter(tags=["Anomaly detection"])

def compute_anomaly_score(
    air_temperature: float,
    process_temperature: float,
    rotational_speed: float,
    torque: float,
) -> tuple[float, bool]:
    features = np.array([[
        air_temperature,
        process_temperature,
        rotational_speed,
        torque,
    ]])

    features_scaled = scaler.transform(features)

    raw_score = model.decision_function(features_scaled)[0]
    anomaly_score = float(1.0 - (raw_score + 0.5))
    is_anomaly = anomaly_score > 0.6

    return round(anomaly_score, 4), is_anomaly

@router.get(
    "/health",
    summary="Service health check",
    description="Lightweight endpoint used to verify that the service is running and reachable."
)
def health_check() -> Dict[str, str]:
    return {"status": "ok"}

@router.get(
    "/stats",
    summary="Dataset overview statistics",
    description=(
        "Provides basic descriptive statistics of the dataset used to train "
        "the anomaly detection model. Intended for inspection and monitoring."
    ),
)
def dataset_stats() -> Dict[str, float]:
    return {
        "samples": int(len(df)),
        "features": len(FEATURE_COLUMNS),
        "mean_air_temp": float(df["Air temperature [K]"].mean()),
        "mean_process_temp": float(df["Process temperature [K]"].mean()),
    }



@router.post(
    "/predict",
    response_model=AnomalyResponse,
    summary="Predict anomaly score for a single sensor event",
    description=(
        "Receives one set of sensor measurements representing a produced item. "
        "The data is preprocessed and passed to the anomaly detection model, "
        "which returns an anomaly score and a boolean flag."
    ),
)
def predict(input_data: SensorInput) -> AnomalyResponse:
    try:
        anomaly_score, is_anomaly = compute_anomaly_score(
            air_temperature=input_data.air_temperature,
            process_temperature=input_data.process_temperature,
            rotational_speed=input_data.rotational_speed,
            torque=input_data.torque,
        )

        return AnomalyResponse(
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@router.get(
    "/stream-simulate",
    summary="Simulate a real-time sensor data stream",
    description=(
        "Streams sensor events one by one using Server-Sent Events (SSE). "
        "Each event represents a produced item and includes the calculated "
        "anomaly score. This endpoint is meant to simulate a live IoT data stream."
    ),
)
async def stream_simulate() -> StreamingResponse:
    LIMIT = 100
    sample_data = df[FEATURE_COLUMNS].head(LIMIT)

    async def generate():
        for _, row in sample_data.iterrows():
            sensor_data = {
                "air_temperature": float(row["Air temperature [K]"]),
                "process_temperature": float(row["Process temperature [K]"]),
                "rotational_speed": float(row["Rotational speed [rpm]"]),
                "torque": float(row["Torque [Nm]"]),
                "timestamp": pd.Timestamp.now().isoformat(),
            }

            anomaly_score, is_anomaly = compute_anomaly_score(
                air_temperature=sensor_data["air_temperature"],
                process_temperature=sensor_data["process_temperature"],
                rotational_speed=sensor_data["rotational_speed"],
                torque=sensor_data["torque"],
            )

            yield f"data: {json.dumps({
                'sensor_data': sensor_data,
                'anomaly_score': anomaly_score,
                'is_anomaly': is_anomaly
            })}\n\n"

            await asyncio.sleep(0.5)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )

