from pydantic import BaseModel, Field


class SensorInput(BaseModel):
    air_temperature: float = Field(..., examples=[298.1])
    process_temperature: float = Field(..., examples=[308.6])
    rotational_speed: float = Field(..., examples=[1550.0])
    torque: float = Field(..., examples=[42.8])

class AnomalyResponse(BaseModel):
    anomaly_score: float
    is_anomaly: bool
