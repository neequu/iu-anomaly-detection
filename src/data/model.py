
import pathlib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

DATA_PATH = pathlib.Path("data.csv")
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
]


if not DATA_PATH.exists():
    raise RuntimeError(
        "data.csv not found"
    )

df = pd.read_csv(DATA_PATH)

X = df[FEATURE_COLUMNS].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=RANDOM_STATE,
)
model.fit(X_scaled)

