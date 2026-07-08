"""
=========================================================
Rice Variety Classification - Prediction Module
=========================================================

Author  : Vansh Dadeech
Version : 2.0
"""

import logging
import joblib
from pathlib import Path
import pandas as pd
import numpy as np


# =========================================================
# Logging
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# =========================================================
# Paths
# =========================================================

MODEL_PATH = Path("models/best_model.pkl")
SCALER_PATH = Path("models/scaler.pkl")


# =========================================================
# Load Model
# =========================================================

try:
    with open(MODEL_PATH, "rb") as file:
        model = joblib.load(file)

    with open(SCALER_PATH, "rb") as file:
        scaler = joblib.load(file)

    logger.info("Model and Scaler Loaded Successfully.")

except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise


# =========================================================
# Prediction Function
# =========================================================

def predict_rice(
    area: float,
    perimeter: float,
    major_axis_length: float,
    minor_axis_length: float,
    eccentricity: float,
    convex_area: float,
    extent: float
) -> dict:

    sample = pd.DataFrame(
    [[
        area,
        perimeter,
        major_axis_length,
        minor_axis_length,
        eccentricity,
        convex_area,
        extent
    ]],
    columns=[
        "Area",
        "Perimeter",
        "Major_Axis_Length",
        "Minor_Axis_Length",
        "Eccentricity",
        "Convex_Area",
        "Extent"
    ]
)

    sample = scaler.transform(sample)

    prediction = int(model.predict(sample)[0])

    label = "Cammeo" if prediction == 0 else "Osmancik"

    confidence = None

    if hasattr(model, "predict_proba"):
        confidence = float(
            np.max(model.predict_proba(sample))
        )

    return {
        "prediction": label,
        "confidence": round(confidence * 100, 2) if confidence else None
    }


# =========================================================
# Test
# =========================================================

if __name__ == "__main__":

    result = predict_rice(
        15231,
        525.57,
        229.75,
        85.09,
        0.92,
        15617,
        0.57
    )

    print(result)