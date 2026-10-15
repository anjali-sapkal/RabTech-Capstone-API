
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "champion_model.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="RabTech ML Prediction API",
    description="REST API for a trained classification model",
    version="1.0.0"
)


class PredictionRequest(BaseModel):
    features: list[float] = Field(
        min_length=30,
        max_length=30,
        description="Exactly 30 numeric features"
    )


class PredictionResponse(BaseModel):
    prediction: int
    probabilities: list[float]
    model_name: str


@app.get("/")
def home():
    return {"message": "RabTech ML API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        values = np.array(request.features, dtype=float).reshape(1, -1)

        if not np.isfinite(values).all():
            raise HTTPException(
                status_code=422,
                detail="Features must contain finite numeric values"
            )

        prediction = int(model.predict(values)[0])

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(values)[0].tolist()
        else:
            probabilities = []

        return PredictionResponse(
            prediction=prediction,
            probabilities=probabilities,
            model_name=type(model).__name__
        )

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )
