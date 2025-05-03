from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import json
import requests
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

app = FastAPI(
    title="Nutrition Plan API",
    description="API for breastfeeding nutrition recommendations",
    version="1.0"
)

# Google Drive File IDs (replace with your actual IDs)
FILE_IDS = {
    "model.joblib": "1jcZwgi6rliXjWxOh3h-60nsyWVRPnqx9",
    "le_region_classes.npy": "1cXJFwJuXUzLiJb1IhvX04g6QBXRcw_Sw",
    "le_stage_classes.npy": "1TqjkwqUET10Bzk-x5hCd2uuHjm3sD9py",
    "le_health_classes.npy": "1njABbBsqsOlPvYmEQKeoycb2PaOkRpY5",
    "le_plan_classes.npy": "1Wgi4_QpSyiNEQp-qVdCwoMp9xpUae4fF",
    "meal_ideas.json": "1w8eYGZnXXn13NeG7BA2KaLEaCCgqJre8"
}

# Cache for loaded assets
_assets = {
    "model": None,
    "encoders": None,
    "meal_ideas": None
}

def download_file(file_id, save_path):
    """Download file from Google Drive"""
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download file {file_id}: {str(e)}"
        )

def load_assets():
    """Load all required assets"""
    try:
        # Download files if missing
        for file_name, file_id in FILE_IDS.items():
            if not os.path.exists(file_name):
                download_file(file_id, file_name)

        # Load model
        if _assets["model"] is None:
            _assets["model"] = joblib.load("model.joblib")

        # Load encoders
        if _assets["encoders"] is None:
            _assets["encoders"] = {
                "region": LabelEncoder().fit(np.load("le_region_classes.npy", allow_pickle=True)),
                "stage": LabelEncoder().fit(np.load("le_stage_classes.npy", allow_pickle=True)),
                "health": LabelEncoder().fit(np.load("le_health_classes.npy", allow_pickle=True)),
                "plan": LabelEncoder().fit(np.load("le_plan_classes.npy", allow_pickle=True))
            }

        # Load meal ideas
        if _assets["meal_ideas"] is None:
            with open("meal_ideas.json", 'r') as f:
                _assets["meal_ideas"] = json.load(f)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Asset loading failed: {str(e)}"
        )

class UserInput(BaseModel):
    age: int
    region: str
    breastfeeding_stage: str
    health_condition: str

@app.on_event("startup")
async def startup_event():
    """Preload assets when starting"""
    try:
        load_assets()
    except Exception as e:
        print(f"Startup warning: {str(e)}")

@app.get("/")
def home():
    return {
        "message": "Nutrition Plan API is running",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "POST /predict"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "ready",
        "model_loaded": _assets["model"] is not None,
        "encoders_loaded": _assets["encoders"] is not None
    }

@app.post("/predict")
def predict(user_input: UserInput):
    try:
        load_assets()
        
        # Validate inputs
        encoders = _assets["encoders"]
        for field, encoder_name in [
            ("region", "region"),
            ("breastfeeding_stage", "stage"),
            ("health_condition", "health")
        ]:
            value = getattr(user_input, field)
            if value not in encoders[encoder_name].classes_:
                raise ValueError(
                    f"Invalid {field}: '{value}'. Valid options: {list(encoders[encoder_name].classes_)}"
                )

        # Prepare features
        features = [
            user_input.age,
            encoders["region"].transform([user_input.region])[0],
            encoders["stage"].transform([user_input.breastfeeding_stage])[0],
            encoders["health"].transform([user_input.health_condition])[0]
        ]

        # Predict
        plan = encoders["plan"].inverse_transform(
            _assets["model"].predict([features])
        )[0]

        return {
            "plan": plan,
            "meal_ideas": _assets["meal_ideas"].get(plan, ["General balanced meal"]),
            "tips": [
                "Ensure adequate hydration",
                "Consume local and seasonal foods",
                "Consult a healthcare provider for personalized advice"
            ]
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
