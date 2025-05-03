from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import json
import requests
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
from typing import Dict, Any

app = FastAPI(
    title="Nutrition Plan API",
    description="API for breastfeeding nutrition recommendations",
    version="1.0",
    docs_url="/docs",
    redoc_url=None
)

# Google Drive File IDs (using your actual IDs)
FILE_IDS = {
    "model.joblib": "1jcZwgi6rliXjWxOh3h-60nsyWVRPnqx9",
    "le_region_classes.npy": "1cXJFwJuXUzLiJb1IhvX04g6QBXRcw_Sw",
    "le_stage_classes.npy": "1TqjkwqUET10Bzk-x5hCd2uuHjm3sD9py",
    "le_health_classes.npy": "1njABbBsqsOlPvYmEQKeoycb2PaOkRpY5",
    "le_plan_classes.npy": "1Wgi4_QpSyiNEQp-qVdCwoMp9xpUae4fF",
    "meal_ideas.json": "1w8eYGZnXXn13NeG7BA2KaLEaCCgqJre8"
}

def download_file(file_id: str, save_path: str, max_retries: int = 3) -> bool:
    """Enhanced downloader with retry logic and streaming"""
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    for attempt in range(max_retries):
        try:
            with requests.get(url, stream=True, timeout=30) as response:
                response.raise_for_status()
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"Successfully downloaded {save_path}")
            return True
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {save_path}: {str(e)}")
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to download {save_path} after {max_retries} attempts"
                )
    return False

def load_assets() -> Dict[str, Any]:
    """Load all required assets with proper cleanup"""
    assets = {}
    try:
        # Download missing files
        for file_name, file_id in FILE_IDS.items():
            if not os.path.exists(file_name):
                download_file(file_id, file_name)

        # Load assets
        assets["model"] = joblib.load("model.joblib")
        
        assets["encoders"] = {
            "region": LabelEncoder().fit(np.load("le_region_classes.npy", allow_pickle=True)),
            "stage": LabelEncoder().fit(np.load("le_stage_classes.npy", allow_pickle=True)),
            "health": LabelEncoder().fit(np.load("le_health_classes.npy", allow_pickle=True)),
            "plan": LabelEncoder().fit(np.load("le_plan_classes.npy", allow_pickle=True))
        }
        
        with open("meal_ideas.json", 'r') as f:
            assets["meal_ideas"] = json.load(f)
            
        return assets
        
    except Exception as e:
        # Cleanup on failure
        for file_name in FILE_IDS:
            if os.path.exists(file_name):
                os.remove(file_name)
        raise HTTPException(
            status_code=500,
            detail=f"Asset loading failed: {str(e)}"
        )

class UserInput(BaseModel):
    age: int = Field(..., gt=15, lt=50, example=30)
    region: str = Field(..., example="South Asia")
    breastfeeding_stage: str = Field(..., example="Lactation")
    health_condition: str = Field(..., example="Anemia")

@app.on_event("startup")
async def startup_event():
    """Initialize with proper error handling"""
    try:
        app.state.assets = load_assets()
        app.state.healthy = True
    except Exception as e:
        app.state.healthy = False
        print(f"Startup failed: {str(e)}")

@app.get("/")
async def home():
    return {
        "status": "running",
        "healthy": app.state.healthy,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": {"method": "POST", "path": "/predict"}
        }
    }

@app.get("/health")
async def health_check():
    return {
        "ready": app.state.healthy,
        "assets_loaded": hasattr(app.state, "assets"),
        "files_present": all(os.path.exists(f) for f in FILE_IDS)
    }

@app.post("/predict")
async def predict(user_input: UserInput):
    if not getattr(app.state, "healthy", False):
        raise HTTPException(503, "Service unavailable")
    
    try:
        assets = app.state.assets
        encoders = assets["encoders"]
        
        # Validate inputs
        for field, encoder in [
            ("region", "region"),
            ("breastfeeding_stage", "stage"),
            ("health_condition", "health")
        ]:
            value = getattr(user_input, field)
            if value not in encoders[encoder].classes_:
                raise ValueError(
                    f"Invalid {field}: '{value}'. Valid options: {list(encoders[encoder].classes_)}"
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
            assets["model"].predict([features])
        )[0]

        return {
            "plan": plan,
            "meal_ideas": assets["meal_ideas"].get(plan, ["General balanced meal"]),
            "tips": [
                "Ensure adequate hydration",
                "Consume local and seasonal foods",
                "Consult a healthcare provider"
            ]
        }

    except ValueError as e:
        raise HTTPException(422, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Prediction error: {str(e)}")
