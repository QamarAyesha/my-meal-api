# api/index.py
from fastapi import FastAPI, HTTPException
import os

app = FastAPI(docs_url=None, redoc_url=None)

# Lazy imports inside functions
def predict(age: int, region: str, stage: str, health: str):
    import joblib
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    
    # Load assets only when needed
    assets = {
        "model": joblib.load("model.joblib"),
        "encoders": {
            "health": LabelEncoder().fit(np.load("le_health_classes.npy")),
            "plan": LabelEncoder().fit(np.load("le_plan_classes.npy")),
            "region": LabelEncoder().fit(np.load("le_region_classes.npy")),
            "stage": LabelEncoder().fit(np.load("le_stage_classes.npy"))
        }
    }
    
    # Your prediction logic here
    return {"plan": "High iron"}

@app.post("/predict")
async def predict_endpoint(age: int, region: str, stage: str, health: str):
    return predict(age, region, stage, health)
