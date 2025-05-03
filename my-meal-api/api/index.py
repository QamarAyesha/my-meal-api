# api/index.py
from fastapi import FastAPI
import joblib
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
import os

app = FastAPI()

# 1. Minimal dependency imports
@app.on_event("startup")
def load_assets():
    """Load files DIRECTLY from deployment package"""
    app.state.model = joblib.load("model.joblib")
    app.state.encoders = {
        "health": LabelEncoder().fit(np.load("le_health_classes.npy")),
        "plan": LabelEncoder().fit(np.load("le_plan_classes.npy")),
        "region": LabelEncoder().fit(np.load("le_region_classes.npy")),
        "stage": LabelEncoder().fit(np.load("le_stage_classes.npy"))
    }
    app.state.meals = json.load(open("meal_ideas.json"))

# 2. Disable unused FastAPI features
app = FastAPI(
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)
