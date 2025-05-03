from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import joblib
import json
import os

app = FastAPI(
    title="Meal Plan Recommender API",
    description="API for personalized meal recommendations",
    version="1.0"
)

# Load assets once at startup
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(BASE_DIR, "..", "models", "meal_recommender_rf.pkl"))
    encoders = joblib.load(os.path.join(BASE_DIR, "..", "models", "label_encoders.pkl"))
    
    with open(os.path.join(BASE_DIR, "..", "models", "meal_ideas.json"), "r") as f:
        meal_ideas = json.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load model or encoders: {str(e)}")

# Request body schema
class UserInput(BaseModel):
    age: int
    region: str
    breastfeeding_stage: str
    health_condition: str

@app.get("/")
def home():
    return {
        "message": "Welcome to the Meal Plan API!",
        "try": "/docs for Swagger UI"
    }

@app.post("/predict")
def predict(user_input: UserInput):
    try:
        # Encode features
        region_enc = encoders['le_region'].transform([user_input.region])[0]
        stage_enc = encoders['le_stage'].transform([user_input.breastfeeding_stage])[0]
        health_enc = encoders['le_health'].transform([user_input.health_condition])[0]

        # Predict
        prediction = model.predict([[user_input.age, region_enc, stage_enc, health_enc]])[0]
        plan = encoders['le_plan'].inverse_transform([prediction])[0]

        return {
            "plan": plan,
            "meal_ideas": meal_ideas.get(plan, ["Balanced meal"]),
            "tips": [
                "Stay hydrated",
                "Include local, seasonal foods"
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
