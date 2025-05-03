from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import json
import requests
import io

app = FastAPI(
    title="Meal Plan Recommender API",
    description="API for personalized meal recommendations",
    version="1.0"
)

MODEL_URL = "https://drive.google.com/uc?export=download&id=1FUM93UEhrz1jgZeXJ0rs11SV0mpkH5--"
ENCODERS_URL = "https://drive.google.com/uc?export=download&id=1d_WKKgWZMn_HmEsiFHWsifrRDwJxbuaZ"
MEAL_IDEAS_URL = "https://drive.google.com/uc?export=download&id=1wdLlm82Lz9rzMCsiZLL9MtGn6WogqpJZ"

def fetch_file(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to load from {url}")
    return response.content

# Global variables for the loaded model and data
model = None
encoders = None
meal_ideas = None

def load_assets():
    global model, encoders, meal_ideas
    if model is None or encoders is None or meal_ideas is None:
        try:
            model = joblib.load(io.BytesIO(fetch_file(MODEL_URL)))
            encoders = joblib.load(io.BytesIO(fetch_file(ENCODERS_URL)))
            meal_ideas = json.loads(fetch_file(MEAL_IDEAS_URL).decode())
        except Exception as e:
            raise RuntimeError(f"Failed to load model or data from URL: {str(e)}")

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
    load_assets()  # Lazy load model and data only when needed
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
