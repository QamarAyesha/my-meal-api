from fastapi import FastAPI
import os

app = FastAPI(docs_url=None)

# Lazy load only when needed
def predict():
    import joblib
    import numpy as np
    model = joblib.load("model.joblib")
    # ... (your prediction logic)
    return {"plan": "High iron"}

@app.post("/predict")
async def predict_endpoint():
    return predict()
