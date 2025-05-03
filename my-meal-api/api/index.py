from fastapi import FastAPI
import os

app = FastAPI(docs_url=None, redoc_url=None)

# Lazy loading pattern
def get_model():
    import joblib
    return joblib.load("model.joblib")

def get_encoder(name):
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.classes_ = np.load(f"le_{name}_classes.npy", allow_pickle=True)
    return le

@app.post("/predict")
async def predict(age: int, region: str, stage: str, health: str):
    model = get_model()
    region_enc = get_encoder("region").transform([region])[0]
    stage_enc = get_encoder("stage").transform([stage])[0]
    health_enc = get_encoder("health").transform([health])[0]
    
    plan = get_encoder("plan").inverse_transform(
        model.predict([[age, region_enc, stage_enc, health_enc]])
    )[0]
    
    return {"plan": plan}
