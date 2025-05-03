# api/index.py
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
import os
import sys
from typing import Dict, Any

# Dependency verification (prints versions on startup)
print("\n=== Dependency Versions ===")
print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")
print(f"scikit-learn: {joblib.__version__}")
print(f"Joblib: {joblib.__version__}")
print("=========================\n")

# Initialize FastAPI with minimal settings
app = FastAPI(
    title="Nutrition Plan API",
    version="1.0",
    docs_url=None,      # Disable Swagger UI
    redoc_url=None,     # Disable ReDoc
    openapi_url=None    # Disable OpenAPI schema
)

def load_assets() -> Dict[str, Any]:
    """Load model and data files with robust error handling"""
    try:
        # Verify files exist first
        required_files = [
            "model.joblib",
            "le_health_classes.npy",
            "le_plan_classes.npy",
            "le_region_classes.npy",
            "le_stage_classes.npy",
            "meal_ideas.json"
        ]
        
        missing = [f for f in required_files if not os.path.exists(f)]
        if missing:
            raise FileNotFoundError(f"Missing files: {missing}")

        # Load with memory checks
        return {
            "model": joblib.load("model.joblib"),
            "encoders": {
                "health": LabelEncoder().fit(np.load("le_health_classes.npy", allow_pickle=True)),
                "plan": LabelEncoder().fit(np.load("le_plan_classes.npy", allow_pickle=True)),
                "region": LabelEncoder().fit(np.load("le_region_classes.npy", allow_pickle=True)),
                "stage": LabelEncoder().fit(np.load("le_stage_classes.npy", allow_pickle=True))
            },
            "meals": json.load(open("meal_ideas.json"))
        }
        
    except Exception as e:
        print(f"\n!!! CRITICAL LOAD ERROR !!!")
        print(f"Error type: {type(e).__name__}")
        print(f"Details: {str(e)}")
        print("\nCurrent directory contents:")
        print(os.listdir())
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize with verification"""
    try:
        app.state.assets = load_assets()
        print("\n✅ All assets loaded successfully")
        print(f"Model type: {type(app.state.assets['model'])}")
        print(f"Encoder classes loaded: {list(app.state.assets['encoders']['health'].classes_)}")
    except Exception as e:
        print("\n❌ Startup failed - API will not function")
        raise RuntimeError(f"Startup failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Enhanced health endpoint"""
    if not hasattr(app.state, 'assets'):
        raise HTTPException(503, "Service starting...")
    
    return {
        "status": "healthy",
        "files_loaded": [
            "model.joblib",
            "le_health_classes.npy",
            "le_plan_classes.npy", 
            "le_region_classes.npy",
            "le_stage_classes.npy",
            "meal_ideas.json"
        ],
        "memory_usage": f"{sys.getsizeof(app.state.assets) / 1024:.1f} KB"
    }

@app.post("/predict")
async def predict(age: int, region: str, stage: str, health: str):
    """
    Optimized prediction endpoint with:
    - Input validation
    - Memory safety
    - Detailed error reporting
    """
    if not hasattr(app.state, 'assets'):
        raise HTTPException(503, "Service not ready")
    
    try:
        # Validate inputs against encoder classes
        encoders = app.state.assets["encoders"]
        for name, value in [("region", region), 
                           ("stage", stage), 
                           ("health", health)]:
            if value not in encoders[name].classes_:
                raise ValueError(
                    f"Invalid {name}: '{value}'. Valid options: {list(encoders[name].classes_)}"
                )

        # Prepare features
        features = [
            age,
            encoders["region"].transform([region])[0],
            encoders["stage"].transform([stage])[0],
            encoders["health"].transform([health])[0]
        ]

        # Predict
        plan = encoders["plan"].inverse_transform(
            app.state.assets["model"].predict([features])
        )[0]

        return {
            "plan": plan,
            "meals": app.state.assets["meals"].get(plan, ["Balanced meal"]),
            "tips": [
                "Stay hydrated",
                "Consume seasonal foods",
                "Consult a nutritionist"
            ]
        }

    except ValueError as e:
        raise HTTPException(422, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Prediction failed: {str(e)}")
