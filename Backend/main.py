from fastapi import FastAPI
from pydantic import BaseModel
# FIX: Absolute import since they are in the same folder
from model import get_prediction 

app = FastAPI(title="Disease Prediction API")

# Data model for validation
class SymptomInput(BaseModel):
    symptoms: str

@app.get("/")
async def root():
    return {"message": "Disease Prediction API is running"}

@app.post("/predict")
async def predict(data: SymptomInput):
    # Calls the prediction function from model.py
    results = get_prediction(data.symptoms)
    return {
        "status": "success",
        "predictions": {
            "Random Forest Prediction": results["rf"],
            "Naive Bayes Prediction": results["nb"],
            "SVM Prediction": results["svm"],
            "Final Prediction": results["final"]
        }
    }