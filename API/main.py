import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model = joblib.load("../model/model.joblib")
vectoriser = joblib.load("../model/vectoriser.joblib")

class PredictionRequest(BaseModel):
    request_text: str

@app.get("/health")
def health():
    return {"status":"ok"}

@app.get("/categories")
def categories():
    return {
        "categories": model.classes_.tolist()
    }

@app.post("/predict")
def predict(data: PredictionRequest):
    text = data.request_text
    #TODO data cleaning?, handle invalid requests
    vectorised_text = vectoriser.transform([text])

    prediction = model.predict(vectorised_text)

    confidence = max(model.predict_proba(vectorised_text)[0])

    return {
        "request_text": text,
        "predicted_category": prediction,
        "confidence": round(float(confidence), 3)
    }