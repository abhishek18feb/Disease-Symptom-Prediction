from fastapi import FastAPI
import numpy as np
import pandas as pd

from app.schemas.request_response import PredictionRequest, PredictionResponse
from app.services.model_loader import symptom_columns, severity_map
from app.services.predictor import predict_primary_disease
from app.services.similarity import similarity_search

app = FastAPI(title="Disease Prediction API")

def build_input_vector(symptoms):
    x = pd.DataFrame(0, index=[0], columns=symptom_columns)

    for s in symptoms:
        s = s.strip()
        if s in severity_map:
            x.loc[0, s] = severity_map[s]

    return x.to_numpy(dtype="float32")

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    input_vector = build_input_vector(request.symptoms)

    primary = predict_primary_disease(input_vector)

    # Example: use same vector for similarity (can be replaced later)
    similar = similarity_search(input_vector, input_vector)

    return {
        "primary_prediction": primary,
        "similar_diseases": similar
    }

