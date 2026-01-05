from pydantic import BaseModel
from typing import List

class PredictionRequest(BaseModel):
    symptoms: List[str]

class SimilarDisease(BaseModel):
    disease: str
    similarity_score: float

class PredictionResponse(BaseModel):
    primary_prediction: dict
    similar_diseases: List[SimilarDisease]
