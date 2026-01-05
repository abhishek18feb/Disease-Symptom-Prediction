import numpy as np
from app.services.model_loader import model, label_encoder

def predict_primary_disease(input_vector):
    probs = model.predict(input_vector)[0]

    pred_idx = np.argmax(probs)
    disease = label_encoder.inverse_transform([pred_idx])[0]

    return {
        "disease": disease,
        "confidence": float(probs[pred_idx])
    }
