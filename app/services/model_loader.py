import tensorflow as tf
import pickle
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model/disease_prediction.keras")
ENCODER_PATH = os.path.join(BASE_DIR, "model/disease_label_encoder.pkl")
SEVERITY_PATH = os.path.join(BASE_DIR, "model/severity_map.pkl")
SYMPTOM_PATH = os.path.join(BASE_DIR, "model/symptom_columns.pkl")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load artifacts
with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

with open(SEVERITY_PATH, "rb") as f:
    severity_map = pickle.load(f)

with open(SYMPTOM_PATH, "rb") as f:
    symptom_columns = pickle.load(f)
