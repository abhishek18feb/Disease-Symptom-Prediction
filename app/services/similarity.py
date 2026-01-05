import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.services.model_loader import label_encoder

def similarity_search(input_vector, disease_embeddings, top_k=5):
    similarities = cosine_similarity(input_vector, disease_embeddings)[0]

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "disease": label_encoder.inverse_transform([idx])[0],
            "similarity_score": float(similarities[idx])
        })

    return results
