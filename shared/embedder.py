import numpy as np
from sentence_transformers import SentenceTransformer

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    return _model


def embed_texts(texts, batch_size=64):
    model = get_model()
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True)


def embed_query(text):
    return embed_texts([text])[0]


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
