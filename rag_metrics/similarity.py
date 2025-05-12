# rag_metrics/similarity.py
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class EmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, text):
        return self.model.encode(text, convert_to_tensor=True)

    def similarity(self, text1, text2):
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        sim = cosine_similarity([emb1], [emb2])[0][0]
        return float(sim)
