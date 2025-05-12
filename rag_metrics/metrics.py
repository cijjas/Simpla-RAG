# rag_metrics/metrics.py
from .similarity import EmbeddingModel

class RAGMetrics:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embedder = EmbeddingModel(model_name)

    def context_precision(self, contexts, ground_truth, threshold=0.7):
        if not contexts:
            return 0.0
        relevant = sum(
            1 for ctx in contexts if self.embedder.similarity(ctx, ground_truth) >= threshold
        )
        return relevant / len(contexts)

    def context_recall(self, contexts, ground_truth):
        combined_context = " ".join(contexts)
        return self.embedder.similarity(combined_context, ground_truth)

    def faithfulness(self, contexts, answer):
        combined_context = " ".join(contexts)
        return self.embedder.similarity(answer, combined_context)

    def answer_relevance(self, question, answer):
        return self.embedder.similarity(question, answer)
