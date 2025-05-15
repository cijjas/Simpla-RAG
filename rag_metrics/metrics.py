# rag_metrics/metrics.py
from typing import List, Sequence
from .similarity import EmbeddingModel
from difflib import SequenceMatcher
import math

class RAGMetrics:
    """
    Métricas de evaluación para un pipeline RAG.
    """

    def __init__(self, model_name: str = "intfloat/e5-base-v2"):
        self.embedder = EmbeddingModel(model_name)

    # ─────────────────────────────────────────── retrieval / context

    def _flags_relevant(
        self, chunks: Sequence[str], reference: str, threshold: float
    ) -> List[int]:
        """Marca 1 si chunk es relevante al reference (similaridad ≥ threshold)."""
        return [
            1 if self.embedder.similarity(c, reference) >= threshold else 0
            for c in chunks
        ]

    # cuantos documentos relevantes hay en el top-k
    def context_precision(
        self, contexts: Sequence[str], reference_answer: str, threshold: float = 0.4
    ) -> float:
        """
        Context Precision@K — versión ponderada como en tu slide:
            Σ_k Precision@k * v_k  /  (# relevantes en top‑K)
        donde v_k = 1 si el chunk k‑ésimo es relevante.
        """
        if not contexts:
            return 0.0

        flags = self._flags_relevant(contexts, reference_answer, threshold)
        total_relevant_topk = sum(flags)
        if total_relevant_topk == 0:
            return 0.0

        cum_rel = 0
        weighted_precision = 0.0
        for k, is_rel in enumerate(flags, start=1):
            if is_rel:
                cum_rel += 1
                weighted_precision += (cum_rel / k)

        return weighted_precision / total_relevant_topk


    # evalua que tan bien estas retriveando
    def context_recall(
        self, contexts: Sequence[str], reference_answer: str
    ) -> float:
        """
        Context Recall — ¿cuánto del ground‑truth está cubierto por el contexto?
        Aproximación: similitud ref‑vs‑contextos concatenados.
        """
        if not contexts:
            return 0.0
        combined = " ".join(contexts)
        return self.embedder.similarity(reference_answer, combined)

    # ─────────────────────────────────────────── faithfulness / relevance

    # model answer vs context + model answer vs reference / 2
    def faithfulness(
        self,
        contexts: Sequence[str],
        model_answer: str,
        reference_answer: str | None = None,
    ) -> float:
        """
        Evalúa si la respuesta está sustentada por el contexto.
        - Siempre compara respuesta vs contexto.
        - Si hay ground‑truth, promedia con respuesta vs referencia
        """
        if not contexts:
            return 0.0
        combined = " ".join(contexts)
        sim_ctx = self.embedder.similarity(model_answer, combined)

        if reference_answer:
            sim_ref = self.embedder.similarity(model_answer, reference_answer)
            return (sim_ctx + sim_ref) / 2
        return sim_ctx

    #  question vs answer (medio raro queres que baje)
    def answer_relevance(self, question: str, answer: str) -> float:
        """¿Qué tan bien la respuesta atiende la pregunta?"""
        return self.embedder.similarity(question, answer)

    # reference vs model ligado a faithfullness
    def answer_correctness(self, reference_answer: str, model_answer: str) -> float:
        """Similitud semántica respuesta generada vs ground‑truth."""
        return self.embedder.similarity(reference_answer, model_answer)


    
    def retrieval_precision_at_k(
        self,
        contexts: Sequence[str],
        reference_answer: str,
        k: int = 5,
        threshold: float = 0.4,
    ) -> float:
        """Precision@k clásica."""
        if not contexts:
            return 0.0
        topk = contexts[:k]
        flags = self._flags_relevant(topk, reference_answer, threshold)
        return sum(flags) / len(topk)

    def retrieval_recall_at_k(
        self,
        contexts: Sequence[str],
        reference_answer: str,
        k: int = 5,
        threshold: float = 0.4,
    ) -> float:
        """Recall@k clásica."""
        if not contexts:
            return 0.0
        flags_all = self._flags_relevant(contexts, reference_answer, threshold)
        total_relevant = sum(flags_all)
        if total_relevant == 0:
            return 0.0
        flags_topk = flags_all[:k]
        return sum(flags_topk) / total_relevant

    def rouge_l(self, reference: str, hypothesis: str) -> float:
        """
        ROUGE‑L F-score basado en LCS (longest common subsequence).
        Sencillo y sin peso β (usa β=1).
        """
        matcher = SequenceMatcher(None, reference, hypothesis)
        lcs = sum(triple.size for triple in matcher.get_matching_blocks())
        if lcs == 0:
            return 0.0
        prec = lcs / len(hypothesis)
        rec = lcs / len(reference)
        return 2 * prec * rec / (prec + rec)

