"""
Evaluation Module for Amharic FinTech RAG.
"""
from src.evaluation.retrieval_metrics import (
    calculate_recall_at_k,
    calculate_precision_at_k,
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_average_precision,
    calculate_map,
    calculate_hit_rate,
    evaluate_retrieval_batch,
    evaluate_retrieval_per_query,
    format_retrieval_summary,
)
from src.evaluation.generation_metrics import (
    extract_citations,
    calculate_faithfulness,
    calculate_answer_relevance,
    calculate_context_relevance,
    calculate_answer_completeness,
    calculate_answer_similarity,
    evaluate_generation_batch,
    format_generation_summary,
)
from src.evaluation.trustworthiness_metrics import (
    calculate_grounding_accuracy,
    calculate_hallucination_rate,
    calculate_citation_accuracy,
    calculate_evidence_coverage,
    calculate_confidence_calibration,
    calculate_overall_trustworthiness,
    evaluate_trustworthiness_batch,
    format_trustworthiness_summary,
)

__all__ = [
    # Retrieval
    "calculate_recall_at_k",
    "calculate_precision_at_k",
    "calculate_mrr",
    "calculate_ndcg_at_k",
    "calculate_average_precision",
    "calculate_map",
    "calculate_hit_rate",
    "evaluate_retrieval_batch",
    "evaluate_retrieval_per_query",
    "format_retrieval_summary",
    # Generation
    "extract_citations",
    "calculate_faithfulness",
    "calculate_answer_relevance",
    "calculate_context_relevance",
    "calculate_answer_completeness",
    "calculate_answer_similarity",
    "evaluate_generation_batch",
    "format_generation_summary",
    # Trustworthiness
    "calculate_grounding_accuracy",
    "calculate_hallucination_rate",
    "calculate_citation_accuracy",
    "calculate_evidence_coverage",
    "calculate_confidence_calibration",
    "calculate_overall_trustworthiness",
    "evaluate_trustworthiness_batch",
    "format_trustworthiness_summary",
]