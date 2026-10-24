"""
src/evaluation/retrieval_metrics.py

Information Retrieval Metrics for Amharic FinTech RAG.

Implements standard IR metrics used in academic research:
- Recall@K: Fraction of relevant documents retrieved in top K
- Precision@K: Fraction of retrieved documents that are relevant
- MRR (Mean Reciprocal Rank): 1/rank of first relevant document
- NDCG@K (Normalized Discounted Cumulative Gain): Ranking quality
- MAP (Mean Average Precision): Overall retrieval effectiveness
- Hit Rate: Whether any relevant document was retrieved

Expected Input:
    retrieved_ids: List[str] - Retrieved chunk IDs in rank order
    relevant_ids: List[str] - Ground truth relevant chunk IDs

Expected Output:
    float - Metric value between 0.0 and 1.0

Design Principles:
- Pure Python/NumPy implementation (no scipy/sklearn dependencies)
- Handles edge cases (empty lists, no relevant docs)
- Supports both single-query and batch evaluation
- Includes statistical measures (mean, std, confidence intervals)
- Publication-ready formatting
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# SINGLE-QUERY METRICS
# ============================================================

def calculate_recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int
) -> float:
    """
    Calculate Recall@K.

    Formula: |{relevant docs} ∩ {retrieved docs in top K}| / |{relevant docs}|

    Args:
        retrieved_ids: List of retrieved chunk IDs in rank order
        relevant_ids: List of ground truth relevant chunk IDs
        k: Number of top results to consider

    Returns:
        Recall@K value between 0.0 and 1.0

    Example:
        >>> retrieved = ["doc1", "doc2", "doc3", "doc4"]
        >>> relevant = ["doc1", "doc3", "doc5"]
        >>> calculate_recall_at_k(retrieved, relevant, k=3)
        0.6666666666666666  # Found 2 out of 3 relevant docs in top 3
    """
    if not relevant_ids:
        logger.warning("No relevant documents provided. Returning 0.0")
        return 0.0

    if not retrieved_ids or k <= 0:
        return 0.0

    retrieved_at_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)

    relevant_retrieved = len(retrieved_at_k & relevant_set)
    return relevant_retrieved / len(relevant_ids)


def calculate_precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int
) -> float:
    """
    Calculate Precision@K.

    Formula: |{relevant docs} ∩ {retrieved docs in top K}| / K

    Args:
        retrieved_ids: List of retrieved chunk IDs in rank order
        relevant_ids: List of ground truth relevant chunk IDs
        k: Number of top results to consider

    Returns:
        Precision@K value between 0.0 and 1.0
    """
    if not retrieved_ids or k <= 0:
        return 0.0

    relevant_set = set(relevant_ids)
    retrieved_at_k = retrieved_ids[:k]

    relevant_retrieved = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_set)
    return relevant_retrieved / min(k, len(retrieved_ids))


def calculate_mrr(
    retrieved_ids: List[str],
    relevant_ids: List[str]
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).

    Formula: 1 / rank_of_first_relevant_document

    Args:
        retrieved_ids: List of retrieved chunk IDs in rank order
        relevant_ids: List of ground truth relevant chunk IDs

    Returns:
        MRR value between 0.0 and 1.0
    """
    if not relevant_ids or not retrieved_ids:
        return 0.0

    relevant_set = set(relevant_ids)

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank

    return 0.0


def calculate_ndcg_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@K).

    Measures ranking quality by considering position of relevant documents.

    Formula: DCG@K / IDCG@K
    where DCG = Σ (rel_i / log2(i+1)) for i in 1..K
    and IDCG is the ideal DCG (all relevant docs at top positions)

    Args:
        retrieved_ids: List of retrieved chunk IDs in rank order
        relevant_ids: List of ground truth relevant chunk IDs
        k: Number of top results to consider

    Returns:
        NDCG@K value between 0.0 and 1.0
    """
    if not relevant_ids or not retrieved_ids or k <= 0:
        return 0.0

    relevant_set = set(relevant_ids)

    # Calculate DCG (Discounted Cumulative Gain)
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_set:
            dcg += 1.0 / math.log2(i + 1)

    # Calculate IDCG (Ideal DCG)
    # Best case: all relevant docs are at the top
    ideal_relevant_count = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_relevant_count + 1))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def calculate_average_precision(
    retrieved_ids: List[str],
    relevant_ids: List[str]
) -> float:
    """
    Calculate Average Precision (AP) for a single query.

    Formula: (1/|relevant|) * Σ (Precision@k * rel(k)) for k where rel(k)=1

    Args:
        retrieved_ids: List of retrieved chunk IDs in rank order
        relevant_ids: List of ground truth relevant chunk IDs

    Returns:
        AP value between 0.0 and 1.0
    """
    if not relevant_ids or not retrieved_ids:
        return 0.0

    relevant_set = set(relevant_ids)

    precision_sum = 0.0
    relevant_count = 0

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            relevant_count += 1
            precision_at_rank = relevant_count / rank
            precision_sum += precision_at_rank

    if relevant_count == 0:
        return 0.0

    return precision_sum / len(relevant_ids)


# Alias for MAP (Mean Average Precision) – for a single query, MAP = AP.
calculate_map = calculate_average_precision


def calculate_hit_rate(
    retrieved_ids: List[str],
    relevant_ids: List[str]
) -> float:
    """
    Calculate Hit Rate: whether at least one relevant chunk was retrieved.

    Returns 1.0 if at least one relevant chunk appears in the results,
    0.0 otherwise.

    Args:
        retrieved_ids: List of retrieved chunk IDs (ordered by rank)
        relevant_ids: List of ground-truth relevant chunk IDs

    Returns:
        Hit Rate (0.0 or 1.0)
    """
    if not relevant_ids or not retrieved_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    return 1.0 if any(cid in relevant_set for cid in retrieved_ids) else 0.0


# ============================================================
# BATCH EVALUATION
# ============================================================

def evaluate_retrieval_batch(
    queries_results: List[Dict[str, Any]],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    Evaluate retrieval metrics across a batch of queries.

    This is the main function for publication-ready retrieval evaluation.

    Args:
        queries_results: List of dicts with keys:
            - 'retrieved_ids': List[str] - Retrieved chunk IDs in rank order
            - 'relevant_ids': List[str] - Ground truth relevant chunk IDs
            - 'query_id': Optional[int] - Query identifier
        k_values: List of K values for Recall@K, Precision@K, and NDCG@K

    Returns:
        Dict with metric names as keys and statistical values as values.
        Includes mean, std, min, max, and 95% confidence intervals.
    """
    if not queries_results:
        logger.warning("Empty queries_results list. Returning zero metrics.")
        return _empty_metrics(k_values)

    # Accumulate per-query metric values
    accumulators: Dict[str, List[float]] = {}

    for i, result in enumerate(queries_results):
        retrieved = result.get("retrieved_ids", [])
        relevant = result.get("relevant_ids", [])
        query_id = result.get("query_id", i)

        if not retrieved:
            logger.warning(f"Query {query_id} has no retrieved IDs. Skipping.")
            continue

        # Calculate metrics for each k
        for k in k_values:
            accumulators.setdefault(f"Recall@{k}", []).append(
                calculate_recall_at_k(retrieved, relevant, k)
            )
            accumulators.setdefault(f"Precision@{k}", []).append(
                calculate_precision_at_k(retrieved, relevant, k)
            )
            accumulators.setdefault(f"NDCG@{k}", []).append(
                calculate_ndcg_at_k(retrieved, relevant, k)
            )

        # Query-level metrics (not k-dependent)
        accumulators.setdefault("MRR", []).append(
            calculate_mrr(retrieved, relevant)
        )
        accumulators.setdefault("MAP", []).append(
            calculate_map(retrieved, relevant)
        )
        accumulators.setdefault("HitRate", []).append(
            calculate_hit_rate(retrieved, relevant)
        )

    # Compute statistics for each metric
    final_metrics = {}
    for metric_name, values in accumulators.items():
        if not values:
            final_metrics[f"{metric_name}_mean"] = 0.0
            final_metrics[f"{metric_name}_std"] = 0.0
            continue

        values_array = np.array(values)
        mean = float(np.mean(values_array))
        std = float(np.std(values_array, ddof=1)) if len(values) > 1 else 0.0

        final_metrics[f"{metric_name}_mean"] = mean
        final_metrics[f"{metric_name}_std"] = std
        final_metrics[f"{metric_name}_min"] = float(np.min(values_array))
        final_metrics[f"{metric_name}_max"] = float(np.max(values_array))

        # 95% Confidence Interval
        if len(values) >= 2 and std > 0:
            ci_low, ci_high = _calculate_confidence_interval(values_array, confidence=0.95)
            final_metrics[f"{metric_name}_ci_low"] = ci_low
            final_metrics[f"{metric_name}_ci_high"] = ci_high
        else:
            final_metrics[f"{metric_name}_ci_low"] = mean
            final_metrics[f"{metric_name}_ci_high"] = mean

    final_metrics["num_queries"] = len(queries_results)
    final_metrics["num_processed"] = len(accumulators.get("MRR", []))

    logger.info(
        f"✅ Retrieval evaluation complete: "
        f"Recall@3={final_metrics.get('Recall@3_mean', 0.0):.4f}, "
        f"MRR={final_metrics.get('MRR_mean', 0.0):.4f}"
    )

    return final_metrics


def _calculate_confidence_interval(
    values: np.ndarray,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """Calculate confidence interval for a list of metric values."""
    n = len(values)
    if n < 2:
        mean = float(np.mean(values)) if n > 0 else 0.0
        return mean, mean

    mean = np.mean(values)
    std_err = np.std(values, ddof=1) / np.sqrt(n)

    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence, 1.96)

    margin = z * std_err
    return float(mean - margin), float(mean + margin)


def _empty_metrics(k_values: List[int]) -> Dict[str, float]:
    """Return empty metrics dictionary when no queries are provided."""
    metrics = {
        "num_queries": 0,
        "num_processed": 0,
    }
    for k in k_values:
        metrics[f"Recall@{k}_mean"] = 0.0
        metrics[f"Recall@{k}_std"] = 0.0
        metrics[f"Recall@{k}_min"] = 0.0
        metrics[f"Recall@{k}_max"] = 0.0
        metrics[f"Recall@{k}_ci_low"] = 0.0
        metrics[f"Recall@{k}_ci_high"] = 0.0

        metrics[f"Precision@{k}_mean"] = 0.0
        metrics[f"Precision@{k}_std"] = 0.0
        metrics[f"Precision@{k}_min"] = 0.0
        metrics[f"Precision@{k}_max"] = 0.0
        metrics[f"Precision@{k}_ci_low"] = 0.0
        metrics[f"Precision@{k}_ci_high"] = 0.0

        metrics[f"NDCG@{k}_mean"] = 0.0
        metrics[f"NDCG@{k}_std"] = 0.0
        metrics[f"NDCG@{k}_min"] = 0.0
        metrics[f"NDCG@{k}_max"] = 0.0
        metrics[f"NDCG@{k}_ci_low"] = 0.0
        metrics[f"NDCG@{k}_ci_high"] = 0.0

    metrics["MRR_mean"] = 0.0
    metrics["MRR_std"] = 0.0
    metrics["MRR_min"] = 0.0
    metrics["MRR_max"] = 0.0
    metrics["MRR_ci_low"] = 0.0
    metrics["MRR_ci_high"] = 0.0

    metrics["MAP_mean"] = 0.0
    metrics["MAP_std"] = 0.0
    metrics["MAP_min"] = 0.0
    metrics["MAP_max"] = 0.0
    metrics["MAP_ci_low"] = 0.0
    metrics["MAP_ci_high"] = 0.0

    metrics["HitRate_mean"] = 0.0
    metrics["HitRate_std"] = 0.0
    metrics["HitRate_min"] = 0.0
    metrics["HitRate_max"] = 0.0
    metrics["HitRate_ci_low"] = 0.0
    metrics["HitRate_ci_high"] = 0.0

    return metrics


# ============================================================
# PER-QUERY RESULTS (For Detailed Analysis)
# ============================================================

def evaluate_retrieval_per_query(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    Evaluate retrieval metrics for a single query.

    Returns all metrics for one query, useful for detailed analysis.

    Args:
        retrieved_ids: List of retrieved chunk IDs in rank order
        relevant_ids: List of ground truth relevant chunk IDs
        k_values: List of K values for Recall@K, Precision@K, and NDCG@K

    Returns:
        Dict with metric names as keys and values as floats
    """
    metrics = {}

    for k in k_values:
        metrics[f"Recall@{k}"] = calculate_recall_at_k(retrieved_ids, relevant_ids, k)
        metrics[f"Precision@{k}"] = calculate_precision_at_k(retrieved_ids, relevant_ids, k)
        metrics[f"NDCG@{k}"] = calculate_ndcg_at_k(retrieved_ids, relevant_ids, k)

    metrics["MRR"] = calculate_mrr(retrieved_ids, relevant_ids)
    metrics["MAP"] = calculate_map(retrieved_ids, relevant_ids)
    metrics["HitRate"] = calculate_hit_rate(retrieved_ids, relevant_ids)

    return metrics


# ============================================================
# PUBLICATION-FRIENDLY FORMATTER
# ============================================================

def format_retrieval_summary(metrics: Dict[str, float]) -> str:
    """
    Format retrieval metrics as a publication-ready table.

    Args:
        metrics: Output from evaluate_retrieval_batch()

    Returns:
        Formatted string table
    """
    if not metrics or metrics.get("num_queries", 0) == 0:
        return "No results available"

    lines = [
        "=" * 70,
        "RETRIEVAL EVALUATION SUMMARY",
        "=" * 70,
        f"Total Queries: {metrics.get('num_queries', 0)}",
        f"Processed: {metrics.get('num_processed', 0)}",
        "",
        "Metric           Mean     Std      Min      Max      95% CI",
        "-" * 70,
    ]

    metric_order = [
        ("Recall@1", "Recall@1"),
        ("Recall@3", "Recall@3"),
        ("Recall@5", "Recall@5"),
        ("Recall@10", "Recall@10"),
        ("Precision@1", "Precision@1"),
        ("Precision@3", "Precision@3"),
        ("Precision@5", "Precision@5"),
        ("Precision@10", "Precision@10"),
        ("NDCG@1", "NDCG@1"),
        ("NDCG@3", "NDCG@3"),
        ("NDCG@5", "NDCG@5"),
        ("NDCG@10", "NDCG@10"),
        ("MRR", "MRR"),
        ("MAP", "MAP"),
        ("HitRate", "HitRate"),
    ]

    for display_name, key in metric_order:
        mean = metrics.get(f"{key}_mean", 0.0)
        std = metrics.get(f"{key}_std", 0.0)
        min_val = metrics.get(f"{key}_min", 0.0)
        max_val = metrics.get(f"{key}_max", 0.0)
        ci_low = metrics.get(f"{key}_ci_low", 0.0)
        ci_high = metrics.get(f"{key}_ci_high", 0.0)

        lines.append(
            f"{display_name:<16} {mean:.4f}  {std:.4f}  {min_val:.4f}  {max_val:.4f}  [{ci_low:.4f}, {ci_high:.4f}]"
        )

    lines.append("=" * 70)

    return "\n".join(lines)


# ============================================================
# DEMO / TEST
# ============================================================

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    print("=" * 70)
    print("RETRIEVAL METRICS DEMO")
    print("=" * 70)

    # Example 1: Perfect retrieval
    print("\n📊 Example 1: Perfect Retrieval")
    retrieved = ["doc1", "doc2", "doc3"]
    relevant = ["doc1", "doc2", "doc3"]

    per_query = evaluate_retrieval_per_query(retrieved, relevant, k_values=[1, 3, 5])

    print("   Per-Query Metrics:")
    for key, value in per_query.items():
        print(f"      {key}: {value:.4f}")

    # Example 2: Batch evaluation
    print("\n📊 Example 2: Batch Evaluation")
    batch_results = [
        {"retrieved_ids": ["doc1", "doc2", "doc3"], "relevant_ids": ["doc1", "doc2"]},
        {"retrieved_ids": ["doc1", "doc4", "doc2"], "relevant_ids": ["doc2", "doc3"]},
        {"retrieved_ids": ["doc5", "doc6", "doc7"], "relevant_ids": ["doc5", "doc6", "doc7"]},
        {"retrieved_ids": ["doc1", "doc2", "doc3"], "relevant_ids": ["doc4", "doc5"]},
    ]

    metrics = evaluate_retrieval_batch(batch_results, k_values=[1, 3, 5])

    print("\n" + format_retrieval_summary(metrics))

    # Example 3: Per-query details
    print("\n📊 Example 3: Per-Query Details (First Query)")
    first_result = batch_results[0]
    per_query = evaluate_retrieval_per_query(
        first_result["retrieved_ids"],
        first_result["relevant_ids"],
        k_values=[1, 3, 5, 10]
    )
    for key, value in per_query.items():
        print(f"   {key}: {value:.4f}")

    print("\n" + "=" * 70)
    print("✅ RETRIEVAL METRICS DEMO COMPLETED")
    print("=" * 70)


# ---------------------
# WHAT IT SOLVES:
# ---------------------
# This module computes publication-ready retrieval metrics for the Amharic
# FinTech RAG system. It implements standard IR metrics:
#   - Recall@k: Coverage of relevant chunks
#   - Precision@k: Accuracy of retrieved chunks
#   - MRR: Rank of first relevant chunk
#   - NDCG@k: Ranking quality with graded relevance
#   - MAP: Average precision across relevant items
#   - Hit Rate: Whether any relevant chunk was retrieved
#
# It handles batch evaluation with statistical measures (mean, std, min, max,
# 95% confidence intervals) and produces publication-ready formatted output.
# ---------------------