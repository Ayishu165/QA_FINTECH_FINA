"""
src/evaluation/generation_metrics.py

Generation Evaluation Metrics for Amharic FinTech RAG.

Purpose:
    Evaluate the quality of generated answers independently of retrieval.

Metrics implemented:
    1. Faithfulness: Percentage of answer supported by evidence (via citations)
    2. Answer Relevance: Semantic/lexical relevance to the question
    3. Context Relevance: How relevant the retrieved context was
    4. Answer Completeness: Percentage of required information covered
    5. Answer Length: Character count (for reference)
    6. Citation Count: Number of citations used

Design Principles:
- Pure Python/NumPy implementation (NO external dependencies)
- No 'evaluate', 'datasets', or 'pyarrow' imports
- Handles edge cases (empty answers, no citations)
- Supports both single-query and batch evaluation
- Publication-ready formatting
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# SINGLE-QUERY METRICS
# ============================================================

def extract_citations(answer: str) -> List[str]:
    """
    Extract citation markers from the answer.

    Supports formats: [1], [2], [Chunk 1], [Source 1]

    Args:
        answer: The generated answer

    Returns:
        List of citation markers as strings
    """
    if not answer:
        return []

    # Pattern for [1], [2], [3] citations
    pattern = r"\[(\d+)\]"
    matches = re.findall(pattern, answer)

    return list(set(matches))  # Remove duplicates


def calculate_faithfulness(
    answer: str,
    citations: List[str],
    retrieved_chunks: List[Dict[str, Any]],
) -> float:
    """
    Calculate Faithfulness: proportion of claims supported by evidence.

    Uses citations as a proxy for faithfulness. If a claim has a
    citation that maps to a retrieved chunk, it is considered faithful.

    Args:
        answer: The generated answer
        citations: List of citation markers found in the answer (e.g., ["1", "2"])
        retrieved_chunks: List of retrieved chunk dicts with 'chunk_id'

    Returns:
        Faithfulness score between 0.0 and 1.0
    """
    if not answer or not citations:
        return 0.0

    # Count sentences/claims in the answer
    sentences = re.split(r'[።!?.]', answer)
    claim_count = len([s for s in sentences if s.strip()])

    if claim_count == 0:
        return 0.0

    # Count valid citations
    chunk_ids = {c.get("chunk_id", "") for c in retrieved_chunks}
    valid_citations = [c for c in citations if c in chunk_ids]

    # Each valid citation supports one claim
    faithfulness = min(1.0, len(valid_citations) / max(claim_count, 1))

    return faithfulness


def calculate_answer_relevance(
    answer: str,
    query: str,
) -> float:
    """
    Calculate Answer Relevance: how well the answer addresses the query.

    Uses lexical overlap (Jaccard similarity) as a proxy for relevance.

    Args:
        answer: The generated answer
        query: The user question

    Returns:
        Answer Relevance score between 0.0 and 1.0
    """
    if not answer or not query:
        return 0.0

    # Tokenize (simple word splitting)
    query_tokens = set(query.lower().split())
    answer_tokens = set(answer.lower().split())

    if not query_tokens:
        return 0.0

    # Jaccard similarity
    intersection = query_tokens & answer_tokens
    union = query_tokens | answer_tokens

    return len(intersection) / len(union) if union else 0.0


def calculate_context_relevance(
    answer: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> float:
    """
    Calculate Context Relevance: how much of the retrieved context was used.

    Measures the overlap between the answer and the retrieved chunks.

    Args:
        answer: The generated answer
        retrieved_chunks: List of retrieved chunk dicts with 'text' or 'document'

    Returns:
        Context Relevance score between 0.0 and 1.0
    """
    if not answer or not retrieved_chunks:
        return 0.0

    # Extract text from chunks
    chunk_texts = []
    for c in retrieved_chunks:
        text = c.get("text", c.get("document", ""))
        if text:
            chunk_texts.append(text)

    if not chunk_texts:
        return 0.0

    # Combine all chunk text
    combined_context = " ".join(chunk_texts)

    # Tokenize
    answer_tokens = set(answer.lower().split())
    context_tokens = set(combined_context.lower().split())

    if not answer_tokens or not context_tokens:
        return 0.0

    # Jaccard similarity between answer and context
    intersection = answer_tokens & context_tokens
    union = answer_tokens | context_tokens

    return len(intersection) / len(union) if union else 0.0


def calculate_answer_completeness(
    answer: str,
    ground_truth_keywords: List[str],
) -> float:
    """
    Calculate Answer Completeness: proportion of required information present.

    Args:
        answer: The generated answer
        ground_truth_keywords: List of keywords/requirements expected in the answer

    Returns:
        Completeness score between 0.0 and 1.0

    Example:
        >>> answer = "ID card and passport photo required"
        >>> requirements = ["ID card", "passport photo", "kebele confirmation"]
        >>> calculate_answer_completeness(answer, requirements)
        0.666  # 2 out of 3 requirements met
    """
    if not answer or not ground_truth_keywords:
        return 0.0 if ground_truth_keywords else 1.0

    answer_lower = answer.lower()

    # Count how many keywords appear in the answer
    matched = sum(1 for kw in ground_truth_keywords if kw.lower() in answer_lower)

    return matched / len(ground_truth_keywords) if ground_truth_keywords else 1.0


def calculate_answer_similarity(
    answer: str,
    reference_answer: str,
) -> float:
    """
    Calculate Answer Similarity to a reference answer.

    Uses ROUGE-L style longest common subsequence similarity.

    Args:
        answer: The generated answer
        reference_answer: The reference answer

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not answer or not reference_answer:
        return 0.0

    # Tokenize
    a_tokens = answer.lower().split()
    r_tokens = reference_answer.lower().split()

    if not a_tokens or not r_tokens:
        return 0.0

    # Longest Common Subsequence length
    m, n = len(a_tokens), len(r_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a_tokens[i - 1] == r_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]

    # ROUGE-L-like F1
    precision = lcs_len / m if m > 0 else 0.0
    recall = lcs_len / n if n > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


# ============================================================
# BATCH EVALUATION
# ============================================================

def evaluate_generation_batch(
    queries_results: List[Dict[str, Any]],
    ground_truth_keywords: Optional[List[List[str]]] = None,
) -> Dict[str, float]:
    """
    Evaluate generation metrics across a batch of queries.

    Args:
        queries_results: List of dicts with keys:
            - 'answer': str - Generated answer
            - 'query': str - User question
            - 'retrieved_chunks': List[Dict] - Retrieved chunks
            - 'citations': Optional[List[str]] - Extracted citations
        ground_truth_keywords: Optional list of keyword lists for completeness

    Returns:
        Dict with metric names as keys and statistical values as values
    """
    if not queries_results:
        logger.warning("Empty queries_results list. Returning zero metrics.")
        return _empty_metrics()

    # Accumulate per-query metric values
    accumulators: Dict[str, List[float]] = {
        "Faithfulness": [],
        "AnswerRelevance": [],
        "ContextRelevance": [],
        "AnswerLength": [],
        "CitationCount": [],
    }

    if ground_truth_keywords:
        accumulators["Completeness"] = []

    for i, result in enumerate(queries_results):
        answer = result.get("answer", "")
        query = result.get("query", "")
        chunks = result.get("retrieved_chunks", [])
        citations = result.get("citations", extract_citations(answer))

        # Faithfulness
        accumulators["Faithfulness"].append(
            calculate_faithfulness(answer, citations, chunks)
        )

        # Answer Relevance
        accumulators["AnswerRelevance"].append(
            calculate_answer_relevance(answer, query)
        )

        # Context Relevance
        accumulators["ContextRelevance"].append(
            calculate_context_relevance(answer, chunks)
        )

        # Answer Length
        accumulators["AnswerLength"].append(len(answer))

        # Citation Count
        accumulators["CitationCount"].append(len(citations))

        # Completeness (if ground truth provided)
        if ground_truth_keywords and i < len(ground_truth_keywords):
            keywords = ground_truth_keywords[i]
            accumulators["Completeness"].append(
                calculate_answer_completeness(answer, keywords)
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

    final_metrics["num_queries"] = len(queries_results)

    logger.info(
        f"✅ Generation evaluation complete: "
        f"Faithfulness={final_metrics.get('Faithfulness_mean', 0.0):.4f}"
    )

    return final_metrics


def _empty_metrics() -> Dict[str, float]:
    """Return empty metrics dictionary when no queries are provided."""
    return {
        "num_queries": 0,
        "Faithfulness_mean": 0.0,
        "Faithfulness_std": 0.0,
        "Faithfulness_min": 0.0,
        "Faithfulness_max": 0.0,
        "AnswerRelevance_mean": 0.0,
        "AnswerRelevance_std": 0.0,
        "AnswerRelevance_min": 0.0,
        "AnswerRelevance_max": 0.0,
        "ContextRelevance_mean": 0.0,
        "ContextRelevance_std": 0.0,
        "ContextRelevance_min": 0.0,
        "ContextRelevance_max": 0.0,
        "Completeness_mean": 0.0,
        "Completeness_std": 0.0,
        "Completeness_min": 0.0,
        "Completeness_max": 0.0,
        "AnswerLength_mean": 0.0,
        "AnswerLength_std": 0.0,
        "AnswerLength_min": 0.0,
        "AnswerLength_max": 0.0,
        "CitationCount_mean": 0.0,
        "CitationCount_std": 0.0,
        "CitationCount_min": 0.0,
        "CitationCount_max": 0.0,
    }


# ============================================================
# PUBLICATION-FRIENDLY FORMATTER
# ============================================================

def format_generation_summary(metrics: Dict[str, float]) -> str:
    """
    Format generation metrics as a publication-ready table.

    Args:
        metrics: Output from evaluate_generation_batch()

    Returns:
        Formatted string table
    """
    if not metrics or metrics.get("num_queries", 0) == 0:
        return "No results available"

    lines = [
        "=" * 70,
        "GENERATION EVALUATION SUMMARY",
        "=" * 70,
        f"Total Queries: {metrics.get('num_queries', 0)}",
        "",
        "Metric                Mean     Std      Min      Max",
        "-" * 70,
    ]

    metric_order = [
        ("Faithfulness", "Faithfulness"),
        ("AnswerRelevance", "Answer Relevance"),
        ("ContextRelevance", "Context Relevance"),
        ("Completeness", "Completeness"),
        ("AnswerLength", "Answer Length (chars)"),
        ("CitationCount", "Citation Count"),
    ]

    for key, display_name in metric_order:
        mean = metrics.get(f"{key}_mean", 0.0)
        std = metrics.get(f"{key}_std", 0.0)
        min_val = metrics.get(f"{key}_min", 0.0)
        max_val = metrics.get(f"{key}_max", 0.0)

        lines.append(
            f"{display_name:<20} {mean:.4f}  {std:.4f}  {min_val:.4f}  {max_val:.4f}"
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
    print("GENERATION METRICS DEMO")
    print("=" * 70)

    # Example batch
    batch_results = [
        {
            "answer": "የCBE የቁጠባ ሂሳብ ለመክፈት መታወቂያ ካርድ ያስፈልጋል። [1] የቀበሌ ማረጋገጫም ያስፈልጋል። [2]",
            "query": "ሂሳብ እንዴት መክፈት እችላለሁ?",
            "retrieved_chunks": [
                {"chunk_id": "chunk_1", "text": "መታወቂያ ካርድ ያስፈልጋል"},
                {"chunk_id": "chunk_2", "text": "የቀበሌ ማረጋገጫ ያስፈልጋል"},
            ],
            "citations": ["1", "2"],
        },
        {
            "answer": "የCBE ካርድ ለማግኘት ወደ ቅርንጫፍ መሄድ ያስፈልጋል። [1]",
            "query": "CBE ካርድ እንዴት ማግኘት እችላለሁ?",
            "retrieved_chunks": [
                {"chunk_id": "chunk_1", "text": "ወደ ቅርንጫፍ መሄድ ያስፈልጋል"},
            ],
            "citations": ["1"],
        },
    ]

    # Ground truth keywords for completeness
    keywords = [
        ["መታወቂያ ካርድ", "የቀበሌ ማረጋገጫ"],
        ["ወደ ቅርንጫፍ መሄድ"],
    ]

    metrics = evaluate_generation_batch(batch_results, ground_truth_keywords=keywords)

    print("\n" + format_generation_summary(metrics))

    print("\n" + "=" * 70)
    print("✅ GENERATION METRICS DEMO COMPLETED")
    print("=" * 70)