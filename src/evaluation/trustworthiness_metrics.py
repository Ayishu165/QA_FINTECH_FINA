"""
src/evaluation/trustworthiness_metrics.py

Trustworthiness Evaluation Metrics for Amharic FinTech RAG.

Purpose:
    Evaluate the trustworthiness of generated answers.

Metrics implemented:
    1. Grounding Accuracy: Percentage of claims supported by evidence
    2. Hallucination Rate: Fraction of answers with unsupported claims
    3. Citation Accuracy: Percentage of citations that are valid
    4. Evidence Coverage: Percentage of evidence chunks cited
    5. Confidence Calibration: Difference between confidence and grounding
    6. Overall Trustworthiness: Composite score (0-1)

Design Principles:
- Pure Python/NumPy implementation (NO external dependencies)
- Handles edge cases (empty answers, no citations)
- Supports batch evaluation
- Publication-ready formatting
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def calculate_grounding_accuracy(
    quality_report: Any,
) -> float:
    """
    Calculate Grounding Accuracy from a QualityReport.

    Args:
        quality_report: QualityReport object from quality_checker

    Returns:
        Grounding accuracy (0.0 to 1.0)
    """
    if quality_report is None:
        return 0.0
    return getattr(quality_report, "grounding_accuracy", 0.0)


def calculate_hallucination_rate(
    quality_report: Any,
) -> float:
    """
    Calculate Hallucination Rate (1 if hallucination detected, else 0).

    Args:
        quality_report: QualityReport object from quality_checker

    Returns:
        Hallucination rate (0.0 or 1.0)
    """
    if quality_report is None:
        return 0.0
    return 1.0 if getattr(quality_report, "hallucination_detected", False) else 0.0


def calculate_citation_accuracy(
    citation_report: Any,
) -> float:
    """
    Calculate Citation Accuracy from a CitationReport.

    Args:
        citation_report: CitationReport object from citation_generator

    Returns:
        Citation accuracy (0.0 to 1.0)
    """
    if citation_report is None:
        return 0.0
    return getattr(citation_report, "citation_accuracy", 0.0)


def calculate_evidence_coverage(
    citation_report: Any,
) -> float:
    """
    Calculate Evidence Coverage from a CitationReport.

    Args:
        citation_report: CitationReport object from citation_generator

    Returns:
        Evidence coverage (0.0 to 1.0)
    """
    if citation_report is None:
        return 0.0
    return getattr(citation_report, "evidence_coverage", 0.0)


def calculate_confidence_calibration(
    quality_report: Any,
) -> float:
    """
    Calculate Confidence Calibration Error.

    Measures the absolute difference between confidence and grounding.

    Args:
        quality_report: QualityReport object from quality_checker

    Returns:
        Calibration error (0.0 to 1.0)
    """
    if quality_report is None:
        return 1.0
    confidence = getattr(quality_report, "confidence_score", 0.0)
    grounding = getattr(quality_report, "grounding_accuracy", 0.0)
    return abs(confidence - grounding)


def calculate_overall_trustworthiness(
    grounding: float,
    citation_accuracy: float,
    evidence_coverage: float,
    hallucination_rate: float,
) -> float:
    """
    Calculate Overall Trustworthiness score (weighted sum).

    Weights:
        - Grounding: 0.35
        - Citation Accuracy: 0.25
        - Evidence Coverage: 0.20
        - Hallucination Rate (inverted): 0.20

    Args:
        grounding: Grounding accuracy (0.0 to 1.0)
        citation_accuracy: Citation accuracy (0.0 to 1.0)
        evidence_coverage: Evidence coverage (0.0 to 1.0)
        hallucination_rate: Hallucination rate (0.0 to 1.0)

    Returns:
        Overall trustworthiness (0.0 to 1.0)
    """
    weights = {
        "grounding": 0.35,
        "citation": 0.25,
        "coverage": 0.20,
        "hallucination": 0.20,
    }
    hallucination_score = 1.0 - hallucination_rate

    score = (
        grounding * weights["grounding"] +
        citation_accuracy * weights["citation"] +
        evidence_coverage * weights["coverage"] +
        hallucination_score * weights["hallucination"]
    )
    return round(min(1.0, max(0.0, score)), 4)


def evaluate_trustworthiness_batch(
    queries_results: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Evaluate trustworthiness metrics across a batch of queries.

    Args:
        queries_results: List of dicts with keys:
            - 'quality_report': QualityReport object
            - 'citation_report': CitationReport object
            - 'chunks': List of chunks (for coverage)

    Returns:
        Dict with metric names as keys and statistical values as values.
    """
    if not queries_results:
        return _empty_trust_metrics()

    accumulators: Dict[str, List[float]] = {
        "GroundingAccuracy": [],
        "HallucinationRate": [],
        "CitationAccuracy": [],
        "EvidenceCoverage": [],
        "ConfidenceCalibration": [],
        "OverallTrustworthiness": [],
    }

    for result in queries_results:
        quality = result.get("quality_report")
        citation = result.get("citation_report")

        grounding = calculate_grounding_accuracy(quality)
        hallucination = calculate_hallucination_rate(quality)
        citation_acc = calculate_citation_accuracy(citation)
        coverage = calculate_evidence_coverage(citation)
        calibration = calculate_confidence_calibration(quality)
        overall = calculate_overall_trustworthiness(
            grounding, citation_acc, coverage, hallucination
        )

        accumulators["GroundingAccuracy"].append(grounding)
        accumulators["HallucinationRate"].append(hallucination)
        accumulators["CitationAccuracy"].append(citation_acc)
        accumulators["EvidenceCoverage"].append(coverage)
        accumulators["ConfidenceCalibration"].append(calibration)
        accumulators["OverallTrustworthiness"].append(overall)

    final_metrics = {}
    for metric_name, values in accumulators.items():
        if not values:
            continue
        arr = np.array(values)
        final_metrics[f"{metric_name}_mean"] = float(np.mean(arr))
        final_metrics[f"{metric_name}_std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        final_metrics[f"{metric_name}_min"] = float(np.min(arr))
        final_metrics[f"{metric_name}_max"] = float(np.max(arr))

    final_metrics["num_queries"] = len(queries_results)

    logger.info(
        f"✅ Trustworthiness evaluation complete: "
        f"Overall Trust={final_metrics.get('OverallTrustworthiness_mean', 0.0):.4f}"
    )

    return final_metrics


def _empty_trust_metrics() -> Dict[str, float]:
    """Return empty metrics dictionary when no queries are provided."""
    return {
        "num_queries": 0,
        "GroundingAccuracy_mean": 0.0,
        "GroundingAccuracy_std": 0.0,
        "GroundingAccuracy_min": 0.0,
        "GroundingAccuracy_max": 0.0,
        "HallucinationRate_mean": 0.0,
        "HallucinationRate_std": 0.0,
        "HallucinationRate_min": 0.0,
        "HallucinationRate_max": 0.0,
        "CitationAccuracy_mean": 0.0,
        "CitationAccuracy_std": 0.0,
        "CitationAccuracy_min": 0.0,
        "CitationAccuracy_max": 0.0,
        "EvidenceCoverage_mean": 0.0,
        "EvidenceCoverage_std": 0.0,
        "EvidenceCoverage_min": 0.0,
        "EvidenceCoverage_max": 0.0,
        "ConfidenceCalibration_mean": 0.0,
        "ConfidenceCalibration_std": 0.0,
        "ConfidenceCalibration_min": 0.0,
        "ConfidenceCalibration_max": 0.0,
        "OverallTrustworthiness_mean": 0.0,
        "OverallTrustworthiness_std": 0.0,
        "OverallTrustworthiness_min": 0.0,
        "OverallTrustworthiness_max": 0.0,
    }


def format_trustworthiness_summary(metrics: Dict[str, float]) -> str:
    """
    Format trustworthiness metrics as a publication-ready table.

    Args:
        metrics: Output from evaluate_trustworthiness_batch()

    Returns:
        Formatted string table
    """
    if not metrics or metrics.get("num_queries", 0) == 0:
        return "No results available"

    lines = [
        "=" * 70,
        "TRUSTWORTHINESS EVALUATION SUMMARY",
        "=" * 70,
        f"Total Queries: {metrics.get('num_queries', 0)}",
        "",
        "Metric                Mean     Std      Min      Max",
        "-" * 70,
    ]

    metric_order = [
        ("GroundingAccuracy", "Grounding Accuracy"),
        ("HallucinationRate", "Hallucination Rate"),
        ("CitationAccuracy", "Citation Accuracy"),
        ("EvidenceCoverage", "Evidence Coverage"),
        ("ConfidenceCalibration", "Confidence Calibration"),
        ("OverallTrustworthiness", "Overall Trustworthiness"),
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


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    print("=" * 70)
    print("TRUSTWORTHINESS METRICS DEMO")
    print("=" * 70)
    # Simple test with dummy data
    dummy_results = [
        {
            "quality_report": type('obj', (object,), {
                "grounding_accuracy": 0.95,
                "hallucination_detected": False,
                "confidence_score": 0.92
            })(),
            "citation_report": type('obj', (object,), {
                "citation_accuracy": 1.0,
                "evidence_coverage": 0.88
            })(),
        },
        {
            "quality_report": type('obj', (object,), {
                "grounding_accuracy": 0.80,
                "hallucination_detected": True,
                "confidence_score": 0.65
            })(),
            "citation_report": type('obj', (object,), {
                "citation_accuracy": 0.75,
                "evidence_coverage": 0.60
            })(),
        }
    ]
    metrics = evaluate_trustworthiness_batch(dummy_results)
    print("\n" + format_trustworthiness_summary(metrics))
    print("=" * 70)
    print("✅ TRUSTWORTHINESS METRICS DEMO COMPLETED")
    print("=" * 70)