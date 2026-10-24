"""
Statistical Significance Testing for RAG Evaluation.

Provides:
- Confidence intervals for all metrics
- Paired t-tests for comparing configurations
- Bootstrap sampling for robust estimates
"""

from __future__ import annotations

import logging
import math
from typing import List, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def calculate_confidence_interval(
    values: List[float],
    confidence_level: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calculate confidence interval using t-distribution.
    
    Args:
        values: List of metric values
        confidence_level: Confidence level (default 0.95)
        
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    if not values or len(values) < 2:
        mean = np.mean(values) if values else 0.0
        return mean, mean, mean
    
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    n = len(values)
    
    # Use t-distribution for small samples
    t_crit = stats.t.ppf((1 + confidence_level) / 2, n - 1)
    margin_of_error = t_crit * (std / math.sqrt(n))
    
    return float(mean), float(mean - margin_of_error), float(mean + margin_of_error)


def paired_t_test(
    values_a: List[float],
    values_b: List[float]
) -> Tuple[float, float]:
    """
    Perform paired t-test to compare two configurations.
    
    Args:
        values_a: Metric values from configuration A
        values_b: Metric values from configuration B
        
    Returns:
        Tuple of (t_statistic, p_value)
    """
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return 0.0, 1.0
    
    t_stat, p_value = stats.ttest_rel(values_a, values_b)
    
    return float(t_stat), float(p_value)


def bootstrap_confidence_interval(
    values: List[float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calculate confidence interval using bootstrap sampling.
    
    Args:
        values: List of metric values
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level
        
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    if not values:
        return 0.0, 0.0, 0.0
    
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    mean = np.mean(bootstrap_means)
    
    return float(mean), float(lower), float(upper)


def is_statistically_significant(
    p_value: float,
    alpha: float = 0.05
) -> bool:
    """
    Check if result is statistically significant.
    
    Args:
        p_value: p-value from statistical test
        alpha: Significance level (default 0.05)
        
    Returns:
        True if p_value < alpha
    """
    return p_value < alpha


# ============================================================
# Demo / Test
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    
    print("=" * 70)
    print("STATISTICAL TESTS DEMO")
    print("=" * 70)
    
    # Example data
    baseline_recall = [0.6, 0.65, 0.7, 0.62, 0.68]
    proposed_recall = [0.8, 0.85, 0.82, 0.88, 0.84]
    
    # Confidence interval
    mean, lower, upper = calculate_confidence_interval(proposed_recall)
    print(f"\n📊 95% Confidence Interval:")
    print(f"   Mean: {mean:.4f}")
    print(f"   Range: [{lower:.4f}, {upper:.4f}]")
    
    # Paired t-test
    t_stat, p_value = paired_t_test(baseline_recall, proposed_recall)
    print(f"\n📊 Paired t-test:")
    print(f"   t-statistic: {t_stat:.4f}")
    print(f"   p-value: {p_value:.6f}")
    print(f"   Significant: {is_statistically_significant(p_value)}")
    
    # Bootstrap
    mean_boot, lower_boot, upper_boot = bootstrap_confidence_interval(proposed_recall)
    print(f"\n📊 Bootstrap 95% CI:")
    print(f"   Mean: {mean_boot:.4f}")
    print(f"   Range: [{lower_boot:.4f}, {upper_boot:.4f}]")
    
    print("\n" + "=" * 70)
    print("✅ STATISTICAL TESTS DEMO COMPLETED")
    print("=" * 70)