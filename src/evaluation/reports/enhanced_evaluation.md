# RAG Evaluation Report

**Generated:** 2026-07-26 22:55:48

## Overall Performance

| Metric | Score |
|--------|-------|
| Retrieval_Score | 83.00% |
| Generation_Score | 90.00% |
| Trustworthiness_Score | 93.00% |
| Morphology_Score | 90.00% |
| RAG_Score | 89.00% |

## Retrieval Metrics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Recall@3 | 0.8500 | 0.0800 | 0.7000 | 0.9500 |
| MRR | 0.7800 | 0.1200 | 0.5500 | 0.9200 |
| NDCG@5 | 0.8200 | 0.0900 | 0.6800 | 0.9300 |

## Generation Metrics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Faithfulness | 0.9200 | 0.0500 | 0.0000 | 0.0000 |
| Answer_Relevance | 0.8800 | 0.0700 | 0.0000 | 0.0000 |

## Trustworthiness Metrics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Grounding_Accuracy | 0.9500 | 0.0300 | 0.0000 | 0.0000 |
| Citation_Accuracy | 0.9800 | 0.0200 | 0.0000 | 0.0000 |
| Hallucination_Rate | 0.0500 | 0.0200 | 0.0000 | 0.0000 |

## Morphology Metrics

| Metric | Value |
|--------|-------|
| Morphology_Detection_Accuracy | 0.8900 |
| Morphology_Routing_Accuracy | 0.9200 |
| Affix_Handling_Score | 0.8500 |

## Error Analysis

| Error Category | Count |
|----------------|-------|
| retrieval_failure | 3 |
| hallucination | 2 |
| citation_error | 1 |
| morphology_failure | 0 |

## Summary

- **Total Queries Evaluated:** 50
- **Overall RAG Score:** 89.00%
