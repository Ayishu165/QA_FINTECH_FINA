#!/usr/bin/env python3
"""
src/evaluation/faithfulness_metrics.py

Faithfulness evaluation: answer grounding against context.
"""

import re
from typing import List, Dict, Any


class FaithfulnessEvaluator:
    """Evaluate faithfulness of an answer against its context."""

    def split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'[።.!?]', text)
        return [s.strip() for s in sentences if s.strip()]

    def sentence_support_score(self, sentence: str, context: str) -> float:
        sentence_words = set(sentence.split())
        context_words = set(context.split())
        if not sentence_words:
            return 0.0
        overlap = sentence_words & context_words
        return len(overlap) / len(sentence_words)

    def evaluate(self, answer: str, context: str) -> Dict[str, Any]:
        sentences = self.split_sentences(answer)
        if not sentences:
            return {"faithfulness": 0.0, "hallucination_risk": 1.0, "supported_sentences": 0, "total_sentences": 0}

        scores = [self.sentence_support_score(s, context) for s in sentences]
        faithfulness = sum(scores) / len(scores)
        supported = sum(1 for s in scores if s >= 0.5)

        return {
            "faithfulness": round(faithfulness, 4),
            "hallucination_risk": round(1 - faithfulness, 4),
            "supported_sentences": supported,
            "total_sentences": len(sentences)
        }


def compute_all_faithfulness_metrics(
    answers: List[str],
    contexts: List[str]
) -> Dict[str, float]:
    """Compute average faithfulness over multiple examples."""
    evaluator = FaithfulnessEvaluator()
    metrics = {"faithfulness": 0.0, "hallucination_risk": 0.0, "supported_sentences": 0.0, "total_sentences": 0.0}
    count = 0

    for ans, ctx in zip(answers, contexts):
        result = evaluator.evaluate(ans, ctx)
        metrics["faithfulness"] += result["faithfulness"]
        metrics["hallucination_risk"] += result["hallucination_risk"]
        metrics["supported_sentences"] += result["supported_sentences"]
        metrics["total_sentences"] += result["total_sentences"]
        count += 1

    if count > 0:
        for key in metrics:
            metrics[key] /= count

    return {k: round(v, 4) for k, v in metrics.items()}