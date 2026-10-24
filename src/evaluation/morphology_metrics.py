"""
Morphology-Specific Metrics for Amharic FinTech RAG.

Validates the core novelty: morphology-aware query analysis and retrieval.

Metrics:
- Morphology Detection Accuracy: Does the system correctly identify complex queries?
- Morphology Routing Accuracy: Does it route to hybrid_morpho_expanded when needed?
- Affix Handling Score: Does retrieval handle inflected forms correctly?
- Compound Word Recognition: Does it preserve compound integrity?
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def calculate_morphology_detection_accuracy(
    predicted_complexity: List[Dict[str, Any]],
    gold_complexity: List[Dict[str, Any]]
) -> float:
    """
    Calculate how accurately the system detects morphological complexity.
    
    Args:
        predicted_complexity: List of dicts with 'morphology_score' and 'level'
        gold_complexity: List of dicts with ground truth 'morphology_score' and 'level'
        
    Returns:
        Accuracy (0.0-1.0) based on level agreement
    """
    if not predicted_complexity or not gold_complexity:
        return 0.0
    
    correct = 0
    total = min(len(predicted_complexity), len(gold_complexity))
    
    for pred, gold in zip(predicted_complexity, gold_complexity):
        pred_level = pred.get("level", "unknown")
        gold_level = gold.get("level", "unknown")
        
        if pred_level == gold_level:
            correct += 1
    
    return correct / total if total > 0 else 0.0


def calculate_morphology_routing_accuracy(
    routing_decisions: List[Dict[str, Any]],
    gold_routing: List[Dict[str, Any]]
) -> float:
    """
    Calculate how accurately the system routes morphology-complex queries.
    
    Expected behavior:
    - If morphology_score >= 0.65 → should use 'hybrid_morpho_expanded'
    - Otherwise → can use any strategy
    
    Args:
        routing_decisions: List of dicts with 'strategy' and 'morphology_score'
        gold_routing: List of dicts with 'expected_strategy' and 'morphology_score'
        
    Returns:
        Routing accuracy (0.0-1.0)
    """
    if not routing_decisions or not gold_routing:
        return 0.0
    
    correct = 0
    total = 0
    
    for pred, gold in zip(routing_decisions, gold_routing):
        pred_strategy = pred.get("strategy", "")
        gold_strategy = gold.get("expected_strategy", "")
        morpho_score = pred.get("morphology_score", 0.0)
        
        # Only evaluate routing for high-morphology queries
        if morpho_score >= 0.65:
            total += 1
            if pred_strategy == "hybrid_morpho_expanded":
                correct += 1
    
    return correct / total if total > 0 else 1.0


def calculate_affix_handling_score(
    queries: List[str],
    retrieved_chunks: List[List[str]],
    gold_chunks: List[List[str]]
) -> float:
    """
    Calculate how well retrieval handles queries with complex affixes.
    
    For queries with high affix density, check if the system retrieves
    chunks containing the root forms.
    
    Args:
        queries: List of query strings
        retrieved_chunks: List of retrieved chunk IDs per query
        gold_chunks: List of gold chunk IDs per query
        
    Returns:
        Affix handling score (0.0-1.0)
    """
    if not queries or not retrieved_chunks:
        return 0.0
    
    scores = []
    
    for i, query in enumerate(queries):
        # Simple affix detection (you can enhance with your QueryAnalyzer)
        import re
        amharic_words = re.findall(r"[\u1200-\u135F]+", query)
        
        if not amharic_words:
            continue
        
        # Check if query has affixes
        prefix_pattern = re.compile(r"^(መ|ተ|አ|ይ|አይ|የ|ለ|በ|ከ|እ|ወ|ስለ)")
        suffix_pattern = re.compile(r"(ች|ዎች|ው|ዋ|ሁ|ህ|ሽ|ን|ም|ነው|ናቸው|አለ|አል)$")
        
        has_affixes = any(
            prefix_pattern.search(w) or suffix_pattern.search(w)
            for w in amharic_words
        )
        
        if has_affixes:
            # Calculate recall for this query
            retrieved = set(retrieved_chunks[i])
            gold = set(gold_chunks[i])
            
            if gold:
                recall = len(retrieved & gold) / len(gold)
                scores.append(recall)
    
    return float(np.mean(scores)) if scores else 0.0


def calculate_compound_word_preservation(
    queries: List[str],
    retrieved_chunks_text: List[List[str]]
) -> float:
    """
    Calculate how well the system preserves compound Amharic words.
    
    Args:
        queries: List of query strings
        retrieved_chunks_text: List of retrieved chunk texts per query
        
    Returns:
        Compound preservation score (0.0-1.0)
    """
    if not queries or not retrieved_chunks_text:
        return 0.0
    
    scores = []
    
    # Common Amharic compound words in FinTech
    compound_words = [
        "የቁጠባ ሂሳብ",  # savings account
        "የብድር ወለድ",  # loan interest
        "የካርድ ገደብ",  # card limit
        "የሞባይል ባንኪንግ",  # mobile banking
    ]
    
    for i, query in enumerate(queries):
        query_compounds = [cw for cw in compound_words if cw in query]
        
        if not query_compounds:
            continue
        
        # Check if retrieved chunks preserve these compounds
        chunks_text = " ".join(retrieved_chunks_text[i])
        preserved = sum(1 for cw in query_compounds if cw in chunks_text)
        
        scores.append(preserved / len(query_compounds))
    
    return float(np.mean(scores)) if scores else 0.0


def evaluate_morphology_batch(
    morphology_results: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Evaluate morphology metrics across a batch of results.
    
    Args:
        morphology_results: List of dicts with:
            - 'predicted_complexity': Dict with morphology_score, level
            - 'gold_complexity': Dict with morphology_score, level
            - 'routing_decision': Dict with strategy, morphology_score
            - 'expected_routing': Dict with expected_strategy
            - 'query': str
            - 'retrieved_chunks': List[str]
            - 'gold_chunks': List[str]
            - 'retrieved_chunks_text': List[str]
            
    Returns:
        Dict with morphology metrics
    """
    if not morphology_results:
        return {
            "Morphology_Detection_Accuracy": 0.0,
            "Morphology_Routing_Accuracy": 0.0,
            "Affix_Handling_Score": 0.0,
            "Compound_Word_Preservation": 0.0,
            "num_queries": 0
        }
    
    # Extract lists
    predicted_complexity = [r.get("predicted_complexity", {}) for r in morphology_results]
    gold_complexity = [r.get("gold_complexity", {}) for r in morphology_results]
    routing_decisions = [r.get("routing_decision", {}) for r in morphology_results]
    gold_routing = [r.get("expected_routing", {}) for r in morphology_results]
    queries = [r.get("query", "") for r in morphology_results]
    retrieved_chunks = [r.get("retrieved_chunks", []) for r in morphology_results]
    gold_chunks = [r.get("gold_chunks", []) for r in morphology_results]
    retrieved_chunks_text = [r.get("retrieved_chunks_text", []) for r in morphology_results]
    
    # Calculate metrics
    metrics = {
        "Morphology_Detection_Accuracy": calculate_morphology_detection_accuracy(
            predicted_complexity, gold_complexity
        ),
        "Morphology_Routing_Accuracy": calculate_morphology_routing_accuracy(
            routing_decisions, gold_routing
        ),
        "Affix_Handling_Score": calculate_affix_handling_score(
            queries, retrieved_chunks, gold_chunks
        ),
        "Compound_Word_Preservation": calculate_compound_word_preservation(
            queries, retrieved_chunks_text
        ),
        "num_queries": len(morphology_results)
    }
    
    logger.info(
        f"✅ Morphology evaluation complete: "
        f"Detection={metrics['Morphology_Detection_Accuracy']:.4f}, "
        f"Routing={metrics['Morphology_Routing_Accuracy']:.4f}"
    )
    
    return metrics


# ============================================================
# Demo / Test
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    
    print("=" * 70)
    print("MORPHOLOGY METRICS DEMO")
    print("=" * 70)
    
    # Mock data
    morphology_results = [
        {
            "predicted_complexity": {"morphology_score": 0.68, "level": "high"},
            "gold_complexity": {"morphology_score": 0.65, "level": "high"},
            "routing_decision": {"strategy": "hybrid_morpho_expanded", "morphology_score": 0.68},
            "expected_routing": {"expected_strategy": "hybrid_morpho_expanded"},
            "query": "የCBE የቁጠባ ሂሳብ እንዴት መክፈት እችላለሁ?",
            "retrieved_chunks": ["chunk_001", "chunk_002"],
            "gold_chunks": ["chunk_001", "chunk_002"],
            "retrieved_chunks_text": ["የቁጠባ ሂሳብ ለመክፈት...", "መታወቂያ ካርድ..."]
        },
        {
            "predicted_complexity": {"morphology_score": 0.45, "level": "medium"},
            "gold_complexity": {"morphology_score": 0.42, "level": "medium"},
            "routing_decision": {"strategy": "hybrid", "morphology_score": 0.45},
            "expected_routing": {"expected_strategy": "hybrid"},
            "query": "የብድር ወለድ ስንት ነው?",
            "retrieved_chunks": ["chunk_003"],
            "gold_chunks": ["chunk_003"],
            "retrieved_chunks_text": ["የብድር ወለድ 12% ነው።"]
        }
    ]
    
    metrics = evaluate_morphology_batch(morphology_results)
    
    print("\n📊 Morphology Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    print("\n" + "=" * 70)
    print("✅ MORPHOLOGY METRICS DEMO COMPLETED")
    print("=" * 70)