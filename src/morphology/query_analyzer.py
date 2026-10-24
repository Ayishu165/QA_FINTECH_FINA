# src/morphology/query_analyzer.py

"""
Morphology-Aware Query Analyzer for Adaptive Amharic FinTech RAG.

This module processes live user queries in real-time, extracting linguistic, 
morphological, domain, and intent features. It enforces the "Morphology Gating" 
mechanism to ensure highly inflected Amharic queries trigger morphological 
query expansion, directly operationalizing the core research novelty.

Design note on Morphology Gating (Critical for Q1 Novelty Defense)
----------------------------------------------------------------------
To maintain consistency with the offline ComplexityEstimator, this module 
calculates a real-time morphological complexity score. If it exceeds the 
0.65 threshold, it overrides standard routing and forces 'hybrid_morpho_expanded' 
retrieval, preventing embedding sparsity for complex Amharic queries.

All thresholds, weights, and normalization bounds are aligned with those used
in `complexity_estimator.py` to ensure consistency across the pipeline.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Optional

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """
    Analyzes live Amharic FinTech queries to generate adaptive retrieval 
    configurations based on morphological, lexical, and domain complexity.
    """

    def __init__(
        self,
        lexicon_dir: Optional[Path] = None,
        low_threshold: float = 0.30,
        high_threshold: float = 0.60,
        morpho_override_threshold: float = 0.65,
    ):
        """
        Initialize the QueryAnalyzer.

        Parameters:
            lexicon_dir: Path to directory containing:
                - banking_keywords.txt
                - financial_entities.txt
                If not provided, uses PROJECT_ROOT / "data" / "lexicons".
            low_threshold: Complexity below this is "low" (0.30 default)
            high_threshold: Complexity above this is "high" (0.60 default)
            morpho_override_threshold: Morphology above this triggers gating (0.65 default)
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.morpho_override_threshold = morpho_override_threshold

        # Component weights (aligned with complexity_estimator)
        self.component_weights = {
            "morphological": 0.45,
            "lexical": 0.25,
            "structural": 0.10,
            "domain": 0.20,
        }

        # Determine lexicon directory
        if lexicon_dir is None:
            PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
            lexicon_dir = PROJECT_ROOT / "data" / "lexicons"

        # Load lexicons
        self.banking_keywords = self._load_word_list(lexicon_dir / "banking_keywords.txt")
        self.financial_entities = self._load_word_list(lexicon_dir / "financial_entities.txt")

        # Fallback entities if lexicon files are missing
        self.fallback_entities = {
            "CBE": {"name": "Commercial Bank of Ethiopia", "type": "Bank"},
            "ንግድ ባንክ": {"name": "Commercial Bank of Ethiopia", "type": "Bank"},
            "ሲቢኢ": {"name": "Commercial Bank of Ethiopia", "type": "Bank"},
            "ብሔራዊ ባንክ": {"name": "National Bank of Ethiopia", "type": "Regulator"},
            "ቴሌብር": {"name": "Telebirr", "type": "Digital Payment"},
            "ኢትዮ ቴሌኮም": {"name": "Ethio Telecom", "type": "Telecom"},
        }

        # Robust Morphology Regex Patterns (Aligned with feature_extractor.py)
        self._prefix_pattern = re.compile(r"^(መ|ተ|አ|ይ|አይ|የ|ለ|በ|ከ|እ|ወ|ስለ)")
        self._suffix_pattern = re.compile(r"(ች|ዎች|ው|ዋ|ሁ|ህ|ሽ|ን|ም|ነው|ናቸው|አለ|አል)$")
        self._amharic_word_pattern = re.compile(r"[\u1200-\u135F\u1369-\u137F]+")

        # Intent & Category Keywords
        self.intent_keywords = {
            "procedure": ["እንዴት", "ማግኘት", "መክፈት", "መጠቀም", "አሰራር", "መመዝገብ"],
            "definition": ["ምን", "ምንድን", "ማለት", "ማን", "ማንነት"],
            "numerical": ["ስንት", "መጠን", "በመቶ", "ዋጋ", "ክፍያ", "ብር"],
            "comparison": ["ልዩነት", "ከ", "ወይም", "ተመሳሳይ", "የተሻለ"],
        }
        self.category_keywords = {
            "card_registration": ["ካርድ", "Visa", "Master", "ፕላቲነም", "ጎልድ"],
            "account_opening": ["ሂሳብ", "መክፈት", "መመዝገብ"],
            "loan": ["ብድር", "ትርፍ", "ማፅደቅ"],
            "interest_rate": ["ወለድ", "መጠን", "ተመን"],
            "mobile_banking": ["ሞባይል", "አፕ", "መተግበሪያ"],
            "digital_payment": ["ክፍያ", "ቴሌብር", "መላክ"],
        }
        self.ambiguous_terms = {"ካርድ", "ገንዘብ", "ሂሳብ", "ባንክ"}

        logger.info(
            f"QueryAnalyzer initialized with thresholds: "
            f"low={self.low_threshold}, high={self.high_threshold}, "
            f"morpho_override={self.morpho_override_threshold}, "
            f"loaded {len(self.banking_keywords)} banking keywords, "
            f"{len(self.financial_entities)} financial entities."
        )

    # ========================================================
    # RESOURCE LOADING
    # ========================================================

    def _load_word_list(self, path: Path) -> Set[str]:
        """Load a newline‑separated text file into a set of lowercase strings."""
        if not path.exists():
            logger.warning(f"Lexicon file not found: {path}")
            return set()
        with path.open("r", encoding="utf-8") as f:
            return {line.strip().lower() for line in f if line.strip()}

    # ========================================================
    # UTILITIES
    # ========================================================

    def _clean_query(self, query: str) -> str:
        """Strips and normalizes whitespace."""
        return re.sub(r"\s+", " ", query.strip())

    def _safe_div(self, a: float, b: float) -> float:
        """Safely divides two numbers, returning 0.0 on division by zero."""
        return a / b if b != 0 else 0.0

    def _clip(self, value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
        """Clips a value to the specified range."""
        return max(minimum, min(maximum, value))

    def _normalize(self, value: float, minimum: float, maximum: float) -> float:
        """Performs min-max normalization with safe clipping."""
        if maximum <= minimum:
            return 0.0
        return self._clip((value - minimum) / (maximum - minimum))

    def _tokenize(self, query: str) -> List[str]:
        """Simple whitespace tokenization for query analysis."""
        return [token for token in query.split() if token.strip()]

    # ========================================================
    # FEATURE EXTRACTION
    # ========================================================

    def _extract_query_features(self, query: str) -> Dict[str, Any]:
        """Extracts general statistical and lexical features from the query."""
        tokens = self._tokenize(query)
        token_count = len(tokens)
        unique_count = len(set(tokens))
        
        english_terms = [t for t in tokens if re.search(r"[A-Za-z]", t)]
        numbers = [t for t in tokens if re.search(r"\d", t)]
        contains_currency = any(symbol in query for symbol in ["ብር", "$", "ETB", "ዶላር"])

        if token_count <= 4:
            length_category = "short"
        elif token_count <= 10:
            length_category = "medium"
        else:
            length_category = "long"

        return {
            "token_count": token_count,
            "unique_token_count": unique_count,
            "character_count": len(query),
            "average_token_length": round(self._safe_div(sum(len(t) for t in tokens), token_count), 3),
            "question_length_category": length_category,
            "contains_number": len(numbers) > 0,
            "contains_currency": contains_currency,
            "contains_english_term": len(english_terms) > 0,
            "lexical_diversity": round(self._safe_div(unique_count, token_count), 4)
        }

    def _extract_morphology(self, query: str) -> Dict[str, Any]:
        """
        Extracts morphological features using robust regex patterns, 
        preventing mid-word fracturing and ensuring accurate affix counting.
        """
        tokens = self._tokenize(query)
        amharic_words = self._amharic_word_pattern.findall(query)
        
        prefix_count = 0
        suffix_count = 0

        for word in amharic_words:
            if self._prefix_pattern.search(word):
                prefix_count += 1
            if self._suffix_pattern.search(word):
                suffix_count += 1

        word_count = len(amharic_words) if amharic_words else 1
        morpheme_total = word_count + prefix_count + suffix_count

        avg_morphemes = self._safe_div(morpheme_total, word_count)
        prefix_density = self._safe_div(prefix_count, word_count)
        suffix_density = self._safe_div(suffix_count, word_count)
        affix_density = self._safe_div(prefix_count + suffix_count, word_count)

        # Calculate morphological complexity using the same formula as complexity_estimator
        affix_score = self._normalize(affix_density, 0.0, 1.0)
        morpheme_score = self._normalize(avg_morphemes, 1.0, 3.0)
        compound_score = 0.0  # Not detected in queries
        prefix_score = self._normalize(prefix_density, 0.0, 0.6)
        suffix_score = self._normalize(suffix_density, 0.0, 0.4)

        morphology_score = (
            affix_score * 0.35
            + morpheme_score * 0.25
            + compound_score * 0.15
            + prefix_score * 0.15
            + suffix_score * 0.10
        )
        morphology_score = round(self._clip(morphology_score), 4)

        return {
            "total_amharic_words": len(amharic_words),
            "total_prefixes": prefix_count,
            "total_suffixes": suffix_count,
            "average_morphemes_per_word": round(avg_morphemes, 3),
            "prefix_density": round(prefix_density, 4),
            "suffix_density": round(suffix_density, 4),
            "affix_density": round(affix_density, 4),
            "morphology_complexity_score": morphology_score,
            "analysis_confidence": 0.95
        }

    def _detect_entity(self, query: str) -> Dict[str, Any]:
        """Detects known FinTech entities in the query."""
        # First, check loaded financial_entities
        tokens = self._tokenize(query)
        for token in tokens:
            if token.lower() in self.financial_entities:
                return {
                    "name": token,
                    "type": "FinancialEntity",
                    "confidence": 0.90
                }

        # Then check fallback entities
        for key, value in self.fallback_entities.items():
            if key in query:
                return {
                    "name": value["name"],
                    "type": value["type"],
                    "confidence": 0.98
                }
        return {"name": None, "type": None, "confidence": 0.0}

    # ========================================================
    # INTENT, CATEGORY & AMBIGUITY
    # ========================================================

    def _detect_intent(self, query: str) -> Dict[str, Any]:
        """Classifies the user's underlying question intent."""
        for intent, keywords in self.intent_keywords.items():
            if any(word in query for word in keywords):
                return {"question_type": intent, "confidence": 0.92}
        return {"question_type": "general", "confidence": 0.80}

    def _detect_query_category(self, query: str) -> str:
        """Categorizes the query into a specific FinTech domain."""
        for category, keywords in self.category_keywords.items():
            if any(word in query for word in keywords):
                return category
        return "general"

    def _calculate_ambiguity(self, query: str) -> Dict[str, Any]:
        """Estimates lexical ambiguity based on polysemous FinTech terms."""
        tokens = set(self._tokenize(query))
        matches = len(tokens.intersection(self.ambiguous_terms))
        score = self._clip(matches / 3.0)
        
        level = "low" if score < 0.20 else ("medium" if score < 0.50 else "high")
        return {"ambiguity_score": round(score, 3), "ambiguity_level": level}

    # ========================================================
    # COMPLEXITY & ROUTING
    # ========================================================

    def _compute_lexical_complexity(self, features: Dict[str, Any]) -> float:
        """Lexical complexity for queries (simplified)."""
        token_count = features["token_count"]
        lexical_diversity = features["lexical_diversity"]
        token_score = self._normalize(token_count, 0.0, 20.0)
        div_score = lexical_diversity  # already 0-1
        return round(self._clip((token_score * 0.4 + div_score * 0.6)), 4)

    def _compute_structural_complexity(self, features: Dict[str, Any]) -> float:
        """Structural complexity for queries based on sentence length."""
        token_count = features["token_count"]
        return self._normalize(token_count, 1.0, 20.0)

    def _compute_domain_complexity(self, query: str, features: Dict[str, Any]) -> float:
        """Domain complexity using banking keywords and entity presence."""
        tokens = self._tokenize(query.lower())
        if not tokens:
            return 0.0

        banking_hits = sum(1 for t in tokens if t in self.banking_keywords)
        keyword_score = self._normalize(banking_hits / len(tokens), 0.0, 0.5)
        currency_score = self._normalize(
            sum(1 for t in tokens if t in {"ብር", "etb", "$"}),
            0.0, 2.0
        )
        score = (keyword_score * 0.6) + (currency_score * 0.4)
        return round(self._clip(score), 4)

    def _estimate_complexity(
        self,
        query: str,
        features: Dict[str, Any],
        morphology: Dict[str, Any],
        entity: Dict[str, Any],
        intent: Dict[str, Any],
        ambiguity: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculates multi-dimensional query complexity scores.
        Uses the same weights and normalization as complexity_estimator.
        """
        # 1. Morphological (already computed)
        morphology_score = morphology["morphology_complexity_score"]

        # 2. Lexical
        lexical_score = self._compute_lexical_complexity(features)

        # 3. Structural
        structural_score = self._compute_structural_complexity(features)

        # 4. Domain
        domain_score = self._compute_domain_complexity(query, features)
        if entity["name"]:
            domain_score = self._clip(domain_score + 0.2)

        # 5. Overall (weighted sum - aligned with complexity_estimator)
        overall = (
            morphology_score * self.component_weights["morphological"]
            + lexical_score * self.component_weights["lexical"]
            + structural_score * self.component_weights["structural"]
            + domain_score * self.component_weights["domain"]
        )
        overall = round(self._clip(overall), 4)

        # Level
        if overall < self.low_threshold:
            level = "low"
        elif overall < self.high_threshold:
            level = "medium"
        else:
            level = "high"

        return {
            "overall": overall,
            "lexical": round(lexical_score, 4),
            "semantic": 0.5,  # placeholder, not computed
            "morphological": round(morphology_score, 4),
            "structural": round(structural_score, 4),
            "domain": round(domain_score, 4),
            "ambiguity": ambiguity["ambiguity_score"],
            "level": level,
            "confidence": 0.95
        }

    def _recommend_retrieval(
        self,
        complexity: Dict[str, Any],
        morphology: Dict[str, Any],
        intent: Dict[str, Any],
        entity: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generates adaptive retrieval configuration.
        
        NOVELTY ENFORCEMENT: If morphological complexity is high (>= 0.65), 
        it overrides the overall score to force morphological query expansion.
        """
        # 1. Morphology Gating Override
        if morphology["morphology_complexity_score"] >= self.morpho_override_threshold:
            return {
                "strategy": "hybrid_morpho_expanded",
                "dense_weight": 0.50,
                "sparse_weight": 0.50,
                "top_k": 15,
                "rerank": True,
                "apply_morpho_expansion": True,
                "reason": "High morphological complexity detected in query; forcing morphological query expansion to prevent embedding sparsity."
            }

        # 2. Intent-Specific Overrides
        if intent["question_type"] == "numerical":
            return {
                "strategy": "hybrid",
                "dense_weight": 0.40,
                "sparse_weight": 0.60,
                "top_k": 10,
                "rerank": False,
                "apply_morpho_expansion": False,
                "reason": "Numerical query detected; prioritizing sparse (BM25) retrieval for exact value matching."
            }

        # 3. Standard Complexity-Based Routing
        if complexity["overall"] < self.low_threshold:
            return {
                "strategy": "dense",
                "dense_weight": 0.80,
                "sparse_weight": 0.20,
                "top_k": 5,
                "rerank": False,
                "apply_morpho_expansion": False,
                "reason": "Low overall complexity; semantic dense retrieval is sufficient."
            }
        elif complexity["overall"] < self.high_threshold:
            return {
                "strategy": "hybrid",
                "dense_weight": 0.60,
                "sparse_weight": 0.40,
                "top_k": 8,
                "rerank": False,
                "apply_morpho_expansion": False,
                "reason": "Moderate complexity; combining semantic and lexical retrieval."
            }
        else:
            return {
                "strategy": "hybrid_rerank",
                "dense_weight": 0.50,
                "sparse_weight": 0.50,
                "top_k": 12,
                "rerank": True,
                "apply_morpho_expansion": False,
                "reason": "High overall complexity; utilizing lexical matching and cross-encoder reranking."
            }

    # ========================================================
    # SIGNALS (For Explainability)
    # ========================================================

    def _generate_signals(
        self,
        query: str,
        features: Dict[str, Any],
        morphology: Dict[str, Any],
        entity: Dict[str, Any]
    ) -> Dict[str, bool]:
        """Generate human-interpretable boolean flags for explainability."""
        tokens = self._tokenize(query.lower())
        return {
            "high_morphological_density": morphology["affix_density"] >= 0.35,
            "high_morpheme_count": morphology["average_morphemes_per_word"] >= 1.8,
            "high_prefix_density": morphology["prefix_density"] >= 0.40,
            "high_suffix_density": morphology["suffix_density"] >= 0.25,
            "high_lexical_entropy": False,  # not computed for queries
            "rare_vocabulary": False,
            "high_type_token_ratio": features["lexical_diversity"] >= 0.60,
            "long_sentences": features["token_count"] >= 20,
            "financial_domain_heavy": (
                entity["name"] is not None
                or features["contains_currency"]
                or any(t in self.banking_keywords for t in tokens)
            ),
            "high_abbreviation_density": False,
            "contains_number": features["contains_number"],
            "contains_english": features["contains_english_term"],
        }

    # ========================================================
    # MAIN ANALYSIS PIPELINE
    # ========================================================

    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Orchestrates the complete real-time query analysis pipeline.

        Returns:
            dict with all extracted information, signals, and retrieval recommendation.
        """
        start_time = time.time()
        timing = {}

        # 1. Cleaning
        t0 = time.time()
        cleaned = self._clean_query(query)
        timing["text_cleaning"] = round((time.time() - t0) * 1000, 3)

        # 2. Features
        t0 = time.time()
        features = self._extract_query_features(cleaned)
        timing["feature_extraction"] = round((time.time() - t0) * 1000, 3)

        # 3. Morphology
        t0 = time.time()
        morphology = self._extract_morphology(cleaned)
        timing["morphology_analysis"] = round((time.time() - t0) * 1000, 3)

        # 4. Entity
        entity = self._detect_entity(cleaned)

        # 5. Intent & category
        intent = self._detect_intent(cleaned)
        category = self._detect_query_category(cleaned)

        # 6. Ambiguity
        ambiguity = self._calculate_ambiguity(cleaned)

        # 7. Complexity
        complexity = self._estimate_complexity(
            cleaned, features, morphology, entity, intent, ambiguity
        )

        # 8. Signals
        signals = self._generate_signals(cleaned, features, morphology, entity)

        # 9. Retrieval recommendation
        retrieval = self._recommend_retrieval(complexity, morphology, intent, entity)

        # 10. Routing reasons
        reasons = []
        if intent["question_type"] != "general":
            reasons.append(f"{intent['question_type']} intent detected")
        if entity["name"]:
            reasons.append(f"Entity identified: {entity['name']}")
        if morphology["morphology_complexity_score"] >= self.morpho_override_threshold:
            reasons.append("Morphology gating override triggered")
        reasons.append(f"Complexity level: {complexity['level']}")

        # 11. Total time
        total_time_ms = (time.time() - start_time) * 1000
        timing["total_processing_time_ms"] = round(total_time_ms, 3)

        return {
            "query": query,
            "cleaned_query": cleaned,
            "query_features": features,
            "morphology": morphology,
            "entity": entity,
            "intent": {
                "question_type": intent["question_type"],
                "query_category": category,
                "confidence": intent["confidence"]
            },
            "complexity": complexity,
            "signals": signals,
            "retrieval_recommendation": retrieval,
            "routing_reason": reasons,
            "timing": timing,
            "quality": {
                "feature_confidence": 0.96,
                "language": "am",
                "pipeline_version": "2.0"
            }
        }


# ========================================================
# TEST EXECUTION
# ========================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    analyzer = QueryAnalyzer()

    test_queries = [
        "የCBE ካርድ እንዴት ማግኘት እችላለሁ?",  # Medium complexity, procedure
        "የብድር ወለድ መጠን ስንት ነው?",         # Numerical, domain-heavy
        "ይከፈላልን?",                          # HIGH morphological complexity
        "የባንክ ሂሳብ ምንድን ነው?"               # Definition, low-medium complexity
    ]

    print("=" * 80)
    print("QUERY ANALYZER DEMO: Real-Time Adaptive Routing")
    print("=" * 80)
    
    for q in test_queries:
        result = analyzer.analyze(q)
        print(f"\nQUERY: '{q}'")
        print(f"  Intent: {result['intent']['question_type']} | Category: {result['intent']['query_category']}")
        print(f"  Morphology Score: {result['morphology']['morphology_complexity_score']} | Overall: {result['complexity']['overall']}")
        print(f"  Strategy: {result['retrieval_recommendation']['strategy']} (Top-K: {result['retrieval_recommendation']['top_k']})")
        print(f"  Morpho Expansion: {result['retrieval_recommendation']['apply_morpho_expansion']}")
        print(f"  Reason: {result['retrieval_recommendation']['reason']}")
        print(f"  Time: {result['timing']['total_processing_time_ms']} ms")
        print(f"  Signals: {json.dumps(result['signals'], ensure_ascii=False, indent=2)}")