# src/retrieval/adaptive_router.py

"""
Adaptive Router for Amharic FinTech RAG.

⭐ THIS IS THE CORE NOVELTY MODULE ⭐

This module is the decision brain of the Query-Adaptive Morphology-Aware 
Retrieval framework. It analyzes the QueryProfile from query_analyzer.py 
and dynamically selects the optimal retrieval strategy based on:

1. Morphological Complexity (primary signal - protects Novelty 1)
2. Query Intent (procedure, numerical, definition, comparison)
3. Domain Entity Detection (bank names, abbreviations)
4. Overall Complexity Score

Strategy Selection Rules:
┌─────────────────────────────────────────────────────────────┐
│ Morphology Score >= 0.65 → hybrid_morpho_expanded (FORCED) │
│ Intent == numerical        → hybrid (sparse-heavy)          │
│ Intent == procedure        → hybrid (dense-heavy, high k)   │
│ Overall < 0.30             → dense                          │
│ Overall 0.30 - 0.60        → hybrid (balanced)              │
│ Overall >= 0.60            → hybrid_rerank                  │
└─────────────────────────────────────────────────────────────┘

The Morphology Gating Override ensures that highly inflected Amharic 
queries ALWAYS trigger morphological expansion, preventing embedding 
sparsity regardless of other complexity signals.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from src.retrieval.dense_retriever import DenseRetriever, RetrievalResult
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


# ==========================================================
# DATACLASSES
# ==========================================================

@dataclass
class RoutingDecision:
    """
    Encapsulates the routing decision made by the adaptive router.
    
    Attributes:
        strategy: Selected retrieval strategy name
        dense_weight: Weight for dense retriever (0.0-1.0)
        sparse_weight: Weight for sparse retriever (0.0-1.0)
        top_k: Number of final results to return
        use_reranker: Whether to apply reranking after retrieval
        apply_morpho_expansion: Whether to expand query with morphological roots
        reasoning: Human-readable explanation of the routing decision
    """
    strategy: str
    dense_weight: float
    sparse_weight: float
    top_k: int
    use_reranker: bool = False
    apply_morpho_expansion: bool = False
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "dense_weight": self.dense_weight,
            "sparse_weight": self.sparse_weight,
            "top_k": self.top_k,
            "use_reranker": self.use_reranker,
            "apply_morpho_expansion": self.apply_morpho_expansion,
            "reasoning": self.reasoning,
        }


@dataclass
class AdaptiveRetrievalResult:
    """
    Complete output from the adaptive router.
    
    Attributes:
        original_query: The raw user query
        expanded_query: The query after morphological expansion (or same as original)
        routing_decision: The RoutingDecision that was applied
        retrieved_chunks: Final ranked list of RetrievalResult
        timing_ms: Total routing + retrieval time in milliseconds
    """
    original_query: str
    expanded_query: str
    routing_decision: RoutingDecision
    retrieved_chunks: List[RetrievalResult] = field(default_factory=list)
    timing_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "expanded_query": self.expanded_query,
            "routing_decision": self.routing_decision.to_dict(),
            "retrieved_chunks": [r.to_dict() for r in self.retrieved_chunks],
            "timing_ms": round(self.timing_ms, 3),
            "chunks_returned": len(self.retrieved_chunks),
        }


# ==========================================================
# MAIN ROUTER CLASS
# ==========================================================

class AdaptiveRouter:
    """
    Morphology-aware adaptive retrieval router.
    
    This is the central decision-making module that selects retrieval 
    strategies based on query characteristics. It enforces the Morphology 
    Gating Override to protect the thesis's core novelty.
    """
    
    # Regex for extracting Amharic verb roots
    _VERB_PREFIX_PATTERN = re.compile(r"^(ይ|ት|እ|አ|ይ|አይ|ሌ|መ|ተ|አስ|አከ|እያ)")
    _VERB_SUFFIX_PATTERN = re.compile(r"(አለ|አል|ለሁ|ለህ|ለች|ለን|ላችሁ|ለሉ|ሎታል|ችሁ|ዋል|ኛል)$")
    
    # Default thresholds
    LOW_THRESHOLD = 0.30
    HIGH_THRESHOLD = 0.60
    MORPHO_OVERRIDE_THRESHOLD = 0.65
    
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        hybrid_retriever: HybridRetriever,
        low_threshold: float = 0.30,
        high_threshold: float = 0.60,
        morpho_override_threshold: float = 0.65,
    ) -> None:
        """
        Initialize the AdaptiveRouter with all retriever dependencies.
        
        Parameters:
            dense_retriever: Initialized DenseRetriever
            sparse_retriever: Initialized SparseRetriever
            hybrid_retriever: Initialized HybridRetriever
            low_threshold: Below this, complexity is "low"
            high_threshold: Above this, complexity is "high"
            morpho_override_threshold: Threshold for morphology gating override
        """
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.hybrid_retriever = hybrid_retriever
        
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.morpho_override_threshold = morpho_override_threshold
        
        logger.info(
            f"✅ AdaptiveRouter initialized:\n"
            f"   Morpho override: {self.morpho_override_threshold}\n"
            f"   Low threshold: {self.low_threshold}\n"
            f"   High threshold: {self.high_threshold}"
        )
    
    # ==========================================================
    # MAIN ROUTING METHOD
    # ==========================================================
    
    def route(
        self,
        query_text: str,
        query_vector: np.ndarray,
        query_profile: Dict[str, Any],
    ) -> AdaptiveRetrievalResult:
        """
        Analyze query profile and execute the optimal retrieval strategy.
        
        This is the main entry point for adaptive retrieval. It:
        1. Extracts signals from the query profile
        2. Makes a routing decision
        3. Optionally expands the query morphologically
        4. Dispatches to the appropriate retriever(s)
        5. Returns the final results
        
        Parameters:
            query_text: The raw user query string
            query_vector: Pre-computed query embedding (1024-dim numpy array)
            query_profile: Dict from query_analyzer.py
            
        Returns:
            AdaptiveRetrievalResult with strategy, reasoning, and retrieved chunks
        """
        start_time = time.time()
        
        # Step 1: Extract signals from query profile
        signals = self._extract_signals(query_profile)
        
        # Step 2: Make routing decision
        decision = self._make_routing_decision(signals)
        
        logger.info(
            f"🧠 Routing Decision:\n"
            f"   Strategy: {decision.strategy}\n"
            f"   Reasoning: {decision.reasoning}\n"
            f"   Weights: dense={decision.dense_weight}, sparse={decision.sparse_weight}\n"
            f"   Top-K: {decision.top_k}\n"
            f"   Reranker: {decision.use_reranker}\n"
            f"   Morpho Expansion: {decision.apply_morpho_expansion}"
        )
        
        # Step 3: Optional morphological query expansion
        active_query = query_text
        if decision.apply_morpho_expansion:
            expanded = self._expand_query_morphologically(query_text)
            if expanded != query_text:
                active_query = expanded
                logger.info(f"🔄 Query expanded: '{query_text}' → '{active_query}'")
        
        # Step 4: Dispatch to retriever(s)
        retrieved_chunks = self._dispatch_retrieval(
            query_text=active_query,
            query_vector=query_vector,
            decision=decision,
        )
        
        # Step 5: Package results
        total_time = (time.time() - start_time) * 1000
        
        result = AdaptiveRetrievalResult(
            original_query=query_text,
            expanded_query=active_query,
            routing_decision=decision,
            retrieved_chunks=retrieved_chunks,
            timing_ms=total_time,
        )
        
        logger.info(
            f"✅ Adaptive routing completed:\n"
            f"   Chunks returned: {len(retrieved_chunks)}\n"
            f"   Total time: {total_time:.1f}ms"
        )
        
        return result
    
    # ==========================================================
    # SIGNAL EXTRACTION
    # ==========================================================
    
    def _extract_signals(self, query_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract routing-relevant signals from the query profile.
        
        Handles missing fields gracefully with safe defaults.
        """
        morphology = query_profile.get("morphology", {})
        complexity = query_profile.get("complexity", {})
        intent = query_profile.get("intent", {})
        entity = query_profile.get("entity", {})
        features = query_profile.get("query_features", {})
        
        return {
            "morphology_score": float(morphology.get("morphology_complexity_score", 0.0)),
            "affix_density": float(morphology.get("affix_density", 0.0)),
            "avg_morphemes": float(morphology.get("average_morphemes_per_word", 1.0)),
            "overall_score": float(complexity.get("overall", 0.0)),
            "complexity_level": str(complexity.get("level", "medium")),
            "intent": str(intent.get("question_type", "general")),
            "category": str(intent.get("query_category", "general")),
            "entity_name": entity.get("name"),
            "entity_type": entity.get("type"),
            "contains_number": bool(features.get("contains_number", False)),
            "contains_currency": bool(features.get("contains_currency", False)),
            "token_count": int(features.get("token_count", 0)),
        }
    
    # ==========================================================
    # ROUTING DECISION
    # ==========================================================
    
    def _make_routing_decision(self, signals: Dict[str, Any]) -> RoutingDecision:
        """
        Apply routing rules to select the optimal retrieval strategy.
        
        Priority order:
        1. Morphology Gating Override (highest priority - protects core novelty)
        2. Intent-specific overrides
        3. Overall complexity-based routing
        """
        morpho_score = signals["morphology_score"]
        overall_score = signals["overall_score"]
        intent = signals["intent"]
        
        # ═══════════════════════════════════════════════════════════
        # RULE 1: Morphology Gating Override (CORE NOVELTY)
        # ═══════════════════════════════════════════════════════════
        if morpho_score >= self.morpho_override_threshold:
            return RoutingDecision(
                strategy="hybrid_morpho_expanded",
                dense_weight=0.50,
                sparse_weight=0.50,
                top_k=15,
                use_reranker=True,
                apply_morpho_expansion=True,
                reasoning=(
                    f"Morphology gating override triggered (score={morpho_score:.2f} >= "
                    f"{self.morpho_override_threshold}). Forcing morphological query expansion "
                    f"to prevent embedding sparsity for highly inflected Amharic text."
                ),
            )
        
        # ═══════════════════════════════════════════════════════════
        # RULE 2: Intent-Specific Overrides
        # ═══════════════════════════════════════════════════════════
        if intent == "numerical":
            return RoutingDecision(
                strategy="hybrid_lexical_heavy",
                dense_weight=0.35,
                sparse_weight=0.65,
                top_k=10,
                use_reranker=False,
                apply_morpho_expansion=False,
                reasoning=(
                    f"Numerical intent detected. Prioritizing sparse (BM25) retrieval "
                    f"(weight=0.65) for exact value and number matching."
                ),
            )
        
        if intent == "procedure":
            return RoutingDecision(
                strategy="hybrid_semantic_heavy",
                dense_weight=0.65,
                sparse_weight=0.35,
                top_k=10,
                use_reranker=False,
                apply_morpho_expansion=False,
                reasoning=(
                    f"Procedure intent detected. Prioritizing dense retrieval "
                    f"(weight=0.65) with higher top_k for broader contextual coverage."
                ),
            )
        
        if intent == "comparison":
            return RoutingDecision(
                strategy="hybrid_balanced",
                dense_weight=0.50,
                sparse_weight=0.50,
                top_k=12,
                use_reranker=True,
                apply_morpho_expansion=False,
                reasoning=(
                    f"Comparison intent detected. Using balanced hybrid retrieval "
                    f"with reranking to surface contrasting information."
                ),
            )
        
        # ═══════════════════════════════════════════════════════════
        # RULE 3: Overall Complexity-Based Routing
        # ═══════════════════════════════════════════════════════════
        if overall_score < self.low_threshold:
            return RoutingDecision(
                strategy="dense",
                dense_weight=0.80,
                sparse_weight=0.20,
                top_k=5,
                use_reranker=False,
                apply_morpho_expansion=False,
                reasoning=(
                    f"Low overall complexity ({overall_score:.2f} < {self.low_threshold}). "
                    f"Semantic dense retrieval is sufficient."
                ),
            )
        
        if overall_score < self.high_threshold:
            return RoutingDecision(
                strategy="hybrid",
                dense_weight=0.60,
                sparse_weight=0.40,
                top_k=8,
                use_reranker=False,
                apply_morpho_expansion=False,
                reasoning=(
                    f"Medium overall complexity ({overall_score:.2f}). "
                    f"Combining semantic and lexical retrieval."
                ),
            )
        
        # High complexity
        return RoutingDecision(
            strategy="hybrid_rerank",
            dense_weight=0.50,
            sparse_weight=0.50,
            top_k=12,
            use_reranker=True,
            apply_morpho_expansion=False,
            reasoning=(
                f"High overall complexity ({overall_score:.2f} >= {self.high_threshold}). "
                f"Using balanced hybrid retrieval with cross-encoder reranking."
            ),
        )
    
    # ==========================================================
    # QUERY EXPANSION
    # ==========================================================
    
    def _expand_query_morphologically(self, query_text: str) -> str:
        """
        Expand a highly inflected query by extracting morphological roots.
        
        This method strips common Amharic verb prefixes and suffixes to 
        isolate the root, then appends it to the query. This gives BM25 
        a chance to match the root concept even when the surface form 
        is highly inflected.
        
        Example: "ይከፈላልን?" → "ይከፈላልን? ከፈል"
        """
        # Find Amharic words in the query
        amharic_words = re.findall(r"[\u1200-\u135F\u1369-\u137F]+", query_text)
        
        if not amharic_words:
            return query_text
        
        roots = set()
        for word in amharic_words:
            # Only process words long enough to have meaningful affixes
            if len(word) < 4:
                continue
            
            root = word
            
            # Strip prefix
            prefix_match = self._VERB_PREFIX_PATTERN.match(root)
            if prefix_match:
                root = root[prefix_match.end():]
            
            # Strip suffix
            suffix_match = self._VERB_SUFFIX_PATTERN.search(root)
            if suffix_match:
                root = root[:suffix_match.start()]
            
            # Only add if the root is meaningfully different and long enough
            if root != word and len(root) >= 2:
                roots.add(root)
        
        if roots:
            expanded = query_text + " " + " ".join(roots)
            return expanded
        
        return query_text
    
    # ==========================================================
    # RETRIEVAL DISPATCH
    # ==========================================================
    
    def _dispatch_retrieval(
        self,
        query_text: str,
        query_vector: np.ndarray,
        decision: RoutingDecision,
    ) -> List[RetrievalResult]:
        """
        Execute the selected retrieval strategy.
        """
        strategy = decision.strategy
        
        # Dense retrieval
        if strategy == "dense":
            return self.dense_retriever.retrieve_with_vector(
                query_vector=query_vector,
                top_k=decision.top_k,
            )
        
        # Hybrid strategies (including all hybrid variants)
        if strategy in ("hybrid", "hybrid_lexical_heavy", "hybrid_semantic_heavy", 
                        "hybrid_balanced", "hybrid_morpho_expanded", "hybrid_rerank"):
            # Overfetch for better fusion coverage
            fetch_k = decision.top_k * 2
            
            dense_results = self.dense_retriever.retrieve_with_vector(
                query_vector=query_vector,
                top_k=fetch_k,
            )
            
            sparse_results = self.sparse_retriever.retrieve(
                query_text=query_text,
                top_k=fetch_k,
            )
            
            fused_results = self.hybrid_retriever.fuse(
                dense_results=dense_results,
                sparse_results=sparse_results,
                dense_weight=decision.dense_weight,
                sparse_weight=decision.sparse_weight,
                top_k=decision.top_k,
            )
            
            return fused_results
        
        # Fallback: dense only
        logger.warning(f"Unknown strategy '{strategy}', falling back to dense retrieval")
        return self.dense_retriever.retrieve_with_vector(
            query_vector=query_vector,
            top_k=decision.top_k,
        )
    
    # ==========================================================
    # UTILITY METHODS
    # ==========================================================
    
    def set_thresholds(
        self,
        low_threshold: Optional[float] = None,
        high_threshold: Optional[float] = None,
        morpho_override_threshold: Optional[float] = None,
    ) -> None:
        """Update routing thresholds."""
        if low_threshold is not None:
            self.low_threshold = low_threshold
        if high_threshold is not None:
            self.high_threshold = high_threshold
        if morpho_override_threshold is not None:
            self.morpho_override_threshold = morpho_override_threshold
        
        logger.info(
            f"✅ Thresholds updated:\n"
            f"   Low: {self.low_threshold}\n"
            f"   High: {self.high_threshold}\n"
            f"   Morpho override: {self.morpho_override_threshold}"
        )
    
    def get_thresholds(self) -> Dict[str, float]:
        """Get current thresholds."""
        return {
            "low_threshold": self.low_threshold,
            "high_threshold": self.high_threshold,
            "morpho_override_threshold": self.morpho_override_threshold,
        }


# ==========================================================
# DEMO / TEST
# ==========================================================

if __name__ == "__main__":
    import json
    import logging
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    print("=" * 70)
    print("ADAPTIVE ROUTER DEMO ⭐ CORE NOVELTY ⭐")
    print("=" * 70)

    # Paths
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
    CHROMA_DB_PATH = EMBEDDINGS_DIR / "chroma_db"
    CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "chunked_documents.json"
    LOCAL_MODEL_PATH = r"D:\models\multilingual-e5-large"

    print(f"\n📁 ChromaDB path: {CHROMA_DB_PATH}")
    print(f"📁 Corpus path: {CORPUS_PATH}")
    print(f"📁 Local model: {LOCAL_MODEL_PATH}")

    # Initialize dependencies
    from src.retrieval.vector_store import VectorStore
    from src.embeddings.embedding_model import get_embedding_model

    vector_store = VectorStore(
        persist_directory=CHROMA_DB_PATH,
        collection_name="amharic_fintech_corpus",
        metadata_path=EMBEDDINGS_DIR / "embedding_metadata.json",
    )

    print(f"\n✅ VectorStore ready: {vector_store.count()} vectors")

    embedding_model = get_embedding_model(
        local_model_path=LOCAL_MODEL_PATH,
        use_fp16=False,
        device="cpu",
    )

    dense_retriever = DenseRetriever(
        vector_store=vector_store,
        local_model_path=LOCAL_MODEL_PATH,
        device="cpu",
        use_fp16=False,
        top_k=10,
    )

    sparse_retriever = SparseRetriever(
        chunks_path=CORPUS_PATH,
        index_cache_path=PROJECT_ROOT / "data" / "retrieval" / "bm25_index.pkl",
    )

    hybrid_retriever = HybridRetriever(rrf_k=60)

    router = AdaptiveRouter(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        hybrid_retriever=hybrid_retriever,
    )

    # Test cases
    test_cases = [
        {
            "query": "የCBE ካርድ እንዴት ማግኘት እችላለሁ?",
            "profile": {
                "morphology": {"morphology_complexity_score": 0.58},
                "complexity": {"overall": 0.54, "level": "medium"},
                "intent": {"question_type": "procedure", "query_category": "card"},
                "entity": {"name": "Commercial Bank of Ethiopia", "type": "Bank"},
                "query_features": {"contains_number": False, "contains_currency": False},
            },
        },
        {
            "query": "የብድር ወለድ መጠን ስንት ነው?",
            "profile": {
                "morphology": {"morphology_complexity_score": 0.72},
                "complexity": {"overall": 0.57, "level": "medium"},
                "intent": {"question_type": "numerical", "query_category": "loan"},
                "entity": {"name": None, "type": None},
                "query_features": {"contains_number": False, "contains_currency": False},
            },
        },
        {
            "query": "ይከፈላልን?",
            "profile": {
                "morphology": {"morphology_complexity_score": 0.85},
                "complexity": {"overall": 0.53, "level": "medium"},
                "intent": {"question_type": "general", "query_category": "general"},
                "entity": {"name": None, "type": None},
                "query_features": {"contains_number": False, "contains_currency": False},
            },
        },
        {
            "query": "ባንክ ምንድን ነው?",
            "profile": {
                "morphology": {"morphology_complexity_score": 0.35},
                "complexity": {"overall": 0.25, "level": "low"},
                "intent": {"question_type": "definition", "query_category": "general"},
                "entity": {"name": None, "type": None},
                "query_features": {"contains_number": False, "contains_currency": False},
            },
        },
    ]

    print("\n" + "=" * 70)
    print("ROUTING DECISIONS")
    print("=" * 70)

    for test in test_cases:
        print(f"\n📝 Query: {test['query']}")
        
        query_vector = embedding_model.encode_query(test["query"])
        
        result = router.route(
            query_text=test["query"],
            query_vector=query_vector,
            query_profile=test["profile"],
        )
        
        print(f"\n   ✅ Strategy: {result.routing_decision.strategy}")
        print(f"      Weights: dense={result.routing_decision.dense_weight}, sparse={result.routing_decision.sparse_weight}")
        print(f"      Top-K: {result.routing_decision.top_k}")
        print(f"      Rerank: {result.routing_decision.use_reranker}")
        print(f"      Morpho Expansion: {result.routing_decision.apply_morpho_expansion}")
        print(f"      Reason: {result.routing_decision.reasoning[:100]}...")

    print("\n" + "=" * 70)
    print("✅ AdaptiveRouter demo completed successfully")
    print("=" * 70)