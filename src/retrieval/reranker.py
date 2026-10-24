# src/retrieval/reranker.py

"""
Morphology-Aware Reranker for Amharic FinTech RAG.

⭐ SECOND MAJOR METHODOLOGICAL CONTRIBUTION ⭐

This module re-scores retrieval candidates using a multi-dimensional, 
interpretable scoring function that combines:

1. Semantic Similarity (0.45 weight): Cosine similarity from dense retrieval
2. Morphological Overlap (0.25 weight): Shared prefixes, suffixes, roots
3. Entity Overlap (0.15 weight): Presence of detected query entities
4. Domain Keyword Overlap (0.15 weight): FinTech keyword density

Final Score Formula:
    score = 0.45 × semantic + 0.25 × morphology + 0.15 × entity + 0.15 × domain

Design Principles:
- Single Responsibility: Only re-scores, doesn't retrieve
- Interpretable: All component scores logged
- Robust: Handles missing metadata gracefully
- Fast: Optimized for CPU inference (~1ms per chunk)
- Ablatable: Can measure contribution of each component
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.retrieval.dense_retriever import RetrievalResult

logger = logging.getLogger(__name__)


class MorphologyAwareReranker:
    """
    Morphology-aware reranker using multi-dimensional scoring.
    
    This reranker improves retrieval precision by re-scoring candidates
    using semantic, morphological, entity, and domain signals.
    """
    
    # Scoring weights (sum to 1.0)
    SEMANTIC_WEIGHT = 0.45
    MORPHOLOGY_WEIGHT = 0.25
    ENTITY_WEIGHT = 0.15
    DOMAIN_WEIGHT = 0.15
    
    # Morphology patterns (reuse from tokenizer)
    _ETHIOPIC_WORD = r"[\u1200-\u135F\u1369-\u137F]+"
    _TOKEN_PATTERN = re.compile(
        f"{_ETHIOPIC_WORD}|[A-Za-z]+|\\d+(?:[,.]\\d+)*|[.,!?;:()\"'\\-—–።፣፤፥፦፧]"
    )
    _CLITIC_PATTERN = re.compile(
        r"(?<![\u1200-\u135F\u1369-\u137F])(ለ|በ|ከ|የ|ስለ)([\u1200-\u135F\u1369-\u137F]+)"
    )
    
    # Common Amharic prefixes and suffixes for morphological overlap
    PREFIXES = {"መ", "ተ", "አ", "ይ", "አይ", "የ", "ለ", "በ", "ከ", "እ", "ወ", "ስለ"}
    SUFFIXES = {"ች", "ዎች", "ው", "ዋ", "ሁ", "ህ", "ሽ", "ን", "ም", "ነው", "ናቸው", "አለ", "አል"}
    
    # FinTech domain keywords (can be loaded from lexicon file)
    DOMAIN_KEYWORDS = {
        "ባንክ", "ሂሳብ", "ካርድ", "ብድር", "ወለድ", "ክፍያ", "ገንዘብ",
        "atm", "pos", "cbe", "telebirr", "etb", "usd", "ብር",
    }
    
    def __init__(
        self,
        semantic_weight: float = 0.45,
        morphology_weight: float = 0.25,
        entity_weight: float = 0.15,
        domain_weight: float = 0.15,
    ) -> None:
        """
        Initialize the reranker with configurable weights.
        
        Parameters:
            semantic_weight: Weight for semantic similarity (default: 0.45)
            morphology_weight: Weight for morphological overlap (default: 0.25)
            entity_weight: Weight for entity overlap (default: 0.15)
            domain_weight: Weight for domain keyword overlap (default: 0.15)
        """
        # Validate weights sum to 1.0
        weight_sum = semantic_weight + morphology_weight + entity_weight + domain_weight
        if abs(weight_sum - 1.0) > 1e-6:
            logger.warning(f"Weights sum to {weight_sum}, normalizing to 1.0")
            total = weight_sum
            semantic_weight /= total
            morphology_weight /= total
            entity_weight /= total
            domain_weight /= total
        
        self.semantic_weight = semantic_weight
        self.morphology_weight = morphology_weight
        self.entity_weight = entity_weight
        self.domain_weight = domain_weight
        
        logger.info(
            f"✅ MorphologyAwareReranker initialized:\n"
            f"   Weights: semantic={self.semantic_weight:.2f}, "
            f"morphology={self.morphology_weight:.2f}, "
            f"entity={self.entity_weight:.2f}, "
            f"domain={self.domain_weight:.2f}"
        )
    
    def rerank(
        self,
        query_text: str,
        candidates: List[RetrievalResult],
        query_profile: Optional[Dict[str, Any]] = None,
        top_n: int = 5,
    ) -> List[RetrievalResult]:
        """
        Re-score and re-rank retrieval candidates.
        
        Parameters:
            query_text: The query string (possibly expanded)
            candidates: List of RetrievalResult from hybrid retrieval
            query_profile: Optional dict from query_analyzer.py
            top_n: Number of final results to return
            
        Returns:
            List of RetrievalResult with rerank score in `score` field,
            and component scores in `metadata["rerank_scores"]`.
        """
        if top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {top_n}")
        
        if not candidates:
            logger.warning("No candidates to rerank")
            return []
        
        start_time = time.time()
        
        # Extract query features
        query_tokens = self._tokenize(query_text)
        query_entities = self._extract_entities(query_profile)
        
        # Score each candidate
        scored_candidates = []
        for candidate in candidates:
            scores = self._compute_scores(
                query_text=query_text,
                query_tokens=query_tokens,
                query_entities=query_entities,
                candidate=candidate,
            )
            
            # Compute final weighted score
            final_score = (
                self.semantic_weight * scores["semantic"] +
                self.morphology_weight * scores["morphology"] +
                self.entity_weight * scores["entity"] +
                self.domain_weight * scores["domain"]
            )
            
            # Enrich candidate with rerank details (store in metadata)
            enriched_metadata = dict(candidate.metadata)
            enriched_metadata["rerank_scores"] = {
                "semantic": round(scores["semantic"], 4),
                "morphology": round(scores["morphology"], 4),
                "entity": round(scores["entity"], 4),
                "domain": round(scores["domain"], 4),
                "final": round(final_score, 4),
            }
            # Preserve the original fusion score if available
            if "fusion_score" in candidate.metadata:
                enriched_metadata["original_fusion_score"] = candidate.metadata["fusion_score"]
            else:
                enriched_metadata["original_fusion_score"] = candidate.score
            
            scored_candidates.append({
                "candidate": candidate,
                "final_score": final_score,
                "scores": scores,
                "enriched_metadata": enriched_metadata,
            })
        
        # Sort by final score descending
        scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Take top_n and build results
        reranked_results = []
        for rank, item in enumerate(scored_candidates[:top_n], start=1):
            candidate = item["candidate"]
            
            # Create new RetrievalResult with rerank score as the main score
            reranked_results.append(
                RetrievalResult(
                    chunk_id=candidate.chunk_id,
                    score=item["final_score"],  # Use rerank score as main score
                    document=candidate.document,
                    metadata=item["enriched_metadata"],
                    retriever_name="reranked",
                    rank=rank,
                )
            )
        
        total_time = (time.time() - start_time) * 1000
        
        top_score = reranked_results[0].score if reranked_results else 0.0
        
        logger.info(
            f"✅ Reranking completed:\n"
            f"   Candidates: {len(candidates)}\n"
            f"   Returned: {len(reranked_results)}\n"
            f"   Top score: {top_score:.4f}\n"
            f"   Timing: {total_time:.1f}ms"
        )
        
        return reranked_results
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text using the same logic as tokenizer.py."""
        if not text or not text.strip():
            return []
        
        # Separate clitics
        text_with_clitics = self._CLITIC_PATTERN.sub(r"\1 \2", text)
        
        # Extract tokens
        raw_tokens = self._TOKEN_PATTERN.findall(text_with_clitics)
        
        # Lowercase Latin tokens
        return [token.lower() if token.isascii() else token for token in raw_tokens]
    
    def _extract_entities(self, query_profile: Optional[Dict[str, Any]]) -> Set[str]:
        """Extract entity names from query profile."""
        if not query_profile:
            return set()
        
        entities = set()
        entity_info = query_profile.get("entity", {})
        
        if entity_info.get("name"):
            entities.add(entity_info["name"].lower())
        
        return entities
    
    def _compute_scores(
        self,
        query_text: str,
        query_tokens: List[str],
        query_entities: Set[str],
        candidate: RetrievalResult,
    ) -> Dict[str, float]:
        """
        Compute all scoring components for a candidate.
        
        Returns:
            Dict with semantic, morphology, entity, domain scores (all 0-1)
        """
        # 1. Semantic score (from dense/hybrid retrieval)
        semantic_score = min(max(candidate.score, 0.0), 1.0)
        
        # 2. Morphological overlap
        morphology_score = self._compute_morphology_overlap(query_tokens, candidate.document)
        
        # 3. Entity overlap
        entity_score = self._compute_entity_overlap(query_entities, candidate.document)
        
        # 4. Domain keyword overlap
        domain_score = self._compute_domain_overlap(candidate.document)
        
        return {
            "semantic": semantic_score,
            "morphology": morphology_score,
            "entity": entity_score,
            "domain": domain_score,
        }
    
    def _compute_morphology_overlap(
        self,
        query_tokens: List[str],
        document: str,
    ) -> float:
        """
        Compute morphological overlap between query and document.
        
        Measures shared prefixes, suffixes, and token overlap.
        """
        if not query_tokens or not document:
            return 0.0
        
        # Tokenize document
        doc_tokens = self._tokenize(document)
        
        if not doc_tokens:
            return 0.0
        
        # Extract prefixes and suffixes from query
        query_prefixes = set()
        query_suffixes = set()
        for token in query_tokens:
            for prefix in self.PREFIXES:
                if token.startswith(prefix) and len(token) > len(prefix) + 1:
                    query_prefixes.add(prefix)
            for suffix in self.SUFFIXES:
                if token.endswith(suffix) and len(token) > len(suffix) + 1:
                    query_suffixes.add(suffix)
        
        # Extract prefixes and suffixes from document
        doc_prefixes = set()
        doc_suffixes = set()
        for token in doc_tokens:
            for prefix in self.PREFIXES:
                if token.startswith(prefix) and len(token) > len(prefix) + 1:
                    doc_prefixes.add(prefix)
            for suffix in self.SUFFIXES:
                if token.endswith(suffix) and len(token) > len(suffix) + 1:
                    doc_suffixes.add(suffix)
        
        # Compute overlap scores
        prefix_overlap = len(query_prefixes & doc_prefixes) / max(len(query_prefixes), 1)
        suffix_overlap = len(query_suffixes & doc_suffixes) / max(len(query_suffixes), 1)
        
        # Token overlap (Jaccard similarity)
        query_set = set(query_tokens)
        doc_set = set(doc_tokens)
        token_overlap = len(query_set & doc_set) / max(len(query_set | doc_set), 1)
        
        # Combine (prefix and suffix overlap weighted higher)
        morphology_score = (
            0.4 * prefix_overlap +
            0.4 * suffix_overlap +
            0.2 * token_overlap
        )
        
        return min(morphology_score, 1.0)
    
    def _compute_entity_overlap(
        self,
        query_entities: Set[str],
        document: str,
    ) -> float:
        """
        Compute entity overlap between query and document.
        
        Returns 1.0 if all query entities appear in document, 0.0 if none.
        """
        if not query_entities:
            return 0.0
        
        doc_lower = document.lower()
        
        # Count how many query entities appear in document
        matches = sum(1 for entity in query_entities if entity in doc_lower)
        
        # Normalize by number of query entities
        entity_score = matches / len(query_entities)
        
        return min(entity_score, 1.0)
    
    def _compute_domain_overlap(self, document: str) -> float:
        """
        Compute domain keyword density in document.
        
        Returns normalized score based on keyword count.
        """
        if not document:
            return 0.0
        
        doc_lower = document.lower()
        
        # Count domain keywords
        keyword_count = sum(1 for keyword in self.DOMAIN_KEYWORDS if keyword in doc_lower)
        
        # Normalize (assume max 10 keywords is "high density")
        domain_score = min(keyword_count / 10.0, 1.0)
        
        return domain_score


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
    print("MORPHOLOGY-AWARE RERANKER DEMO ⭐ SECOND NOVELTY ⭐")
    print("=" * 70)

    # Paths
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
    CHROMA_DB_PATH = EMBEDDINGS_DIR / "chroma_db"
    CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "chunked_documents.json"
    LOCAL_MODEL_PATH = r"D:\models\multilingual-e5-large"
    INDEX_CACHE_PATH = PROJECT_ROOT / "data" / "retrieval" / "bm25_index.pkl"

    print(f"\n📁 ChromaDB path: {CHROMA_DB_PATH}")
    print(f"📁 Corpus path: {CORPUS_PATH}")
    print(f"📁 Local model: {LOCAL_MODEL_PATH}")

    # Import dependencies
    from src.retrieval.vector_store import VectorStore
    from src.retrieval.dense_retriever import DenseRetriever
    from src.retrieval.sparse_retriever import SparseRetriever
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.embeddings.embedding_model import get_embedding_model

    # 1. Initialize VectorStore (using persist_directory)
    vector_store = VectorStore(
        persist_directory=CHROMA_DB_PATH,
        collection_name="amharic_fintech_corpus",
        metadata_path=EMBEDDINGS_DIR / "embedding_metadata.json",
    )
    print(f"\n✅ VectorStore ready: {vector_store.count()} vectors")

    # 2. Initialize Embedding Model
    embedding_model = get_embedding_model(
        local_model_path=LOCAL_MODEL_PATH,
        use_fp16=False,
        device="cpu",
    )

    # 3. Initialize DenseRetriever
    dense_retriever = DenseRetriever(
        vector_store=vector_store,
        local_model_path=LOCAL_MODEL_PATH,
        device="cpu",
        use_fp16=False,
        top_k=10,
    )

    # 4. Initialize SparseRetriever (using chunks_path)
    sparse_retriever = SparseRetriever(
        chunks_path=CORPUS_PATH,
        index_cache_path=INDEX_CACHE_PATH,
    )

    # 5. Initialize HybridRetriever
    hybrid_retriever = HybridRetriever(rrf_k=60)

    # 6. Initialize Reranker
    print("\n" + "=" * 70)
    print("INITIALIZING RERANKER")
    print("=" * 70)
    reranker = MorphologyAwareReranker()

    # Test cases
    test_cases = [
        {
            "query": "የCBE ካርድ እንዴት ማግኘት እችላለሁ?",
            "profile": {"entity": {"name": "Commercial Bank of Ethiopia", "type": "Bank"}},
            "description": "Entity-heavy query (should boost entity score)",
        },
        {
            "query": "100,000 ብር ገደብ",
            "profile": {"entity": {"name": None, "type": None}},
            "description": "Numerical query (should boost domain score)",
        },
        {
            "query": "ይከፈላልን?",
            "profile": {"entity": {"name": None, "type": None}},
            "description": "Highly inflected verb (should boost morphology score)",
        },
    ]

    print("\n" + "=" * 70)
    print("RETRIEVAL AND RERANKING")
    print("=" * 70)

    for i, test in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test['description']}")
        print(f"   Query: {test['query']}")

        # Get candidates from hybrid retrieval
        query_vector = embedding_model.encode_query(test["query"])

        dense_results = dense_retriever.retrieve_with_vector(query_vector=query_vector, top_k=10)
        sparse_results = sparse_retriever.retrieve(query_text=test["query"], top_k=10)

        hybrid_results = hybrid_retriever.fuse(
            dense_results=dense_results,
            sparse_results=sparse_results,
            dense_weight=0.5,
            sparse_weight=0.5,
            top_k=10,
        )

        print(f"\n   Before Reranking (Top 3):")
        for r in hybrid_results[:3]:
            print(f"      Rank {r.rank}: {r.chunk_id} (score={r.score:.4f})")

        # Rerank
        reranked = reranker.rerank(
            query_text=test["query"],
            candidates=hybrid_results,
            query_profile=test["profile"],
            top_n=5,
        )

        print(f"\n   After Reranking (Top 3):")
        for r in reranked[:3]:
            scores = r.metadata.get("rerank_scores", {})
            print(f"      Rank {r.rank}: {r.chunk_id}")
            print(f"         Final: {scores.get('final', 0):.4f}")
            print(f"         Semantic: {scores.get('semantic', 0):.4f}")
            print(f"         Morphology: {scores.get('morphology', 0):.4f}")
            print(f"         Entity: {scores.get('entity', 0):.4f}")
            print(f"         Domain: {scores.get('domain', 0):.4f}")

    print("\n" + "=" * 70)
    print("✅ Reranker demo completed successfully")
    print("=" * 70)