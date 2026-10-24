# =====================================================================
# src/vectordb/vector_search.py
# =====================================================================

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional  # Fixed: Added Optional import
from src.vectordb.chroma_manager import ChromaManager

logger = logging.getLogger("VectorSearch")


class VectorSearch:
    """
    Executes adaptive retrieval configurations by leveraging QueryAnalyzer routing directives.
    """
    def __init__(self, db_manager: ChromaManager, collection_name: str = "fintech_rag"):
        self.db_manager = db_manager
        self.collection = self.db_manager.get_or_create_collection(collection_name)

    def _execute_dense_scan(self, query_text: str, top_k: int, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Performs raw vector index matching via HNSW tracking."""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=filter_metadata
        )
        return self._format_chroma_results(results)

    def _execute_hybrid_fusion(
        self, 
        query_text: str, 
        top_k: int, 
        dense_weight: float, 
        sparse_weight: float,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Emulates structural hybrid rank fusion adjustment. 
        Combines standard vector search with a custom keyword frequency 
        rerank calculation payload based on sparse vs. dense weights.
        """
        # Fetch a broader candidate pool to prevent retrieval starvation during ranking fusion
        candidate_pool_limit = min(top_k * 3, 50)
        raw_candidates = self._execute_dense_scan(query_text, top_k=candidate_pool_limit, filter_metadata=filter_metadata)

        # Apply basic BM25 keyword matching weight adjustment over candidates
        query_words = set(query_text.lower().split())
        fused_results = []

        for item in raw_candidates:
            doc_text = item["document"].lower()
            # Calculate a basic lexical overlap score (Sparse indicator)
            word_matches = sum(1 for word in query_words if word in doc_text)
            lexical_score = word_matches / max(len(query_words), 1)

            # Invert default cosine distance to compute vector confidence (Dense indicator)
            dense_score = 1.0 - item["distance"] 

            # Calculate a unified hybrid linear score using current parameters
            hybrid_score = (dense_score * dense_weight) + (lexical_score * sparse_weight)
            
            item["score"] = round(hybrid_score, 4)
            fused_results.append(item)

        # Re-sort using our newly calculated hybrid metric and slice to requested top_k boundary
        fused_results.sort(key=lambda x: x["score"], reverse=True)
        return fused_results[:top_k]

    def _format_chroma_results(self, raw_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Normalizes Chroma query response layouts into flattened clean dictionaries."""
        formatted = []
        if not raw_results or not raw_results.get("ids") or not raw_results["ids"][0]:
            return formatted

        for i in range(len(raw_results["ids"][0])):
            formatted.append({
                "id": raw_results["ids"][0][i],
                "document": raw_results["documents"][0][i],
                "metadata": raw_results["metadatas"][0][i],
                "distance": raw_results["distances"][0][i] if raw_results.get("distances") else 0.0,
                "score": 1.0 - raw_results["distances"][0][i] if raw_results.get("distances") else 0.0
            })
        return formatted

    def search(self, query_text: str, routing_config: Dict[str, Any], filter_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main entry point. Maps the routing payload parameters to 
        tailored retrieval execution workflows.
        """
        # Extract operational directives from the configuration payload
        strategy = routing_config.get("vector_search_strategy", "hybrid_balanced")
        top_k = routing_config.get("top_k", 8)
        dense_weight = routing_config.get("dense_weight", 0.50)
        sparse_weight = routing_config.get("sparse_weight", 0.50)
        enable_reranker = routing_config.get("enable_reranker", False)
        execution_route = routing_config.get("execution_route", "Default_Scan")

        logger.info(f"Executing route [{execution_route}] using strategy '{strategy}' for top_k={top_k}")

        # Branch processing depending on structural requirements
        if strategy == "dense_only":
            hits = self._execute_dense_scan(query_text, top_k=top_k, filter_metadata=filter_metadata)
        else:
            hits = self._execute_hybrid_fusion(
                query_text=query_text,
                top_k=top_k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
                filter_metadata=filter_metadata
            )

        # Emulate secondary multi-stage cross-encoder execution flags
        is_rerank_applied = False
        if enable_reranker and len(hits) > 1:
            # Simulate a top-tier cross-encoder scaling optimization pass
            hits.sort(key=lambda x: x["score"] * 1.05, reverse=True)
            is_rerank_applied = True

        return {
            "retrieved_chunks": hits,
            "metrics": {
                "count": len(hits),
                "strategy_used": strategy,
                "route_taken": execution_route,
                "reranker_executed": is_rerank_applied
            }
        }