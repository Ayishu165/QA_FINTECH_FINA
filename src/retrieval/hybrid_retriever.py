# src/retrieval/hybrid_retriever.py

"""
Hybrid Retriever (RRF Fusion) for Amharic FinTech RAG.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.retrieval.dense_retriever import DenseRetriever, RetrievalResult
from src.retrieval.sparse_retriever import SparseRetriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retriever using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, rrf_k: int = 60) -> None:
        if rrf_k < 1:
            raise ValueError(f"rrf_k must be >= 1, got {rrf_k}")
        self.rrf_k = rrf_k
        logger.info(f"✅ HybridRetriever initialized with RRF k={rrf_k}")

    # ==========================================================
    # CORE FUSION METHOD
    # ==========================================================

    def fuse(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
        dense_weight: float = 0.60,
        sparse_weight: float = 0.40,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """
        Fuse dense and sparse retrieval results using RRF.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if dense_weight < 0 or sparse_weight < 0:
            raise ValueError(f"Weights must be non-negative: dense={dense_weight}, sparse={sparse_weight}")

        # Normalize weights
        weight_sum = dense_weight + sparse_weight
        if weight_sum > 0 and abs(weight_sum - 1.0) > 1e-6:
            dense_weight /= weight_sum
            sparse_weight /= weight_sum

        start_time = time.time()

        if not dense_results and not sparse_results:
            logger.warning("Both retrievers returned empty results")
            return []

        if not dense_results:
            logger.warning("Dense retriever returned empty, using only sparse results")
            return self._single_retriever_fallback(sparse_results, "sparse", top_k)

        if not sparse_results:
            logger.warning("Sparse retriever returned empty, using only dense results")
            return self._single_retriever_fallback(dense_results, "dense", top_k)

        dense_rank_map = {r.chunk_id: (r.rank, r) for r in dense_results}
        sparse_rank_map = {r.chunk_id: (r.rank, r) for r in sparse_results}
        all_chunk_ids = set(dense_rank_map.keys()) | set(sparse_rank_map.keys())

        rrf_scores = {}
        fusion_details = {}

        for chunk_id in all_chunk_ids:
            score = 0.0
            dense_contrib = 0.0
            sparse_contrib = 0.0
            dense_rank = None
            sparse_rank = None

            if chunk_id in dense_rank_map:
                rank, _ = dense_rank_map[chunk_id]
                dense_contrib = dense_weight / (self.rrf_k + rank)
                score += dense_contrib
                dense_rank = rank

            if chunk_id in sparse_rank_map:
                rank, _ = sparse_rank_map[chunk_id]
                sparse_contrib = sparse_weight / (self.rrf_k + rank)
                score += sparse_contrib
                sparse_rank = rank

            rrf_scores[chunk_id] = score
            fusion_details[chunk_id] = {
                "dense_rank": dense_rank,
                "sparse_rank": sparse_rank,
                "dense_contribution": round(dense_contrib, 6),
                "sparse_contribution": round(sparse_contrib, 6),
                "appeared_in_both": dense_rank is not None and sparse_rank is not None,
            }

        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        top_chunk_ids = sorted_chunk_ids[:top_k]

        fused_results = []
        for rank, chunk_id in enumerate(top_chunk_ids, start=1):
            fusion_score = rrf_scores[chunk_id]
            details = fusion_details[chunk_id]

            if chunk_id in dense_rank_map:
                base_result = dense_rank_map[chunk_id][1]
            else:
                base_result = sparse_rank_map[chunk_id][1]

            enriched_metadata = dict(base_result.metadata)
            enriched_metadata["fusion_details"] = details
            enriched_metadata["dense_weight"] = round(dense_weight, 4)
            enriched_metadata["sparse_weight"] = round(sparse_weight, 4)
            enriched_metadata["rrf_k"] = self.rrf_k
            enriched_metadata["fusion_method"] = "reciprocal_rank_fusion"
            enriched_metadata["fusion_score"] = fusion_score

            fused_results.append(
                RetrievalResult(
                    chunk_id=base_result.chunk_id,
                    score=fusion_score,
                    document=base_result.document,
                    metadata=enriched_metadata,
                    retriever_name="hybrid",
                    rank=rank,
                )
            )

        total_time = (time.time() - start_time) * 1000
        both_count = sum(1 for d in fusion_details.values() if d["appeared_in_both"])

        top_score = fused_results[0].score if fused_results else 0.0

        logger.info(
            f"✅ Hybrid fusion completed:\n"
            f"   Dense results: {len(dense_results)}\n"
            f"   Sparse results: {len(sparse_results)}\n"
            f"   Unique chunks: {len(all_chunk_ids)}\n"
            f"   Appeared in both: {both_count}\n"
            f"   Fused results: {len(fused_results)}\n"
            f"   Weights: dense={dense_weight:.2f}, sparse={sparse_weight:.2f}\n"
            f"   Top score: {top_score:.6f}\n"
            f"   Timing: {total_time:.1f}ms"
        )

        return fused_results

    # ==========================================================
    # SINGLE RETRIEVER FALLBACK
    # ==========================================================

    def _single_retriever_fallback(
        self,
        results: List[RetrievalResult],
        retriever_name: str,
        top_k: int,
    ) -> List[RetrievalResult]:
        top_results = results[:top_k]
        hybrid_results = []
        for rank, result in enumerate(top_results, start=1):
            fusion_score = 1.0 / (self.rrf_k + result.rank)

            if retriever_name == "dense":
                details = {
                    "dense_rank": result.rank,
                    "sparse_rank": None,
                    "dense_contribution": round(fusion_score, 6),
                    "sparse_contribution": 0.0,
                    "appeared_in_both": False,
                }
            else:
                details = {
                    "dense_rank": None,
                    "sparse_rank": result.rank,
                    "dense_contribution": 0.0,
                    "sparse_contribution": round(fusion_score, 6),
                    "appeared_in_both": False,
                }

            enriched_metadata = dict(result.metadata)
            enriched_metadata["fusion_details"] = details
            enriched_metadata["single_retriever_fallback"] = retriever_name
            enriched_metadata["rrf_k"] = self.rrf_k
            enriched_metadata["fusion_method"] = "reciprocal_rank_fusion"
            enriched_metadata["fusion_score"] = fusion_score

            hybrid_results.append(
                RetrievalResult(
                    chunk_id=result.chunk_id,
                    score=fusion_score,
                    document=result.document,
                    metadata=enriched_metadata,
                    retriever_name="hybrid",
                    rank=rank,
                )
            )

        return hybrid_results

    # ==========================================================
    # RETRIEVE AND FUSE
    # ==========================================================

    def retrieve_and_fuse(
        self,
        query_text: str,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        query_vector: Optional[np.ndarray] = None,
        dense_weight: float = 0.60,
        sparse_weight: float = 0.40,
        top_k: int = 5,
        overfetch_factor: int = 2,
    ) -> List[RetrievalResult]:
        fetch_k = top_k * overfetch_factor
        logger.info(f"🔍 Retrieving with overfetch: fetch_k={fetch_k}, final top_k={top_k}, "
                    f"weights: dense={dense_weight:.2f}, sparse={sparse_weight:.2f}")

        if query_vector is not None:
            dense_results = dense_retriever.retrieve_with_vector(
                query_vector=query_vector,
                top_k=fetch_k,
            )
        else:
            dense_results = dense_retriever.retrieve(
                query_text=query_text,
                top_k=fetch_k,
            )

        # Sparse retriever does NOT accept include_metadata
        sparse_results = sparse_retriever.retrieve(
            query_text=query_text,
            top_k=fetch_k,
        )

        return self.fuse(
            dense_results=dense_results,
            sparse_results=sparse_results,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            top_k=top_k,
        )


# ==========================================================
# DEMO
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
    print("HYBRID RETRIEVER (RRF FUSION) DEMO")
    print("=" * 70)

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
    CHROMA_DB_PATH = EMBEDDINGS_DIR / "chroma_db"
    CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "chunked_documents.json"
    LOCAL_MODEL_PATH = r"D:\models\multilingual-e5-large"

    print(f"\n📁 ChromaDB path: {CHROMA_DB_PATH}")
    print(f"📁 Corpus path: {CORPUS_PATH}")
    print(f"📁 Local model: {LOCAL_MODEL_PATH}")

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
    print(f"\n✅ HybridRetriever initialized with RRF k={hybrid_retriever.rrf_k}")

    test_cases = [
        {
            "query": "የCBE ካርድ እንዴት ማግኘት እችላለሁ?",
            "dense_weight": 0.65,
            "sparse_weight": 0.35,
            "description": "Semantic-heavy",
        },
        {
            "query": "100,000 ብር ገደብ",
            "dense_weight": 0.35,
            "sparse_weight": 0.65,
            "description": "Lexical-heavy",
        },
        {
            "query": "CBE Visa Platinum ካርድ ገደብ",
            "dense_weight": 0.50,
            "sparse_weight": 0.50,
            "description": "Balanced",
        },
    ]

    print("\n" + "=" * 70)
    print("HYBRID RETRIEVAL RESULTS (RRF Fusion)")
    print("=" * 70)

    for i, test in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test['description']}")
        print(f"   Query: {test['query']}")
        print(f"   Weights: dense={test['dense_weight']}, sparse={test['sparse_weight']}")

        query_vector = embedding_model.encode_query(test["query"])
        results = hybrid_retriever.retrieve_and_fuse(
            query_text=test["query"],
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            query_vector=query_vector,
            dense_weight=test["dense_weight"],
            sparse_weight=test["sparse_weight"],
            top_k=3,
            overfetch_factor=2,
        )

        print(f"\n   Fused {len(results)} results:\n")
        for result in results:
            details = result.metadata.get("fusion_details", {})
            fusion_score = result.metadata.get("fusion_score", result.score)
            print(f"   Rank {result.rank}:")
            print(f"      Chunk ID: {result.chunk_id}")
            print(f"      Fusion Score: {fusion_score:.6f}")
            print(f"      Dense Rank: {details.get('dense_rank', 'N/A')}")
            print(f"      Sparse Rank: {details.get('sparse_rank', 'N/A')}")
            print(f"      Dense Contrib: {details.get('dense_contribution', 0):.6f}")
            print(f"      Sparse Contrib: {details.get('sparse_contribution', 0):.6f}")
            print(f"      In Both: {details.get('appeared_in_both', False)}")
            print(f"      Preview: {result.document[:100]}...")
            print()

    print("\n" + "=" * 70)
    print("✅ HybridRetriever demo completed successfully")
    print("=" * 70)