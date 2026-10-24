# src/retrieval/dense_retriever.py

"""
Dense Retriever for Amharic FinTech RAG.

This module performs pure semantic retrieval using dense vector embeddings
from multilingual-e5-large. It serves as:
1. A baseline retriever for ablation studies (dense-only vs hybrid)
2. The semantic component in hybrid retrieval (RRF fusion)
3. The primary retriever for LOW morphological complexity queries

Design Principles:
- Single Responsibility: Only performs semantic search, no fusion logic
- Standardized Output: Returns RetrievalResult for consistency across retrievers
- Timing Transparency: Measures encoding, search, and total time for evaluation
- Error Clarity: All failures raise actionable exceptions
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.embeddings.embedding_model import get_embedding_model
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


# ==========================================================
# Standardized Result Dataclass
# ==========================================================

@dataclass
class RetrievalResult:
    """
    Standardized retrieval result from any retriever.

    Attributes:
        chunk_id: Unique identifier for the chunk.
        score: Similarity/relevance score (higher is better).
        document: The chunk text.
        metadata: Dictionary of metadata for the chunk.
        retriever_name: Name of the retriever that produced this result.
        rank: Position in the ranked list.
    """
    chunk_id: str
    score: float
    document: str
    metadata: Dict[str, Any]
    retriever_name: str
    rank: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "score": self.score,
            "document": self.document,
            "metadata": self.metadata,
            "retriever_name": self.retriever_name,
            "rank": self.rank,
        }


class DenseRetriever:
    """
    Semantic retriever using dense vector embeddings.

    This retriever encodes queries using multilingual-e5-large and performs
    cosine similarity search against the vector store. It excels at capturing
    conceptual similarity but may miss exact keyword matches.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model_name: str = "intfloat/multilingual-e5-large",
        local_model_path: Optional[str] = None,
        device: Optional[str] = None,
        use_fp16: bool = False,
        use_mirror: bool = False,
        top_k: int = 10,
    ):
        """
        Initialize the DenseRetriever.

        Parameters:
            vector_store: Initialized VectorStore instance.
            embedding_model_name: Name of the embedding model.
            local_model_path: Local model directory path.
            device: 'cuda', 'cpu', or None (auto-detect).
            use_fp16: Enable FP16 for embedding model.
            use_mirror: Use HF mirror for model loading.
            top_k: Default number of results to return.

        Raises:
            TypeError: If vector_store is not a VectorStore instance.
        """
        if not isinstance(vector_store, VectorStore):
            raise TypeError(
                f"vector_store must be a VectorStore instance, got {type(vector_store)}"
            )

        self.vector_store = vector_store
        self.top_k = top_k
        self.device = device
        self.use_fp16 = use_fp16
        self.use_mirror = use_mirror

        # Verify vector store has vectors
        if self.vector_store.count() == 0:
            logger.warning("⚠️ VectorStore is empty. Dense retrieval will return no results.")

        # Load embedding model (singleton)
        logger.info("Loading embedding model for dense retrieval...")
        self.embedding_model = get_embedding_model(
            model_name=embedding_model_name,
            local_model_path=local_model_path,
            device=device,
            use_fp16=use_fp16,
            use_mirror=use_mirror,
        )

        self.embedding_dim = self.embedding_model.get_embedding_dim()
        self.model_name = self.embedding_model.get_model_name()

        logger.info(
            f"✅ DenseRetriever initialized:\n"
            f"   Model: {self.model_name}\n"
            f"   Dimension: {self.embedding_dim}\n"
            f"   Device: {self.embedding_model.get_device()}\n"
            f"   Top-K: {self.top_k}\n"
            f"   Vectors available: {self.vector_store.count()}"
        )

    # ==========================================================
    # Core Retrieval Methods
    # ==========================================================

    def retrieve(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
    ) -> List[RetrievalResult]:
        """
        Retrieve semantically similar chunks for the given query.

        This method:
        1. Encodes the query using the embedding model (with "query: " prefix)
        2. Searches the vector store for top-k nearest neighbors
        3. Returns standardized RetrievalResult objects

        Parameters:
            query_text: The user's query string.
            top_k: Number of results to return (defaults to self.top_k).
            where: Optional ChromaDB where clause for metadata filtering.
            include_metadata: Include full metadata in results.

        Returns:
            List of RetrievalResult objects, sorted by cosine similarity descending.

        Raises:
            ValueError: If query_text is empty or top_k < 1.
            RuntimeError: If encoding or search fails.
        """
        # Validate inputs
        if not query_text or not query_text.strip():
            raise ValueError("query_text cannot be empty")

        top_k = top_k or self.top_k
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        # Start timing
        start_time = time.time()

        # Step 1: Encode query with E5 query prefix
        encode_start = time.time()
        try:
            query_embedding = self.embedding_model.encode_query(query_text)
        except Exception as e:
            raise RuntimeError(f"Failed to encode query: {e}") from e
        encode_time = time.time() - encode_start

        logger.debug(f"Query encoded in {encode_time * 1000:.2f}ms")

        # Step 2: Search vector store
        search_start = time.time()
        try:
            search_results = self.vector_store.search(
                query_vector=query_embedding.tolist(),
                top_k=top_k,
                where=where,
                include_metadata=include_metadata,
            )
        except Exception as e:
            raise RuntimeError(f"Vector search failed: {e}") from e
        search_time = time.time() - search_start

        logger.debug(f"Vector search completed in {search_time * 1000:.2f}ms")

        # Step 3: Convert to RetrievalResult
        retrieval_results = []
        for rank, result in enumerate(search_results, start=1):
            retrieval_results.append(
                RetrievalResult(
                    chunk_id=result.chunk_id,
                    score=result.score,
                    document=result.document,
                    metadata=result.metadata,
                    retriever_name="dense",
                    rank=rank,
                )
            )

        total_time = time.time() - start_time

        # Fixed: Conditional formatting properly handled
        top_score = retrieval_results[0].score if retrieval_results else 0.0

        logger.info(
            f"✅ Dense retrieval completed:\n"
            f"   Query: {query_text[:50]}{'...' if len(query_text) > 50 else ''}\n"
            f"   Results: {len(retrieval_results)}\n"
            f"   Top score: {top_score:.4f}\n"
            f"   Timing: encode={encode_time * 1000:.1f}ms, search={search_time * 1000:.1f}ms, total={total_time * 1000:.1f}ms"
        )

        return retrieval_results

    # ==========================================================
    # Pre-encoded Query Retrieval
    # ==========================================================

    def retrieve_with_vector(
        self,
        query_vector: np.ndarray,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
    ) -> List[RetrievalResult]:
        """
        Retrieve using a pre-computed query vector (skip encoding step).

        This is useful when the query has already been encoded by the pipeline
        (e.g., in retrieval_pipeline.py), avoiding redundant encoding.

        Parameters:
            query_vector: Pre-computed query embedding (must be 1024-dim, L2-normalized).
            top_k: Number of results to return.
            where: Optional ChromaDB where clause.
            include_metadata: Include full metadata in results.

        Returns:
            List of RetrievalResult objects.

        Raises:
            ValueError: If top_k < 1.
            RuntimeError: If search fails.
        """
        top_k = top_k or self.top_k
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        start_time = time.time()

        # Search vector store
        search_start = time.time()
        try:
            search_results = self.vector_store.search(
                query_vector=query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector,
                top_k=top_k,
                where=where,
                include_metadata=include_metadata,
            )
        except Exception as e:
            raise RuntimeError(f"Vector search failed: {e}") from e
        search_time = time.time() - search_start

        # Convert to RetrievalResult
        retrieval_results = []
        for rank, result in enumerate(search_results, start=1):
            retrieval_results.append(
                RetrievalResult(
                    chunk_id=result.chunk_id,
                    score=result.score,
                    document=result.document,
                    metadata=result.metadata,
                    retriever_name="dense",
                    rank=rank,
                )
            )

        total_time = time.time() - start_time

        top_score = retrieval_results[0].score if retrieval_results else 0.0

        logger.info(
            f"✅ Dense retrieval (pre-encoded) completed:\n"
            f"   Results: {len(retrieval_results)}\n"
            f"   Top score: {top_score:.4f}\n"
            f"   Timing: search={search_time * 1000:.1f}ms, total={total_time * 1000:.1f}ms"
        )

        return retrieval_results

    # ==========================================================
    # Batch Retrieval Methods
    # ==========================================================

    def retrieve_batch(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        show_progress: bool = False,
    ) -> List[List[RetrievalResult]]:
        """
        Retrieve chunks for multiple queries.

        Parameters:
            queries: List of query strings.
            top_k: Number of results per query.
            where: Optional ChromaDB where clause.
            include_metadata: Include full metadata in results.
            show_progress: Show progress bar.

        Returns:
            List of retrieval results, one per query.
        """
        top_k = top_k or self.top_k
        results = []

        from tqdm import tqdm

        iterator = tqdm(queries, desc="Retrieving", disable=not show_progress)
        for query in iterator:
            result = self.retrieve(
                query_text=query,
                top_k=top_k,
                where=where,
                include_metadata=include_metadata,
            )
            results.append(result)

        return results

    # ==========================================================
    # Utility Methods
    # ==========================================================

    def get_query_embedding(self, query_text: str) -> np.ndarray:
        """
        Get the embedding for a query (with E5 query prefix).

        Parameters:
            query_text: Query string.

        Returns:
            Query embedding as numpy array.
        """
        return self.embedding_model.encode_query(query_text)

    def batch_get_query_embeddings(
        self,
        queries: List[str],
        show_progress: bool = False,
    ) -> List[np.ndarray]:
        """Get embeddings for multiple queries."""
        from tqdm import tqdm

        embeddings = []
        iterator = tqdm(queries, desc="Encoding queries", disable=not show_progress)
        for query in iterator:
            embeddings.append(self.get_query_embedding(query))
        return embeddings

    def get_model_name(self) -> str:
        """Get the model name."""
        return self.model_name

    def get_embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self.embedding_dim

    def get_device(self) -> str:
        """Get the device the model is on."""
        return self.embedding_model.get_device()

    def set_top_k(self, top_k: int) -> None:
        """Update the default top_k value."""
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        self.top_k = top_k
        logger.info(f"Top-K updated to {top_k}")

    def vector_store_stats(self) -> Dict[str, Any]:
        """Get statistics from the underlying vector store."""
        return self.vector_store.statistics()


# ==========================================================
# DEMO / TEST
# ==========================================================

if __name__ == "__main__":
    import json
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    print("=" * 70)
    print("DENSE RETRIEVER DEMO")
    print("=" * 70)

    # Paths
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
    CHROMA_DB_PATH = EMBEDDINGS_DIR / "chroma_db"
    LOCAL_MODEL_PATH = r"D:\models\multilingual-e5-large"

    print(f"\n📁 ChromaDB path: {CHROMA_DB_PATH}")
    print(f"📁 Local model: {LOCAL_MODEL_PATH}")

    # 1. Initialize VectorStore
    vector_store = VectorStore(
        persist_directory=CHROMA_DB_PATH,
        collection_name="amharic_fintech_corpus",
        metadata_path=EMBEDDINGS_DIR / "embedding_metadata.json",
    )

    print(f"\n✅ VectorStore ready: {vector_store.count()} vectors")

    # 2. Initialize DenseRetriever
    retriever = DenseRetriever(
        vector_store=vector_store,
        local_model_path=LOCAL_MODEL_PATH,
        device="cpu",
        use_fp16=False,
        top_k=5,
    )

    # 3. Test queries (UPDATED: Added first query)
    test_queries = [
        "CBE birr ምንድን ነው?",
        "የCBE ካርድ እንዴት ማግኘት እችላለሁ?",
        "የብድር ወለድ መጠን ስንት ነው?",
        "የባንክ ሂሳብ ምንድን ነው?",
    ]

    print("\n" + "=" * 70)
    print("RETRIEVAL RESULTS")
    print("=" * 70)

    for query in test_queries:
        print(f"\n📝 Query: {query}")
        try:
            results = retriever.retrieve(query, top_k=3)

            print(f"   Total results: {len(results)}")
            for result in results:
                print(f"\n   Rank {result.rank}:")
                print(f"      Chunk ID: {result.chunk_id}")
                print(f"      Score: {result.score:.4f}")
                print(f"      Preview: {result.document[:120]}...")
                print(f"      Metadata: token_count={result.metadata.get('token_count', 'N/A')}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    print("\n" + "=" * 70)
    print("✅ DenseRetriever demo completed successfully")
    print("=" * 70)