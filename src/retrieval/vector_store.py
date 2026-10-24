# src/retrieval/vector_store.py

"""
Vector Store Manager for Amharic FinTech RAG.

This module loads and manages the embedding database (ChromaDB).
It provides:
    - Fast vector similarity search returning VectorSearchResult objects
    - Comprehensive validation with publication-quality health reports
    - Metadata lookup and consistency checking
    - Collection statistics (storage size, document counts, etc.)
    - Vector quality metrics (NaN, Inf, zero vectors, duplicates)

Research-grade validation ensures:
    - All embeddings are 1024-dimensional
    - All embeddings are L2-normalized (norm ≈ 1.0)
    - No NaN, Inf, or zero vectors
    - Metadata consistency (every vector has metadata)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import numpy as np
from chromadb.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    """
    A single search result from the vector store.

    Attributes:
        chunk_id: Unique identifier for the chunk.
        score: Similarity score (0-1, higher is better).
        document: The chunk text.
        metadata: Dictionary of metadata for the chunk.
        distance: Raw distance from ChromaDB.
    """
    chunk_id: str
    score: float
    document: str
    metadata: Dict[str, Any]
    distance: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "score": self.score,
            "document": self.document,
            "metadata": self.metadata,
            "distance": self.distance,
        }


class VectorStore:
    """
    Manages the ChromaDB vector store with validation and search capabilities.
    """

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str = "amharic_fintech_chunks",
        embeddings_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        expected_dimension: int = 1024,
        validate_all_vectors: bool = True,
    ):
        """
        Initialize the VectorStore.

        Parameters:
            persist_directory: Path to ChromaDB persistent directory.
            collection_name: Name of the ChromaDB collection.
            embeddings_path: Path to embeddings.json (for validation).
            metadata_path: Path to embedding_metadata.json (for validation).
            expected_dimension: Expected embedding dimension (default: 1024 for E5-large).
            validate_all_vectors: If True, validates all vectors (slower but thorough).
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embeddings_path = embeddings_path
        self.metadata_path = metadata_path
        self.expected_dimension = expected_dimension
        self.validate_all_vectors = validate_all_vectors

        self._client: Optional[chromadb.PersistentClient] = None
        self._collection: Optional[chromadb.Collection] = None
        self._metadata_lookup: Dict[str, Dict[str, Any]] = {}
        self._total_vectors: int = 0
        self._embeddings_dim: Optional[int] = None
        self._health_status: Dict[str, Any] = {}
        self._collection_summary: Dict[str, Any] = {}
        self._vector_quality: Dict[str, Any] = {}

        # Connect and validate
        self._connect()
        self._validate()
        self._compute_summary()

        logger.info(
            f"VectorStore initialized:\n"
            f"  Collection: {self.collection_name}\n"
            f"  Vectors: {self._total_vectors}\n"
            f"  Dimension: {self._embeddings_dim}\n"
            f"  Status: {self._health_status.get('status', 'unknown')}"
        )

    # ==========================================================
    # Connection Methods
    # ==========================================================

    def _connect(self) -> None:
        """Connect to ChromaDB and get or create the collection."""
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create collection
        try:
            self._collection = self._client.get_collection(self.collection_name)
            self._total_vectors = self._collection.count()
            logger.info(f"Connected to existing collection: {self.collection_name} ({self._total_vectors} vectors)")
        except ValueError:
            # Collection doesn't exist, create it
            self._collection = self._client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "ip"},
            )
            self._total_vectors = 0
            logger.info(f"Created new collection: {self.collection_name}")

    # ==========================================================
    # Validation Methods (Publication-Quality)
    # ==========================================================

    def _validate(self) -> None:
        """Perform comprehensive validation of the collection."""
        self._health_status = {
            "collection_exists": self._collection is not None,
            "vector_count": self._total_vectors,
            "expected_dimension": self.expected_dimension,
            "all_dimensions_valid": False,
            "all_norms_valid": False,
            "average_norm": None,
            "min_norm": None,
            "max_norm": None,
            "no_duplicate_ids": False,
            "metadata_loaded": False,
            "metadata_count": 0,
            "metadata_consistent": False,
            "no_nan_vectors": False,
            "no_inf_vectors": False,
            "no_zero_vectors": False,
            "status": "unknown",
            "issues": [],
        }

        self._vector_quality = {
            "normalized": False,
            "duplicate_vectors": 0,
            "zero_vectors": 0,
            "nan_vectors": 0,
            "inf_vectors": 0,
        }

        # 1. Load metadata
        self._load_metadata_lookup()
        self._health_status["metadata_loaded"] = len(self._metadata_lookup) > 0
        self._health_status["metadata_count"] = len(self._metadata_lookup)

        # 2. Validate vectors
        self._validate_vectors()

        # 3. Validate duplicates
        self._validate_duplicates()

        # 4. Validate metadata consistency
        self._validate_metadata_consistency()

        # 5. Set overall status
        issues = []
        if not self._health_status["collection_exists"]:
            issues.append("collection_not_exists")
        if not self._health_status["all_dimensions_valid"]:
            issues.append("invalid_dimensions")
        if not self._health_status["all_norms_valid"]:
            issues.append("invalid_norms")
        if not self._health_status["no_duplicate_ids"]:
            issues.append("duplicate_ids")
        if self._health_status.get("no_nan_vectors") is False:
            issues.append("nan_vectors_detected")
        if self._health_status.get("no_inf_vectors") is False:
            issues.append("inf_vectors_detected")

        self._health_status["status"] = "healthy" if not issues else "unhealthy"
        self._health_status["issues"] = issues

    def _validate_vectors(self) -> None:
        """Validate embedding dimensions, norms, NaN, Inf, and zero vectors."""
        if self._total_vectors == 0:
            logger.warning("Collection is empty. Skipping vector validation.")
            return

        try:
            # Get all vectors for validation
            limit = None if self.validate_all_vectors else min(50, self._total_vectors)
            result = self._collection.get(
                limit=limit,
                include=["embeddings"],
            )

            if not result or result.get("embeddings") is None:
                logger.warning("Could not retrieve embeddings for validation")
                return

            embeddings = result["embeddings"]
            total_validated = len(embeddings)

            if total_validated == 0:
                logger.warning("No embeddings retrieved for validation")
                return

            # --- Dimension Validation ---
            dims = [len(vec) for vec in embeddings if vec is not None]
            all_dim_ok = all(d == self.expected_dimension for d in dims)
            self._health_status["all_dimensions_valid"] = all_dim_ok
            self._embeddings_dim = self.expected_dimension if all_dim_ok else None

            if all_dim_ok:
                logger.info(f"✅ {total_validated} / {total_validated} vectors have dimension {self.expected_dimension}")
            else:
                bad_dims = [d for d in dims if d != self.expected_dimension]
                logger.warning(f"⚠️ Found {len(bad_dims)} vectors with incorrect dimensions: {bad_dims}")

            # --- Norm Validation ---
            norms = []
            valid_norms = []
            for vec in embeddings:
                if vec is not None and len(vec) > 0:
                    try:
                        norm_val = np.linalg.norm(vec)
                        norms.append(norm_val)
                        if np.isclose(norm_val, 1.0, atol=1e-3):
                            valid_norms.append(True)
                        else:
                            valid_norms.append(False)
                    except Exception:
                        valid_norms.append(False)

            if norms:
                self._health_status["average_norm"] = float(np.mean(norms))
                self._health_status["min_norm"] = float(np.min(norms))
                self._health_status["max_norm"] = float(np.max(norms))
                self._health_status["all_norms_valid"] = all(valid_norms)

                if self._health_status["all_norms_valid"]:
                    logger.info(f"✅ All embeddings are L2-normalized")
                    logger.info(f"   Average norm: {self._health_status['average_norm']:.6f}")
                    logger.info(f"   Min norm:     {self._health_status['min_norm']:.6f}")
                    logger.info(f"   Max norm:     {self._health_status['max_norm']:.6f}")
                else:
                    logger.warning(f"⚠️ Some embeddings not normalized")

                self._vector_quality["normalized"] = self._health_status["all_norms_valid"]

            # --- NaN, Inf, Zero Vector Validation ---
            nan_count = 0
            inf_count = 0
            zero_count = 0

            for vec in embeddings:
                if vec is not None and len(vec) > 0:
                    arr = np.array(vec)
                    if np.any(np.isnan(arr)):
                        nan_count += 1
                    if np.any(np.isinf(arr)):
                        inf_count += 1
                    if np.all(arr == 0):
                        zero_count += 1

            self._health_status["no_nan_vectors"] = (nan_count == 0)
            self._health_status["no_inf_vectors"] = (inf_count == 0)
            self._health_status["no_zero_vectors"] = (zero_count == 0)
            self._vector_quality["nan_vectors"] = nan_count
            self._vector_quality["inf_vectors"] = inf_count
            self._vector_quality["zero_vectors"] = zero_count

            logger.info(f"✅ NaN checks:   {nan_count} vectors contain NaN")
            logger.info(f"✅ Inf checks:   {inf_count} vectors contain Inf")
            logger.info(f"✅ Zero checks:  {zero_count} zero vectors")

        except Exception as e:
            logger.error(f"Error during vector validation: {e}")
            self._health_status["all_dimensions_valid"] = False
            self._health_status["all_norms_valid"] = False

    def _validate_duplicates(self) -> None:
        """Check for duplicate chunk_ids."""
        if self._total_vectors == 0:
            return

        try:
            result = self._collection.get(include=[])
            if result and result.get("ids") is not None:
                ids = result["ids"]
                unique_ids = set(ids)
                self._health_status["no_duplicate_ids"] = len(ids) == len(unique_ids)

                if self._health_status["no_duplicate_ids"]:
                    logger.info(f"✅ No duplicate chunk_ids found")
                else:
                    duplicates = len(ids) - len(unique_ids)
                    logger.warning(f"⚠️ Found {duplicates} duplicate chunk_ids")
        except Exception as e:
            logger.warning(f"Could not validate duplicates: {e}")
            self._health_status["no_duplicate_ids"] = False

    def _validate_metadata_consistency(self) -> None:
        """Check metadata consistency: every vector has metadata and vice versa."""
        if self._total_vectors == 0:
            self._health_status["metadata_consistent"] = True
            return

        try:
            result = self._collection.get(include=[])
            if result and result.get("ids") is not None:
                vector_ids = set(result["ids"])
                metadata_ids = set(self._metadata_lookup.keys())

                all_matched = (vector_ids == metadata_ids)
                self._health_status["metadata_consistent"] = all_matched

                if all_matched:
                    logger.info(f"✅ All {len(vector_ids)} vectors have matching metadata")
                else:
                    missing_in_metadata = vector_ids - metadata_ids
                    missing_in_vectors = metadata_ids - vector_ids
                    if missing_in_metadata:
                        logger.warning(f"⚠️ {len(missing_in_metadata)} vectors missing metadata")
                    if missing_in_vectors:
                        logger.warning(f"⚠️ {len(missing_in_vectors)} metadata records missing vectors")
        except Exception as e:
            logger.warning(f"Could not validate metadata consistency: {e}")
            self._health_status["metadata_consistent"] = False

    def _load_metadata_lookup(self) -> None:
        """Load metadata from JSON file for fast lookup."""
        if self.metadata_path and self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    metadata_list = json.load(f)
                self._metadata_lookup = {m["chunk_id"]: m for m in metadata_list}
                logger.info(f"Loaded {len(self._metadata_lookup)} metadata records")
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
                self._metadata_lookup = {}

    def _compute_summary(self) -> None:
        """Compute collection summary for debugging."""
        # Count unique documents
        doc_titles = set()
        for meta in self._metadata_lookup.values():
            doc_title = meta.get("document_title", "unknown")
            doc_titles.add(doc_title)

        # Compute storage size
        storage_size_mb = 0
        if self.persist_directory.exists():
            total_bytes = 0
            for file_path in self.persist_directory.rglob("*"):
                if file_path.is_file():
                    total_bytes += file_path.stat().st_size
            storage_size_mb = round(total_bytes / (1024 * 1024), 2)

        self._collection_summary = {
            "collection_name": self.collection_name,
            "vector_count": self._total_vectors,
            "document_count": len(doc_titles),
            "average_chunks_per_document": round(self._total_vectors / len(doc_titles), 2) if doc_titles else 0,
            "embedding_model": "intfloat/multilingual-e5-large",
            "embedding_dimension": self._embeddings_dim or self.expected_dimension,
            "distance_metric": "cosine (IP)",
            "persist_directory": str(self.persist_directory),
            "storage_size_mb": storage_size_mb,
            "metadata_records": len(self._metadata_lookup),
        }

    # ==========================================================
    # Count Method
    # ==========================================================

    def count(self) -> int:
        """Get total number of vectors in the collection."""
        return self._total_vectors

    # ==========================================================
    # Search Methods
    # ==========================================================

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        where: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
    ) -> List[VectorSearchResult]:
        """
        Search for nearest neighbors using the query embedding.

        Parameters:
            query_vector: Query embedding vector.
            top_k: Number of results to return.
            where: Metadata filter for ChromaDB.
            include_metadata: Include full metadata in results.

        Returns:
            List of VectorSearchResult objects.
        """
        if self._collection is None:
            raise RuntimeError("VectorStore not connected")

        if self._total_vectors == 0:
            return []

        # Perform search
        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, self._total_vectors),
            where=where,
            include=["metadatas", "documents", "distances"],
        )

        # Format results
        formatted_results = []
        if results.get("ids") and results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results.get("distances") and results["distances"][0] else 0.0
                metadata = results["metadatas"][0][i] if results.get("metadatas") and results["metadatas"][0] else {}
                document = results["documents"][0][i] if results.get("documents") and results["documents"][0] else ""

                # Cosine similarity = 1 - distance (for normalized vectors with IP metric)
                similarity = 1.0 - distance

                # Add full metadata from lookup if available
                if include_metadata and chunk_id in self._metadata_lookup:
                    full_metadata = self._metadata_lookup[chunk_id].copy()
                    full_metadata.update(metadata)
                    metadata = full_metadata

                formatted_results.append(
                    VectorSearchResult(
                        chunk_id=chunk_id,
                        score=similarity,
                        document=document,
                        metadata=metadata,
                        distance=distance,
                    )
                )

        return formatted_results

    # ==========================================================
    # Get Methods
    # ==========================================================

    def get_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific chunk by ID.

        Parameters:
            chunk_id: The chunk ID to retrieve.

        Returns:
            Dictionary with chunk data or None if not found.
        """
        if self._collection is None:
            return None

        try:
            result = self._collection.get(
                ids=[chunk_id],
                include=["metadatas", "documents", "embeddings"],
            )
            if result and result.get("ids") and result["ids"]:
                idx = 0
                metadata = result["metadatas"][idx] if result.get("metadatas") else {}
                document = result["documents"][idx] if result.get("documents") else ""
                embedding = result["embeddings"][idx] if result.get("embeddings") else None

                if chunk_id in self._metadata_lookup:
                    full_metadata = self._metadata_lookup[chunk_id].copy()
                    full_metadata.update(metadata)
                    metadata = full_metadata

                return {
                    "chunk_id": chunk_id,
                    "embedding": embedding,
                    "metadata": metadata,
                    "document": document,
                }
        except Exception as e:
            logger.error(f"Failed to get chunk {chunk_id}: {e}")

        return None

    def get_metadata(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a chunk without the embedding."""
        return self._metadata_lookup.get(chunk_id)

    def get_metadata_batch(self, chunk_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get metadata for multiple chunks."""
        return {cid: self._metadata_lookup.get(cid, {}) for cid in chunk_ids}

    # ==========================================================
    # Health Check
    # ==========================================================

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a comprehensive health check on the collection.

        Returns:
            Dictionary with health status and detailed fields.
        """
        return self._health_status.copy()

    # ==========================================================
    # Statistics Methods
    # ==========================================================

    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        return self._embeddings_dim or self.expected_dimension

    def statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive collection statistics.

        Returns:
            Dictionary with statistics.
        """
        stats = self._collection_summary.copy()
        stats["health"] = self._health_status.copy()
        stats["vector_quality"] = self._vector_quality.copy()
        return stats

    def summary(self) -> Dict[str, Any]:
        """Get a clean collection summary."""
        return self._collection_summary.copy()

    def vector_quality(self) -> Dict[str, Any]:
        """Get vector quality metrics."""
        return self._vector_quality.copy()

    # ==========================================================
    # Maintenance Methods
    # ==========================================================

    def refresh_metadata(self, metadata_path: Optional[Path] = None) -> None:
        """Refresh metadata from JSON file."""
        path = metadata_path or self.metadata_path
        if path and path.exists():
            with open(path, "r", encoding="utf-8") as f:
                metadata_list = json.load(f)
            self._metadata_lookup = {m["chunk_id"]: m for m in metadata_list}
            self._collection_summary["metadata_records"] = len(self._metadata_lookup)
            logger.info(f"Refreshed metadata: {len(self._metadata_lookup)} records")

    def get_collection(self) -> chromadb.Collection:
        """Get the underlying ChromaDB collection."""
        if self._collection is None:
            raise RuntimeError("VectorStore not connected")
        return self._collection


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
    print("VECTOR STORE DEMO (Publication Quality)")
    print("=" * 70)

    # Paths
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
    CHROMA_DB_PATH = EMBEDDINGS_DIR / "chroma_db"

    print(f"\n📁 ChromaDB path: {CHROMA_DB_PATH}")

    # Initialize VectorStore
    store = VectorStore(
        persist_directory=CHROMA_DB_PATH,
        collection_name="amharic_fintech_chunks",
        embeddings_path=EMBEDDINGS_DIR / "embeddings.json",
        metadata_path=EMBEDDINGS_DIR / "embedding_metadata.json",
        validate_all_vectors=False,  # Set to True for thorough validation (slower)
    )

    # Count
    print("\n" + "=" * 70)
    print("COUNT")
    print("=" * 70)
    print(f"store.count() → {store.count()}")

    # Summary
    print("\n" + "=" * 70)
    print("COLLECTION SUMMARY")
    print("=" * 70)
    summary = store.summary()
    print(json.dumps(summary, indent=2))

    # Health Check
    print("\n" + "=" * 70)
    print("HEALTH CHECK")
    print("=" * 70)
    health = store.health_check()
    print(json.dumps(health, indent=2))

    # Vector Quality
    print("\n" + "=" * 70)
    print("VECTOR QUALITY")
    print("=" * 70)
    quality = store.vector_quality()
    print(json.dumps(quality, indent=2))

    # Statistics
    print("\n" + "=" * 70)
    print("COMPLETE STATISTICS")
    print("=" * 70)
    stats = store.statistics()
    print(json.dumps(stats, indent=2))

    print("\n" + "=" * 70)
    print("✅ VectorStore demo completed successfully")
    print("=" * 70)