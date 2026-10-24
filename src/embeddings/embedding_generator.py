# src/embeddings/embedding_generator.py

"""
Embedding Generator for Amharic FinTech RAG.

This module reads document chunks, generates embeddings using the singleton
embedding model, and produces three distinct, structured outputs:
1. Embedding records (heavy vector data for embeddings.json)
2. Metadata records (lightweight lookup data for embedding_metadata.json)
3. Processing statistics (quality and performance report for embedding_statistics.json)

This strict separation ensures optimal storage and seamless ingestion into ChromaDB.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Use absolute import to avoid relative import issues
from src.embeddings.embedding_model import get_embedding_model

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings for document chunks and returns structured records.

    This class handles the complete embedding generation pipeline:
        1. Validates and deduplicates input chunks
        2. Prepares text for encoding (with proper E5 prefixes)
        3. Encodes in batches with progress tracking
        4. Constructs structured records with metadata
        5. Returns embedding records, metadata records, and statistics
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        local_model_path: Optional[str] = None,
        device: Optional[str] = None,
        use_fp16: bool = False,
        max_length: int = 512,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        use_mirror: bool = False,
    ):
        """
        Initialize the embedding generator.

        Parameters:
            model_name: HuggingFace model name (default: intfloat/multilingual-e5-large)
            local_model_path: Local directory containing the model files.
                              Recommended: D:\\models\\multilingual-e5-large
            device: 'cuda', 'cpu', or None (auto-detect)
            use_fp16: Enable FP16 mixed precision (faster on GPU). Default False.
            max_length: Maximum token length for truncation
            batch_size: Batch size for encoding
            normalize_embeddings: L2 normalize output vectors
            use_mirror: If True, use HF mirror (hf-mirror.com)
        """
        self.batch_size = batch_size
        self.use_mirror = use_mirror
        self.local_model_path = local_model_path
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings

        # Load the singleton model with all parameters
        logger.info("Loading embedding model...")
        self.model = get_embedding_model(
            model_name=model_name,
            local_model_path=local_model_path,
            device=device,
            use_fp16=use_fp16,
            max_length=max_length,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            use_mirror=use_mirror,
        )

        self.embedding_dim = self.model.get_embedding_dim()

        logger.info(
            f"✅ EmbeddingGenerator initialized:\n"
            f"   Model: {self.model.get_model_name()}\n"
            f"   Dimension: {self.embedding_dim}\n"
            f"   Device: {self.model.get_device()}\n"
            f"   Batch Size: {self.batch_size}\n"
            f"   Local Path: {self.local_model_path or 'Not provided'}"
        )

    # ==========================================================
    # Validation & Deduplication
    # ==========================================================

    def _validate_and_deduplicate(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate and deduplicate chunks for deterministic processing.

        - Skips chunks with empty or whitespace-only text.
        - Removes duplicate chunk_ids (keeps the first occurrence).
        - Sorts by chunk_id to guarantee reproducible output ordering.

        Returns:
            List of validated and deduplicated chunks.
        """
        if not chunks:
            return []

        # Filter out empty
        valid = []
        for chunk in chunks:
            text = chunk.get("chunk_text", "")
            if text and str(text).strip():
                valid.append(chunk)

        # Deduplicate by chunk_id
        seen_ids = set()
        unique = []
        for chunk in valid:
            cid = chunk.get("chunk_id")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                unique.append(chunk)

        # Sort for strict reproducibility
        unique.sort(key=lambda x: x.get("chunk_id", ""))

        logger.info(
            f"📊 Validation: {len(chunks)} input → {len(valid)} valid → "
            f"{len(unique)} unique chunks"
        )

        return unique

    # ==========================================================
    # Core Generation
    # ==========================================================

    def generate(
        self,
        chunks: List[Dict[str, Any]],
        is_query: bool = False,
        show_progress: bool = True,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process a list of chunks and generate embedding, metadata, and stats records.

        Parameters:
            chunks: List of chunk dicts (must have 'chunk_text' and 'chunk_id').
            is_query: If True, adds "query: " prefix; else "passage: " prefix.
            show_progress: Show a tqdm progress bar.

        Returns:
            Tuple of (embedding_records, metadata_records, statistics_dict).
        """
        start_time = time.time()

        logger.info(f"🚀 Starting embedding generation for {len(chunks)} chunks...")

        # 1. Validate
        validated = self._validate_and_deduplicate(chunks)
        if not validated:
            logger.warning("⚠️ No valid chunks to process.")
            return [], [], self._empty_stats()

        # 2. Prepare data
        texts = []
        meta_list = []
        for chunk in validated:
            text = str(chunk.get("chunk_text", "")).strip()
            if not text:
                continue
            texts.append(text)
            meta_list.append({
                "chunk_id": chunk.get("chunk_id", "unknown"),
                "document_title": chunk.get("document_title", "unknown"),
                "token_count": chunk.get("token_count", 0),
                "character_count": len(text),
                "sentence_count": chunk.get("sentence_count", 0),
                "complexity_score": chunk.get("complexity_score", 0.0),
                "complexity_level": chunk.get("complexity_level", "unknown"),
                "language": chunk.get("language", "am"),
            })

        total = len(texts)
        if total == 0:
            logger.warning("⚠️ No non-empty texts after validation.")
            return [], [], self._empty_stats()

        logger.info(
            f"📝 Encoding {total} chunks with batch_size={self.batch_size}, "
            f"is_query={is_query}"
        )

        # 3. Encode in batches with progress
        embedding_records = []
        metadata_records = []
        total_tokens = 0
        total_chars = 0
        failed_count = 0
        failed_batches = 0

        iterator = tqdm(
            range(0, total, self.batch_size),
            desc="Generating embeddings",
            disable=not show_progress,
            unit="batch",
        )

        for i in iterator:
            batch_texts = texts[i:i + self.batch_size]
            batch_meta = meta_list[i:i + self.batch_size]

            try:
                embeddings = self.model.encode_batch(
                    batch_texts,
                    is_query=is_query,
                    batch_size=len(batch_texts),
                    show_progress=False,
                )
            except Exception as e:
                logger.error(f"❌ Batch encoding failed at index {i}: {e}")
                failed_count += len(batch_texts)
                failed_batches += 1
                continue

            # 4. Build records for this batch
            for idx, meta in enumerate(batch_meta):
                emb_vector = embeddings[idx].tolist()
                norm = float(np.linalg.norm(embeddings[idx]))

                # Heavy record (for embeddings.json)
                embedding_records.append({
                    "chunk_id": meta["chunk_id"],
                    "document_title": meta["document_title"],
                    "embedding": emb_vector,
                    "embedding_dimension": self.embedding_dim,
                    "embedding_norm": round(norm, 6),
                    "token_count": meta["token_count"],
                    "character_count": meta["character_count"],
                    "sentence_count": meta["sentence_count"],
                    "embedding_model": self.model.get_model_name(),
                    "created_at": datetime.utcnow().isoformat(),
                })

                # Lightweight record (for embedding_metadata.json)
                metadata_records.append({
                    "chunk_id": meta["chunk_id"],
                    "document_title": meta["document_title"],
                    "token_count": meta["token_count"],
                    "character_count": meta["character_count"],
                    "sentence_count": meta["sentence_count"],
                    "complexity_score": meta["complexity_score"],
                    "complexity_level": meta["complexity_level"],
                    "language": meta["language"],
                    "embedding_dimension": self.embedding_dim,
                    "embedding_norm": round(norm, 6),
                    "embedding_model": self.model.get_model_name(),
                    "created_at": datetime.utcnow().isoformat(),
                })

                total_tokens += meta["token_count"]
                total_chars += meta["character_count"]

        elapsed = time.time() - start_time
        embedded_count = len(embedding_records)

        # Count unique documents
        doc_titles = set(r["document_title"] for r in metadata_records)

        # 5. Compute Statistics (for embedding_statistics.json)
        stats = {
            "model_name": self.model.get_model_name(),
            "vector_dimension": self.embedding_dim,
            "total_documents": len(doc_titles),
            "total_input_chunks": len(chunks),
            "total_validated_chunks": len(validated),
            "embedded_chunks": embedded_count,
            "failed_chunks": failed_count,
            "failed_batches": failed_batches,
            "average_tokens_per_chunk": round(total_tokens / embedded_count, 2) if embedded_count > 0 else 0.0,
            "average_characters_per_chunk": round(total_chars / embedded_count, 2) if embedded_count > 0 else 0.0,
            "average_embedding_norm": round(
                sum(r["embedding_norm"] for r in embedding_records) / embedded_count, 6
            ) if embedded_count > 0 else 0.0,
            "device": self.model.get_device(),
            "batch_size": self.batch_size,
            "processing_time_seconds": round(elapsed, 3),
            "throughput_chunks_per_second": round(embedded_count / elapsed, 2) if elapsed > 0 else 0.0,
            "use_mirror": self.use_mirror,
            "local_model_path": self.local_model_path,
            "normalize_embeddings": self.normalize_embeddings,
            "max_length": self.max_length,
            "use_fp16": self.use_fp16,
            "pipeline_version": "2.0",
            "created_at": datetime.utcnow().isoformat(),
        }

        logger.info(
            f"✅ Generated {embedded_count} embeddings in {elapsed:.2f}s "
            f"({stats['throughput_chunks_per_second']} chunks/s)"
        )

        return embedding_records, metadata_records, stats

    # ==========================================================
    # Helper: Empty Statistics
    # ==========================================================

    def _empty_stats(self) -> Dict[str, Any]:
        """Return empty statistics dictionary."""
        return {
            "model_name": self.model.get_model_name(),
            "vector_dimension": self.embedding_dim,
            "total_documents": 0,
            "total_input_chunks": 0,
            "total_validated_chunks": 0,
            "embedded_chunks": 0,
            "failed_chunks": 0,
            "failed_batches": 0,
            "average_tokens_per_chunk": 0.0,
            "average_characters_per_chunk": 0.0,
            "average_embedding_norm": 0.0,
            "device": self.model.get_device(),
            "batch_size": self.batch_size,
            "processing_time_seconds": 0.0,
            "throughput_chunks_per_second": 0.0,
            "use_mirror": self.use_mirror,
            "local_model_path": self.local_model_path,
            "normalize_embeddings": self.normalize_embeddings,
            "max_length": self.max_length,
            "use_fp16": self.use_fp16,
            "pipeline_version": "2.0",
            "created_at": datetime.utcnow().isoformat(),
        }


# ==========================================================
# DEMO
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
    print("EMBEDDING GENERATOR DEMO")
    print("=" * 70)

    # ==========================================================
    # YOUR LOCAL MODEL PATH - FIXED
    # ==========================================================
    LOCAL_MODEL_PATH = r"D:\models\multilingual-e5-large"

    print(f"\n📁 Using local model from: {LOCAL_MODEL_PATH}")

    # Sample chunks (simulating real data from chunked_documents.json)
    sample_chunks = [
        {
            "chunk_id": "cbe_cards_corpus.md::chunk_0000",
            "document_title": "cbe_cards_corpus.md",
            "chunk_text": "Commercial Bank of Ethiopia የካርድ አገልግሎቶች (CBE Cards) ለባንክ አገልግሎትዎ ምቾት ከአገር ውስጥ እና ዓለም አቀፍ የካርድ አገልግሎቶቻችን ይምረጡ።",
            "token_count": 25,
            "sentence_count": 1,
            "complexity_score": 0.45,
            "complexity_level": "medium",
            "language": "am",
        },
        {
            "chunk_id": "cbe_cards_corpus.md::chunk_0001",
            "document_title": "cbe_cards_corpus.md",
            "chunk_text": "ካርዱ 100,000 ብር ነው፣ በጣም ጠቃሚ ነው!",
            "token_count": 8,
            "sentence_count": 1,
            "complexity_score": 0.30,
            "complexity_level": "low",
            "language": "am",
        },
        {
            "chunk_id": "cbe_cards_corpus.md::chunk_0002",
            "document_title": "cbe_cards_corpus.md",
            "chunk_text": "",  # empty, should be skipped
            "token_count": 0,
            "sentence_count": 0,
            "complexity_score": 0.0,
            "complexity_level": "unknown",
            "language": "am",
        },
        {
            "chunk_id": "cbe_cards_corpus.md::chunk_0000",  # Duplicate, should be skipped
            "document_title": "cbe_cards_corpus.md",
            "chunk_text": "Duplicate text",
            "token_count": 2,
            "sentence_count": 1,
            "complexity_score": 0.1,
            "complexity_level": "low",
            "language": "am",
        },
    ]

    # Initialize the generator with local model
    generator = EmbeddingGenerator(
        model_name="intfloat/multilingual-e5-large",
        local_model_path=LOCAL_MODEL_PATH,  # Uses your downloaded model
        use_fp16=False,  # CPU doesn't need FP16
        device="cpu",    # Force CPU (remove if you have GPU)
        batch_size=2,
        use_mirror=False,  # No need for mirror since we have local model
    )

    print("\n" + "=" * 70)
    print("GENERATING EMBEDDINGS...")
    print("=" * 70)

    emb_records, meta_records, stats = generator.generate(
        sample_chunks,
        is_query=False,      # passage encoding (for documents)
        show_progress=True,
    )

    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)
    print(json.dumps(stats, indent=2))

    if emb_records:
        # Show first embedding record without the huge vector
        first_emb = emb_records[0].copy()
        first_emb["embedding"] = f"<{len(first_emb['embedding'])}-dim vector>"
        print("\n" + "=" * 70)
        print("FIRST EMBEDDING RECORD (vector truncated for display)")
        print("=" * 70)
        print(json.dumps(first_emb, indent=2, ensure_ascii=False))

        print("\n" + "=" * 70)
        print("FIRST METADATA RECORD")
        print("=" * 70)
        print(json.dumps(meta_records[0], indent=2, ensure_ascii=False))

        # Validate embedding norms
        print("\n" + "=" * 70)
        print("EMBEDDING VALIDATION")
        print("=" * 70)

        all_norms = [r["embedding_norm"] for r in emb_records]
        avg_norm = sum(all_norms) / len(all_norms)
        print(f"✅ Average embedding norm: {avg_norm:.6f} (should be ~1.0)")
        print(f"✅ Total records: {len(emb_records)}")
        print(f"✅ Embedding dimension: {emb_records[0]['embedding_dimension']}")

        if all(abs(n - 1.0) < 0.01 for n in all_norms):
            print("✅ All embeddings are properly normalized!")
        else:
            print("⚠️ Some embeddings have norm != 1.0")

    print("\n" + "=" * 70)
    print("✅ Embedding generator test completed successfully!")
    print("=" * 70)