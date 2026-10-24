# src/embeddings/embedding_pipeline.py

"""
Embedding Pipeline Orchestrator for Amharic FinTech RAG.

This module orchestrates the end-to-end embedding generation workflow.
It reads pre-processed chunked documents, drives the embedding generator,
saves the resulting vectors, metadata, and statistics to disk, and optionally
syncs embeddings to ChromaDB for production retrieval.

This strict separation of generation and persistence ensures 100%
computational reproducibility, a critical requirement for Q1 research.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from src.embeddings.embedding_generator import EmbeddingGenerator

logger = logging.getLogger(__name__)


def save_json(data: Any, path: Path) -> None:
    """
    Saves data to a JSON file with UTF-8 encoding and pretty printing.

    Parameters:
        data: The Python object to serialize (dict, list, etc.).
        path: The destination file path.
    """
    data_size = len(data) if isinstance(data, (list, dict)) else 1
    logger.info(f"💾 Saving {data_size} items to: {path.name}")

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def sync_to_chromadb(
    embedding_records: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
    chroma_path: Path,
    collection_name: str = "amharic_fintech_chunks",
) -> Dict[str, Any]:
    """
    Sync embeddings to a persistent ChromaDB collection.

    Parameters:
        embedding_records: List of full embedding records.
        chunks: Original chunks (for document text lookup).
        chroma_path: Directory for ChromaDB persistence.
        collection_name: Name of the ChromaDB collection.

    Returns:
        Dictionary with sync statistics.
    """
    if not embedding_records:
        logger.warning("⚠️ No records to sync to ChromaDB.")
        return {"synced": 0, "status": "skipped"}

    logger.info(f"🔄 Syncing {len(embedding_records)} embeddings to ChromaDB...")

    # Initialize ChromaDB client
    client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(anonymized_telemetry=False),
    )

    # Get or create collection (using Inner Product for cosine similarity on normalized vectors)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "ip"},
    )

    logger.info(f"📁 ChromaDB collection: {collection_name} at {chroma_path}")

    # Build a text lookup for documents
    text_lookup = {chunk.get("chunk_id"): chunk.get("chunk_text", "") for chunk in chunks}

    # Prepare data
    ids = []
    embeddings = []
    metadatas = []
    documents = []

    for rec in embedding_records:
        cid = rec["chunk_id"]
        ids.append(cid)
        embeddings.append(rec["embedding"])

        # Get document text (or empty string if not found)
        doc_text = text_lookup.get(cid, "")
        documents.append(doc_text[:1000] if doc_text else "")

        # Flatten metadata for ChromaDB
        metadatas.append({
            "document_title": rec["document_title"],
            "token_count": rec["token_count"],
            "character_count": rec["character_count"],
            "sentence_count": rec.get("sentence_count", 0),
            "complexity_score": rec.get("complexity_score", 0.0),
            "complexity_level": rec.get("complexity_level", "unknown"),
            "embedding_norm": rec["embedding_norm"],
            "embedding_model": rec["embedding_model"],
            "created_at": rec["created_at"],
        })

    # Batch upsert (100 at a time)
    batch_size = 100
    total = len(ids)
    batches = (total + batch_size - 1) // batch_size

    logger.info(f"📦 Upserting {total} records in {batches} batches...")

    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        try:
            collection.upsert(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                metadatas=metadatas[i:end],
                documents=documents[i:end],
            )
            logger.debug(f"   Batch {i//batch_size + 1}/{batches} completed")
        except Exception as e:
            logger.error(f"❌ Failed to upsert batch {i//batch_size + 1}: {e}")
            raise

    # Get collection count
    try:
        count = collection.count()
        logger.info(f"📊 ChromaDB collection now contains {count} records")
    except Exception:
        count = total

    logger.info(f"✅ Successfully synced {total} embeddings to ChromaDB collection '{collection_name}'")

    return {
        "synced": total,
        "collection_name": collection_name,
        "collection_count": count,
        "status": "success",
    }


def run_embedding_pipeline(
    input_path: Path,
    output_dir: Path,
    model_name: str = "intfloat/multilingual-e5-large",
    local_model_path: Optional[str] = None,
    batch_size: int = 32,
    device: Optional[str] = None,
    use_fp16: bool = False,
    use_mirror: bool = False,
    max_length: int = 512,
    normalize_embeddings: bool = True,
    sync_to_chroma: bool = True,
    chroma_collection_name: str = "amharic_fintech_chunks",
) -> Dict[str, Any]:
    """
    Executes the complete embedding generation and saving workflow.

    Purpose:
        Reads chunked documents, generates embeddings using the specified model,
        saves the embeddings, metadata, and statistics to the output directory,
        and optionally syncs to ChromaDB.

    Parameters:
        input_path: Path to the input JSON file (e.g., chunked_documents.json).
        output_dir: Directory to save the output JSON files.
        model_name: HuggingFace model identifier.
        local_model_path: Path to the locally downloaded model directory.
        batch_size: Number of chunks to encode per batch.
        device: 'cuda', 'cpu', or None (auto-detect).
        use_fp16: Enable FP16 mixed precision (recommended for CUDA).
        use_mirror: Use Hugging Face mirror (hf-mirror.com).
        max_length: Maximum token length for truncation.
        normalize_embeddings: L2 normalize output vectors.
        sync_to_chroma: If True, sync embeddings to ChromaDB.
        chroma_collection_name: Name of the ChromaDB collection.

    Returns:
        Dictionary with pipeline statistics.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If the input JSON is not a list or is empty.
        RuntimeError: If the embedding generation fails catastrophically.
    """
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    pipeline_start = time.time()

    logger.info("=" * 70)
    logger.info("🚀 STARTING EMBEDDING PIPELINE")
    logger.info("=" * 70)

    # 1. Validate Input
    if not input_path.exists():
        raise FileNotFoundError(f"❌ Input file not found: {input_path}")

    logger.info(f"📂 Loading chunks from: {input_path}")
    with input_path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not isinstance(chunks, list):
        raise ValueError("❌ Input JSON must be a list of chunk dictionaries.")

    if not chunks:
        raise ValueError("❌ Input JSON list is empty. Nothing to process.")

    logger.info(f"📄 Loaded {len(chunks)} chunks.")

    # 2. Prepare Output Directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output directory: {output_dir}")

    # 3. Initialize Generator
    logger.info("🔧 Initializing EmbeddingGenerator...")
    generator = EmbeddingGenerator(
        model_name=model_name,
        local_model_path=local_model_path,
        device=device,
        use_fp16=use_fp16,
        max_length=max_length,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        use_mirror=use_mirror,
    )

    # 4. Generate Embeddings, Metadata, and Stats
    logger.info("📝 Starting embedding generation...")
    generation_start = time.time()

    try:
        # is_query=False because we are encoding document passages, not user queries
        embedding_records, metadata_records, stats = generator.generate(
            chunks=chunks,
            is_query=False,
            show_progress=True,
        )
    except Exception as e:
        logger.error(f"❌ Catastrophic failure during embedding generation: {e}")
        raise RuntimeError("Embedding generation failed.") from e

    generation_time = time.time() - generation_start
    logger.info(f"✅ Generation completed in {generation_time:.2f} seconds.")

    # 5. Save Artifacts
    logger.info("💾 Saving outputs...")
    save_json(embedding_records, output_dir / "embeddings.json")
    save_json(metadata_records, output_dir / "embedding_metadata.json")

    # Add pipeline metadata to stats
    stats.update({
        "generation_time_seconds": round(generation_time, 3),
        "total_pipeline_time_seconds": round(time.time() - pipeline_start, 3),
        "input_file": str(input_path),
        "output_directory": str(output_dir),
        "model_name": model_name,
        "local_model_path": local_model_path,
        "device": device or "auto-detected",
        "batch_size": batch_size,
        "use_fp16": use_fp16,
        "use_mirror": use_mirror,
        "max_length": max_length,
        "normalize_embeddings": normalize_embeddings,
        "pipeline_version": "2.0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })

    save_json(stats, output_dir / "embedding_statistics.json")

    # 6. Sync to ChromaDB
    chromadb_result = {}
    if sync_to_chroma and embedding_records:
        try:
            chroma_path = output_dir / "chroma_db"
            chromadb_result = sync_to_chromadb(
                embedding_records=embedding_records,
                chunks=chunks,
                chroma_path=chroma_path,
                collection_name=chroma_collection_name,
            )
            stats["chromadb"] = chromadb_result
        except Exception as e:
            logger.error(f"❌ ChromaDB sync failed: {e}")
            stats["chromadb"] = {"status": "failed", "error": str(e)}
    elif sync_to_chroma and not embedding_records:
        logger.warning("⚠️ No embeddings to sync to ChromaDB.")
        stats["chromadb"] = {"status": "skipped", "reason": "no_embeddings"}

    # 7. Final Summary
    total_time = time.time() - pipeline_start
    logger.info("=" * 70)
    logger.info("✅ EMBEDDING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)
    logger.info(f"📊 Total chunks processed: {stats['embedded_chunks']}")
    logger.info(f"⏱️  Generation time: {generation_time:.2f}s")
    logger.info(f"⏱️  Total pipeline time: {total_time:.2f}s")
    logger.info(f"📁 Output directory: {output_dir}")
    if sync_to_chroma and "chromadb" in stats:
        logger.info(f"📦 ChromaDB status: {stats['chromadb'].get('status', 'unknown')}")
    logger.info("=" * 70)

    return stats


# ==========================================================
# CLASS WRAPPER (for compatibility with existing imports)
# ==========================================================

class EmbeddingPipeline:
    """
    Wrapper class for the embedding pipeline.

    This provides a class-based interface for the run_embedding_pipeline function.
    Use this if you prefer object-oriented style or if your existing code
    expects an EmbeddingPipeline class.

    Example:
        pipeline = EmbeddingPipeline(
            input_path="data/processed/chunked_documents.json",
            output_dir="data/embeddings",
            local_model_path="D:/models/multilingual-e5-large",
        )
        stats = pipeline.build()
    """

    def __init__(
        self,
        input_path: Path,
        output_dir: Path,
        model_name: str = "intfloat/multilingual-e5-large",
        local_model_path: Optional[str] = None,
        batch_size: int = 32,
        device: Optional[str] = None,
        use_fp16: bool = False,
        use_mirror: bool = False,
        max_length: int = 512,
        normalize_embeddings: bool = True,
        sync_to_chroma: bool = True,
        chroma_collection_name: str = "amharic_fintech_chunks",
    ):
        """
        Initialize the pipeline with parameters.

        All parameters are passed directly to run_embedding_pipeline.
        """
        self.input_path = input_path
        self.output_dir = output_dir
        self.model_name = model_name
        self.local_model_path = local_model_path
        self.batch_size = batch_size
        self.device = device
        self.use_fp16 = use_fp16
        self.use_mirror = use_mirror
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings
        self.sync_to_chroma = sync_to_chroma
        self.chroma_collection_name = chroma_collection_name

    def build(self) -> Dict[str, Any]:
        """Run the pipeline with the stored parameters."""
        return run_embedding_pipeline(
            input_path=self.input_path,
            output_dir=self.output_dir,
            model_name=self.model_name,
            local_model_path=self.local_model_path,
            batch_size=self.batch_size,
            device=self.device,
            use_fp16=self.use_fp16,
            use_mirror=self.use_mirror,
            max_length=self.max_length,
            normalize_embeddings=self.normalize_embeddings,
            sync_to_chroma=self.sync_to_chroma,
            chroma_collection_name=self.chroma_collection_name,
        )


# ==========================================================
# MAIN EXECUTION
# ==========================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Resolve project paths dynamically
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "chunked_documents.json"
    DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "embeddings"
    DEFAULT_MODEL_PATH = Path(r"D:\models\multilingual-e5-large")

    parser = argparse.ArgumentParser(
        description="Run the embedding pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with local model
  python -m src.embeddings.embedding_pipeline

  # Run with GPU and FP16
  python -m src.embeddings.embedding_pipeline --device cuda --fp16 --batch-size 64

  # Run with mirror (if Hugging Face is blocked)
  python -m src.embeddings.embedding_pipeline --mirror --device cpu

  # Skip ChromaDB sync
  python -m src.embeddings.embedding_pipeline --no-chromadb
        """
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help=f"Path to chunked_documents.json (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--model", type=str, default="intfloat/multilingual-e5-large",
        help="Model name (default: intfloat/multilingual-e5-large)"
    )
    parser.add_argument(
        "--local-model", type=Path, default=DEFAULT_MODEL_PATH,
        help=f"Local model directory (default: {DEFAULT_MODEL_PATH})"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'cuda' or 'cpu' (auto-detect if not specified)"
    )
    parser.add_argument(
        "--fp16", action="store_true",
        help="Enable FP16 (only recommended on GPU)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for encoding (default: 32)"
    )
    parser.add_argument(
        "--mirror", action="store_true",
        help="Use Hugging Face mirror (hf-mirror.com)"
    )
    parser.add_argument(
        "--collection", type=str, default="amharic_fintech_chunks",
        help="ChromaDB collection name (default: amharic_fintech_chunks)"
    )
    parser.add_argument(
        "--no-chromadb", action="store_true",
        help="Skip ChromaDB sync (only save JSON files)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("EMBEDDING PIPELINE")
    print("=" * 70)
    print(f"📁 Input:  {args.input}")
    print(f"📁 Output: {args.output}")
    print(f"🤖 Model:  {args.model}")
    print(f"📂 Local:  {args.local_model if args.local_model else 'Not provided'}")
    print(f"💻 Device: {args.device or 'auto-detect'}")
    print(f"📦 Batch:  {args.batch_size}")
    print(f"🔀 Mirror: {args.mirror}")
    print(f"📦 Chroma: {'Skip' if args.no_chromadb else 'Sync'}")
    print("=" * 70)

    try:
        stats = run_embedding_pipeline(
            input_path=args.input,
            output_dir=args.output,
            model_name=args.model,
            local_model_path=str(args.local_model) if args.local_model else None,
            batch_size=args.batch_size,
            device=args.device,
            use_fp16=args.fp16,
            use_mirror=args.mirror,
            sync_to_chroma=not args.no_chromadb,
            chroma_collection_name=args.collection,
        )

        print("\n" + "=" * 70)
        print("PIPELINE EXECUTION REPORT")
        print("=" * 70)
        print(json.dumps(stats, indent=2, ensure_ascii=False))

    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        raise