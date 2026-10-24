# src/embeddings/embedding_model.py

"""
Embedding Model Wrapper for Amharic FinTech RAG.

This module is strictly responsible for loading the transformer model, 
handling tokenization, mean pooling, and L2 normalization. It provides 
a reusable encoder and performs NO file I/O.

Model Selection: intfloat/multilingual-e5-large
Rationale: State-of-the-art multilingual model with excellent performance 
on low-resource languages like Amharic. Supports asymmetric retrieval 
(passage vs. query prefixes).

CRITICAL: E5 requires prefixes for optimal performance:
    - Query: "query: " + text
    - Passage: "passage: " + text
This module automatically applies the correct prefix based on context.

Local Model Path: D:\models\multilingual-e5-large
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Singleton embedding model wrapper with E5 prefix support.

    Features:
        - SentenceTransformer backend (simpler, battle-tested)
        - Automatic E5 query/passage prefix addition
        - L2 normalization (cosine similarity ready)
        - CPU/GPU support with configurable FP16
        - Singleton pattern for efficient reuse
        - Local model path support (avoids re-downloading)
        - Hugging Face mirror support (for regions with access issues)
        - Robust error handling for common issues (missing config, etc.)
    """

    _instance: Optional["EmbeddingModel"] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs) -> "EmbeddingModel":
        """Singleton pattern: ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        local_model_path: Optional[str] = None,
        device: Optional[str] = None,
        use_fp16: bool = False,  # False for CPU; set True if GPU
        max_length: int = 512,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        use_mirror: bool = False,
    ) -> None:
        """
        Initialize the embedding model.

        Parameters:
            model_name: HuggingFace model name (default: intfloat/multilingual-e5-large)
            local_model_path: Local directory containing the model files.
                              If provided, uses local model instead of downloading.
            device: 'cuda', 'cpu', or None (auto-detect)
            use_fp16: Enable FP16 mixed precision (faster on GPU). Default False for CPU.
            max_length: Maximum token length for truncation
            batch_size: Default batch size for encoding
            normalize_embeddings: L2 normalize output vectors (recommended)
            use_mirror: If True, use HF mirror (hf-mirror.com) for downloads
        """
        # Singleton check: only initialize once
        if EmbeddingModel._initialized:
            logger.debug("EmbeddingModel already initialized, reusing instance")
            return

        self.model_name = model_name
        self.local_model_path = local_model_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.use_fp16 = use_fp16
        self.use_mirror = use_mirror

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Resolve model path (local or remote)
        model_path = self._resolve_model_path()

        # Load the model
        logger.info(f"Loading embedding model from: {model_path} on {self.device}...")

        try:
            # trust_remote_code=True helps bypass strict config parsing issues
            self.model = SentenceTransformer(
                model_path,
                device=self.device,
                trust_remote_code=True,
            )
        except TypeError as e:
            # This catches the specific error when SentenceTransformers 3.x fails to read pooling config
            if "embedding_dimension" in str(e):
                logger.error(
                    "❌ CRITICAL: SentenceTransformers failed to read the pooling config.\n"
                    "This usually means the '1_Pooling/config.json' file is missing or corrupted.\n"
                    "Please ensure your local model at D:\\models\\multilingual-e5-large is complete.\n"
                    "See the supervisor's instructions to fix this."
                )
            raise e

        # Enable FP16 only if requested AND on GPU
        if self.use_fp16 and self.device == "cuda":
            self.model.half()
            logger.info("✅ FP16 (mixed precision) enabled for model")
        elif self.use_fp16 and self.device == "cpu":
            logger.warning("⚠️ FP16 requested but model is on CPU. Ignoring.")

        # Ensure evaluation mode
        self.model.eval()

        # Get embedding dimension
        try:
            self.dimension = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            logger.warning(f"⚠️ Could not get embedding dimension via get_sentence_embedding_dimension(): {e}")
            # Fallback: manually infer from encoding
            test_emb = self.model.encode(["test"], convert_to_numpy=True)
            self.dimension = test_emb.shape[1]

        logger.info(f"✅ Model loaded successfully. Vector dimension: {self.dimension}")

        # Mark as initialized
        EmbeddingModel._initialized = True

    # ==========================================================
    # Model Path Resolution
    # ==========================================================

    def _resolve_model_path(self) -> str:
        """
        Resolve the model path to use.

        Priority:
            1. Local path if provided and exists with required files
            2. Use mirror if enabled (sets HF_ENDPOINT)
            3. Use model_name (download from Hugging Face)
        """
        # 1. Check local path
        if self.local_model_path:
            path_obj = Path(self.local_model_path)

            # Required files for a valid model
            required_files = ["config.json", "model.safetensors", "tokenizer.json"]

            if path_obj.exists():
                missing_files = [f for f in required_files if not (path_obj / f).exists()]
                if not missing_files:
                    logger.info(f"✅ Using validated local model from: {self.local_model_path}")
                    return str(path_obj.resolve())
                else:
                    logger.warning(
                        f"⚠️ Local model at '{self.local_model_path}' is incomplete.\n"
                        f"   Missing files: {missing_files}\n"
                        f"   Falling back to Hugging Face download."
                    )
            else:
                logger.warning(
                    f"⚠️ Local model path '{self.local_model_path}' does not exist.\n"
                    f"   Falling back to Hugging Face download."
                )

        # 2. Use mirror if requested
        if self.use_mirror:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            logger.info("🌐 Using Hugging Face mirror: https://hf-mirror.com")

        # 3. Use model name (download from HF)
        logger.info(f"⬇️ Downloading model from Hugging Face: {self.model_name}")
        return self.model_name

    # ==========================================================
    # E5 Prefix Methods
    # ==========================================================

    @staticmethod
    def _add_query_prefix(text: str) -> str:
        """Add E5 query prefix for retrieval queries."""
        return f"query: {text}"

    @staticmethod
    def _add_passage_prefix(text: str) -> str:
        """Add E5 passage prefix for document indexing."""
        return f"passage: {text}"

    def _apply_prefixes(self, texts: List[str], is_query: bool = True) -> List[str]:
        """
        Apply E5 prefixes to texts.

        CRITICAL: E5 requires these prefixes for optimal performance.
        Without them, retrieval quality degrades significantly.
        """
        if not texts:
            return []

        prefix_fn = self._add_query_prefix if is_query else self._add_passage_prefix
        return [prefix_fn(text) for text in texts]

    # ==========================================================
    # Core Encoding Methods
    # ==========================================================

    def encode_batch(
        self,
        texts: List[str],
        is_query: bool = True,
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode a batch of texts into dense vectors.

        Parameters:
            texts: List of text strings
            is_query: If True, add "query: " prefix; else add "passage: " prefix
            batch_size: Batch size (defaults to self.batch_size)
            show_progress: Show tqdm progress bar

        Returns:
            NumPy array of shape (len(texts), dimension) with L2-normalized vectors.
        """
        if not texts:
            return np.array([], dtype=np.float32)

        batch_size = batch_size or self.batch_size

        # Apply E5 prefixes (CRITICAL for performance)
        prefixed_texts = self._apply_prefixes(texts, is_query=is_query)

        # Encode with SentenceTransformer
        embeddings = self.model.encode(
            prefixed_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=show_progress,
        )

        # Ensure float32 (SentenceTransformer may return float16)
        if embeddings.dtype == np.float16:
            embeddings = embeddings.astype(np.float32)

        return embeddings

    # ==========================================================
    # Convenience Methods
    # ==========================================================

    def encode_query(self, text: str, show_progress: bool = False) -> np.ndarray:
        """Encode a single query string."""
        result = self.encode_batch([text], is_query=True, show_progress=show_progress)
        return result[0] if len(result) > 0 else np.array([], dtype=np.float32)

    def encode_queries(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode multiple query strings."""
        return self.encode_batch(
            texts,
            is_query=True,
            batch_size=batch_size,
            show_progress=show_progress,
        )

    def encode_passage(self, text: str, show_progress: bool = False) -> np.ndarray:
        """Encode a single passage string."""
        result = self.encode_batch([text], is_query=False, show_progress=show_progress)
        return result[0] if len(result) > 0 else np.array([], dtype=np.float32)

    def encode_passages(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode multiple passage strings."""
        return self.encode_batch(
            texts,
            is_query=False,
            batch_size=batch_size,
            show_progress=show_progress,
        )

    def __call__(self, texts: List[str]) -> np.ndarray:
        """Allow calling the instance directly (defaults to passage encoding)."""
        return self.encode_batch(texts, is_query=False)

    # ==========================================================
    # Utility Methods
    # ==========================================================

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.dimension

    def get_model_name(self) -> str:
        """Return the model name."""
        return self.model_name

    def get_device(self) -> str:
        """Return the device the model is on."""
        return self.device


# ==========================================================
# Singleton Factory
# ==========================================================

def get_embedding_model(
    model_name: str = "intfloat/multilingual-e5-large",
    local_model_path: Optional[str] = None,
    device: Optional[str] = None,
    use_fp16: bool = False,
    max_length: int = 512,
    batch_size: int = 32,
    normalize_embeddings: bool = True,
    use_mirror: bool = False,
) -> EmbeddingModel:
    """
    Get or create the singleton embedding model instance.

    Parameters:
        model_name: HuggingFace model name
        local_model_path: Local model directory (avoid re-downloading)
        device: 'cuda', 'cpu', or None (auto-detect)
        use_fp16: Enable FP16 on GPU
        max_length: Maximum token length
        batch_size: Default batch size
        normalize_embeddings: L2 normalize output
        use_mirror: Use HF mirror (hf-mirror.com)

    Returns:
        Singleton EmbeddingModel instance
    """
    model = EmbeddingModel(
        model_name=model_name,
        local_model_path=local_model_path,
        device=device,
        use_fp16=use_fp16,
        max_length=max_length,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        use_mirror=use_mirror,
    )
    return model


# ==========================================================
# DEMO / TEST
# ==========================================================

if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    print("=" * 60)
    print("EMBEDDING MODEL DEMO (with E5 Prefix Support)")
    print("=" * 60)

    # ==========================================================
    # YOUR LOCAL MODEL PATH - FIXED
    # ==========================================================
    LOCAL_MODEL_PATH = r"D:\models\multilingual-e5-large"

    print(f"\n📁 Using local model from: {LOCAL_MODEL_PATH}")

    # Load the model (singleton) from local path
    model = get_embedding_model(
        local_model_path=LOCAL_MODEL_PATH,  # Uses your downloaded model
        use_fp16=False,  # CPU doesn't need FP16
        device="cpu",    # Force CPU (remove this if you have GPU)
        use_mirror=False,  # No need for mirror since we have local model
    )

    print(f"\n✅ Model: {model.get_model_name()}")
    print(f"✅ Device: {model.get_device()}")
    print(f"✅ Dimension: {model.get_embedding_dim()}")

    # Test query encoding
    query = "የCBE ካርድ እንዴት ማግኘት እችላለሁ?"

    print("\n" + "=" * 40)
    print("Query Encoding (with 'query:' prefix)")
    print("=" * 40)

    query_emb = model.encode_query(query)
    print(f"Query: {query}")
    print(f"Prefixed: query: {query}")
    print(f"Embedding shape: {query_emb.shape}")
    print(f"Embedding norm (should be ~1.0): {np.linalg.norm(query_emb):.6f}")
    print(f"First 5 values: {query_emb[:5].tolist()}")

    # Test passage encoding
    passage = "የኢትዮጵያ ንግድ ባንክ የካርድ አገልግሎቶች ናቸው።"

    print("\n" + "=" * 40)
    print("Passage Encoding (with 'passage:' prefix)")
    print("=" * 40)

    passage_emb = model.encode_passage(passage)
    print(f"Passage: {passage}")
    print(f"Prefixed: passage: {passage}")
    print(f"Embedding shape: {passage_emb.shape}")
    print(f"Embedding norm (should be ~1.0): {np.linalg.norm(passage_emb):.6f}")
    print(f"First 5 values: {passage_emb[:5].tolist()}")

    # Quick validation
    print("\n" + "=" * 40)
    print("Validation")
    print("=" * 40)

    if abs(np.linalg.norm(query_emb) - 1.0) < 0.01:
        print("✅ Query embedding is properly normalized (norm ≈ 1.0)")
    else:
        print(f"⚠️ Query embedding norm is {np.linalg.norm(query_emb):.6f} (expected ~1.0)")

    if abs(np.linalg.norm(passage_emb) - 1.0) < 0.01:
        print("✅ Passage embedding is properly normalized (norm ≈ 1.0)")
    else:
        print(f"⚠️ Passage embedding norm is {np.linalg.norm(passage_emb):.6f} (expected ~1.0)")

    print("\n" + "=" * 60)
    print("✅ Embedding model test completed successfully!")
    print("=" * 60)