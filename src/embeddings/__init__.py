# src/embeddings/__init__.py

"""
Embedding Module for Amharic FinTech RAG.

This module provides:
    - EmbeddingModel: Singleton model wrapper with E5 prefix support
    - get_embedding_model: Factory function for the singleton
    - EmbeddingGenerator: Generates embeddings and returns structured records
    - run_embedding_pipeline: Orchestrates the complete embedding workflow
"""

from .embedding_model import EmbeddingModel, get_embedding_model
from .embedding_generator import EmbeddingGenerator
from .embedding_pipeline import run_embedding_pipeline

__all__ = [
    "EmbeddingModel",
    "get_embedding_model",
    "EmbeddingGenerator",
    "run_embedding_pipeline",
]