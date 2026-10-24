"""
Retrieval Module for Amharic FinTech RAG.
This file exports shared dataclasses used across all retrieval components.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RetrievalResult:
    """
    Standardized retrieval result returned by all retrievers.
    """
    chunk_id: str
    score: float
    document: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    retriever_name: str = "unknown"
    rank: int = 0
    fusion_score: Optional[float] = None
    rerank_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "chunk_id": self.chunk_id,
            "score": round(self.score, 6),
            "document": self.document,
            "metadata": self.metadata,
            "retriever_name": self.retriever_name,
            "rank": self.rank,
        }
        if self.fusion_score is not None:
            result["fusion_score"] = round(self.fusion_score, 6)
        if self.rerank_score is not None:
            result["rerank_score"] = round(self.rerank_score, 6)
        return result


__all__ = ["RetrievalResult"]