# src/retrieval/retrieval_pipeline.py

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.embeddings.embedding_model import get_embedding_model
from src.morphology.query_analyzer import QueryAnalyzer
from src.retrieval.dense_retriever import RetrievalResult          # fixed import
from src.retrieval.adaptive_router import AdaptiveRouter, RoutingDecision
from src.retrieval.reranker import MorphologyAwareReranker

logger = logging.getLogger(__name__)


@dataclass
class RetrievalReport:
    original_query: str
    expanded_query: str
    strategy: str
    reasoning: str
    final_context: List[RetrievalResult] = field(default_factory=list)
    timing_ms: float = 0.0
    routing_decision: Optional[RoutingDecision] = None
    query_profile: Optional[Dict[str, Any]] = None
    chunks_returned: int = 0
    breakdown: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "expanded_query": self.expanded_query,
            "strategy": self.strategy,
            "reasoning": self.reasoning,
            "final_context": [r.to_dict() for r in self.final_context],
            "timing_ms": round(self.timing_ms, 3),
            "routing_decision": self.routing_decision.to_dict() if self.routing_decision else None,
            "query_profile": self.query_profile,
            "chunks_returned": self.chunks_returned,
            "breakdown": {k: round(v, 3) for k, v in self.breakdown.items()},
        }


class RetrievalPipeline:
    """End‑to‑end retrieval pipeline orchestrator."""
    
    def __init__(
        self,
        embedding_model_name: str = "intfloat/multilingual-e5-large",
        local_model_path: Optional[str] = None,
        device: Optional[str] = None,
        use_fp16: bool = False,
        use_mirror: bool = False,
        query_analyzer: Optional[QueryAnalyzer] = None,
        adaptive_router: Optional[AdaptiveRouter] = None,
        reranker: Optional[MorphologyAwareReranker] = None,
    ) -> None:
        logger.info("Loading embedding model for retrieval pipeline...")
        self.embedding_model = get_embedding_model(
            model_name=embedding_model_name,
            local_model_path=local_model_path,
            device=device,
            use_fp16=use_fp16,
            use_mirror=use_mirror,
        )
        
        self.query_analyzer = query_analyzer or QueryAnalyzer()
        
        if adaptive_router is None:
            raise ValueError("adaptive_router must be provided")
        self.adaptive_router = adaptive_router
        self.reranker = reranker
        
        model_name_str = getattr(self.embedding_model, "model_name", None) or \
                         getattr(self.embedding_model, "name", "unknown")
        logger.info(
            f"✅ RetrievalPipeline initialized:\n"
            f"   Embedding model: {model_name_str}\n"
            f"   Query analyzer: {self.query_analyzer.__class__.__name__}\n"
            f"   Adaptive router: {self.adaptive_router.__class__.__name__}\n"
            f"   Reranker: {reranker.__class__.__name__ if reranker else 'disabled'}"
        )
    
    def retrieve(
        self,
        query_text: str,
        query_profile: Optional[Dict[str, Any]] = None,
    ) -> RetrievalReport:
        if not query_text or not query_text.strip():
            raise ValueError("query_text cannot be empty")
        
        start_time = time.time()
        breakdown = {}
        logger.info(f"🚀 Starting retrieval pipeline for query: {query_text[:50]}...")
        
        # Encode
        encode_start = time.time()
        try:
            query_vector = self.embedding_model.encode_query(query_text)
            breakdown["encode_ms"] = (time.time() - encode_start) * 1000
        except Exception as e:
            raise RuntimeError(f"Query encoding failed: {e}") from e
        
        # Analyze
        analyze_start = time.time()
        try:
            if query_profile is None:
                query_profile = self.query_analyzer.analyze(query_text)
            breakdown["analyze_ms"] = (time.time() - analyze_start) * 1000
        except Exception as e:
            logger.warning(f"Query analysis failed: {e}. Using default profile.")
            query_profile = self._default_query_profile(query_text)
            breakdown["analyze_ms"] = (time.time() - analyze_start) * 1000
        
        # Route
        route_start = time.time()
        try:
            adaptive_result = self.adaptive_router.route(
                query_text=query_text,
                query_vector=query_vector,
                query_profile=query_profile,
            )
            breakdown["route_ms"] = (time.time() - route_start) * 1000
        except Exception as e:
            raise RuntimeError(f"Routing failed: {e}") from e
        
        # Rerank using ORIGINAL query for accuracy
        rerank_start = time.time()
        initial_chunks = adaptive_result.retrieved_chunks
        final_context = initial_chunks
        
        if adaptive_result.routing_decision.use_reranker and self.reranker:
            try:
                final_context = self.reranker.rerank(
                    query_text=query_text,
                    candidates=initial_chunks,
                    query_profile=query_profile,
                    top_n=adaptive_result.routing_decision.top_k,
                )
                breakdown["rerank_ms"] = (time.time() - rerank_start) * 1000
            except Exception as e:
                logger.warning(f"Reranking failed: {e}. Using un‑reranked results.")
                breakdown["rerank_ms"] = (time.time() - rerank_start) * 1000
                final_context = initial_chunks
        else:
            breakdown["rerank_ms"] = 0.0
            top_k = adaptive_result.routing_decision.top_k
            if len(final_context) > top_k:
                final_context = final_context[:top_k]
        
        total_time = (time.time() - start_time) * 1000
        breakdown["total_ms"] = total_time
        
        report = RetrievalReport(
            original_query=query_text,
            expanded_query=adaptive_result.expanded_query,
            strategy=adaptive_result.routing_decision.strategy,
            reasoning=adaptive_result.routing_decision.reasoning,
            final_context=final_context,
            timing_ms=total_time,
            routing_decision=adaptive_result.routing_decision,
            query_profile=query_profile,
            chunks_returned=len(final_context),
            breakdown=breakdown,
        )
        
        logger.info(
            f"✅ Retrieval pipeline completed:\n"
            f"   Strategy: {report.strategy}\n"
            f"   Chunks returned: {report.chunks_returned}\n"
            f"   Total time: {report.timing_ms:.1f}ms"
        )
        return report
    
    def _default_query_profile(self, query_text: str) -> Dict[str, Any]:
        return {
            "query": query_text,
            "cleaned_query": query_text,
            "morphology": {"morphology_complexity_score": 0.5, "affix_density": 0.3, "average_morphemes_per_word": 1.5},
            "complexity": {"overall": 0.5, "level": "medium"},
            "intent": {"question_type": "general", "query_category": "general"},
            "entity": {"name": None, "type": None},
            "query_features": {"token_count": len(query_text.split()), "contains_number": False, "contains_currency": False},
        }
    
    def retrieve_batch(
        self,
        queries: List[str],
        show_progress: bool = False,
    ) -> List[RetrievalReport]:
        logger.info(f"🚀 Starting batch retrieval for {len(queries)} queries")
        reports = []
        from tqdm import tqdm
        
        iterator = tqdm(queries, desc="Retrieving", disable=not show_progress)
        for i, query in enumerate(iterator, 1):
            try:
                report = self.retrieve(query_text=query)
                reports.append(report)
            except Exception as e:
                logger.error(f"Failed to retrieve for query {i}: {e}")
                reports.append(
                    RetrievalReport(
                        original_query=query,
                        expanded_query=query,
                        strategy="error",
                        reasoning=f"Retrieval failed: {e}",
                        final_context=[],
                        timing_ms=0.0,
                        chunks_returned=0,
                    )
                )
        return reports


# ==========================================================
# DEMO – fully corrected with the original, working signatures
# ==========================================================
if __name__ == "__main__":
    import logging
    from pathlib import Path

    from src.retrieval.vector_store import VectorStore
    from src.retrieval.dense_retriever import DenseRetriever
    from src.retrieval.sparse_retriever import SparseRetriever
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.adaptive_router import AdaptiveRouter
    from src.retrieval.reranker import MorphologyAwareReranker
    from src.morphology.query_analyzer import QueryAnalyzer

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    print("=" * 70)
    print("RETRIEVAL PIPELINE DEMO ⭐ END‑TO‑END ORCHESTRATION ⭐")
    print("=" * 70)

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
    CHROMA_DB_PATH = EMBEDDINGS_DIR / "chroma_db"
    CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "chunked_documents.json"
    LOCAL_MODEL_PATH = r"D:\models\multilingual-e5-large"

    print(f"\n📁 ChromaDB path: {CHROMA_DB_PATH}")
    print(f"📁 Corpus path: {CORPUS_PATH}")
    print(f"📁 Local model: {LOCAL_MODEL_PATH}")

    try:
        print("\n🔧 Initializing dependencies...")

        # VectorStore – uses persist_directory, not chroma_path
        vector_store = VectorStore(
            persist_directory=CHROMA_DB_PATH,
            collection_name="amharic_fintech_corpus",
            metadata_path=EMBEDDINGS_DIR / "embedding_metadata.json",
        )
        print(f"✅ VectorStore ready: {vector_store.count()} vectors")

        # DenseRetriever – original working signature (no 'embedding_model' object)
        dense_retriever = DenseRetriever(
            vector_store=vector_store,
            local_model_path=LOCAL_MODEL_PATH,
            device="cpu",
            use_fp16=False,
            top_k=10,
        )

        # SparseRetriever – only chunks_path (no top_k in constructor)
        sparse_retriever = SparseRetriever(
            chunks_path=CORPUS_PATH,
        )

        # HybridRetriever
        hybrid_retriever = HybridRetriever(rrf_k=60)

        query_analyzer = QueryAnalyzer()

        # AdaptiveRouter – with all three thresholds
        adaptive_router = AdaptiveRouter(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            hybrid_retriever=hybrid_retriever,
            low_threshold=0.30,
            high_threshold=0.60,
            morpho_override_threshold=0.65,
        )

        reranker = MorphologyAwareReranker()

        # Pipeline – loads its own embedding model internally
        pipeline = RetrievalPipeline(
            embedding_model_name="intfloat/multilingual-e5-large",
            local_model_path=LOCAL_MODEL_PATH,
            device="cpu",
            use_fp16=False,
            query_analyzer=query_analyzer,
            adaptive_router=adaptive_router,
            reranker=reranker,
        )

        test_queries = [
            "የCBE ካርድ እንዴት ማግኘት እችላለሁ?",
            "100,000 ብር ገደብ",
            "ይከፈላልን?",
        ]

        print("\n" + "=" * 70)
        print("PIPELINE RESULTS")
        print("=" * 70)

        for query in test_queries:
            print(f"\n📝 Query: {query}")
            report = pipeline.retrieve(query)

            print(f"\n   Strategy: {report.strategy}")
            print(f"   Reasoning: {report.reasoning}")
            print(f"   Chunks returned: {report.chunks_returned}")
            print(f"   Total time: {report.timing_ms:.1f}ms")
            print(f"   Breakdown: {report.breakdown}")

            print(f"\n   Top 3 Results:")
            for r in report.final_context[:3]:
                print(f"      Rank {r.rank}: {r.chunk_id} (score={r.score:.4f})")
                if "rerank_scores" in r.metadata:
                    rs = r.metadata["rerank_scores"]
                    print(f"         Rerank: semantic={rs.get('semantic',0):.3f}, "
                          f"morphology={rs.get('morphology',0):.3f}, "
                          f"entity={rs.get('entity',0):.3f}, "
                          f"domain={rs.get('domain',0):.3f}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("✅ RetrievalPipeline demo completed successfully")
    print("=" * 70)