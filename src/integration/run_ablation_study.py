"""
Ablation Study: Prove each component's contribution.

Compares 4 configurations of YOUR method:
  1. Hybrid RRF only (no morphology, no adaptive routing)
  2. Hybrid + Morphology Reranking (no adaptive routing)
  3. Hybrid + Adaptive Routing (no morphology reranking)
  4. Full Proposed Method (all components)

Generates Table 2 for your PhD paper.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    PROJECT_ROOT,
    CHUNKED_DOCS_PATH,
    EMBEDDINGS_DIR,
    CHROMA_DB_PATH,
    EVAL_DATASET_DIR,
    EVAL_REPORTS_DIR,
    EMBEDDING_CONFIG,
    RETRIEVAL_CONFIG,
)
from src.evaluation.retrieval_metrics import evaluate_retrieval_batch

logger = logging.getLogger(__name__)


class AblationRetrievers:
    """Container for ablation configurations."""
    
    def __init__(self):
        logger.info("🔧 Initializing ablation retrievers...")
        
        from src.retrieval.vector_store import VectorStore
        from src.retrieval.dense_retriever import DenseRetriever
        from src.retrieval.sparse_retriever import SparseRetriever
        from src.retrieval.hybrid_retriever import HybridRetriever
        from src.retrieval.adaptive_router import AdaptiveRouter
        from src.retrieval.reranker import MorphologyAwareReranker
        from src.morphology.query_analyzer import QueryAnalyzer
        from src.embeddings.embedding_model import get_embedding_model
        
        # Shared components
        self.vector_store = VectorStore(
            persist_directory=CHROMA_DB_PATH,
            collection_name="amharic_fintech_chunks",
            metadata_path=EMBEDDINGS_DIR / "embedding_metadata.json",
        )
        self.embedding_model = get_embedding_model(**EMBEDDING_CONFIG)
        
        self.dense_retriever = DenseRetriever(
            vector_store=self.vector_store,
            local_model_path=EMBEDDING_CONFIG["local_model_path"],
            device=EMBEDDING_CONFIG["device"],
            use_fp16=EMBEDDING_CONFIG["use_fp16"],
            top_k=10,
        )
        
        self.sparse_retriever = SparseRetriever(
            chunks_path=CHUNKED_DOCS_PATH,
            index_cache_path=PROJECT_ROOT / "data" / "retrieval" / "bm25_index.pkl",
        )
        
        self.hybrid_retriever = HybridRetriever(rrf_k=RETRIEVAL_CONFIG["hybrid_rrf_k"])
        self.query_analyzer = QueryAnalyzer()
        self.reranker = MorphologyAwareReranker()
        
        # ✅ FIX: Initialize adaptive_router HERE inside __init__
        self.adaptive_router = AdaptiveRouter(
            dense_retriever=self.dense_retriever,
            sparse_retriever=self.sparse_retriever,
            hybrid_retriever=self.hybrid_retriever,
            low_threshold=RETRIEVAL_CONFIG["low_threshold"],
            high_threshold=RETRIEVAL_CONFIG["high_threshold"],
            morpho_override_threshold=RETRIEVAL_CONFIG["morpho_override_threshold"],
        )
        
        logger.info("✅ Ablation retrievers initialized")
    
    def hybrid_only(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Config 1: Hybrid RRF only (baseline within proposed)."""
        dense_results = self.dense_retriever.retrieve(query_text=query, top_k=top_k * 2)
        sparse_results = self.sparse_retriever.retrieve(query_text=query, top_k=top_k * 2)
        
        fused = self.hybrid_retriever.fuse(
            dense_results=dense_results,
            sparse_results=sparse_results,
            dense_weight=0.5,
            sparse_weight=0.5,
            top_k=top_k,
        )
        return [{"chunk_id": r.chunk_id, "document": r.document} for r in fused]
    
    def hybrid_plus_morphology(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Config 2: Hybrid + Morphology Reranking (no adaptive routing)."""
        dense_results = self.dense_retriever.retrieve(query_text=query, top_k=top_k * 2)
        sparse_results = self.sparse_retriever.retrieve(query_text=query, top_k=top_k * 2)
        
        fused = self.hybrid_retriever.fuse(
            dense_results=dense_results,
            sparse_results=sparse_results,
            dense_weight=0.5,
            sparse_weight=0.5,
            top_k=top_k * 2,
        )
        
        reranked = self.reranker.rerank(
            query_text=query,
            candidates=fused,
            query_profile=None,
            top_n=top_k,
        )
        return [{"chunk_id": r.chunk_id, "document": r.document} for r in reranked]
    
    def hybrid_plus_adaptive(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Config 3: Hybrid + Adaptive Routing (no morphology reranking)."""
        # Analyze query
        query_profile = self.query_analyzer.analyze(query)
        query_vector = self.embedding_model.encode_query(query)
        
        # ✅ FIX: self.adaptive_router now exists because we initialized it in __init__
        adaptive_result = self.adaptive_router.route(
            query_text=query,
            query_vector=query_vector,
            query_profile=query_profile,
        )
        
        chunks = adaptive_result.retrieved_chunks[:top_k]
        return [{"chunk_id": r.chunk_id, "document": r.document} for r in chunks]
    
    def full_proposed(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Config 4: Full proposed method (all components)."""
        from src.retrieval.retrieval_pipeline import RetrievalPipeline
        
        pipeline = RetrievalPipeline(
            embedding_model_name=EMBEDDING_CONFIG["model_name"],
            local_model_path=EMBEDDING_CONFIG["local_model_path"],
            device=EMBEDDING_CONFIG["device"],
            use_fp16=EMBEDDING_CONFIG["use_fp16"],
            query_analyzer=self.query_analyzer,
            adaptive_router=self.adaptive_router,
            reranker=self.reranker,
        )
        
        report = pipeline.retrieve(query_text=query)
        chunks = report.final_context[:top_k]
        return [{"chunk_id": r.chunk_id, "document": r.document} for r in chunks]


def run_ablation_study(
    test_dataset_path: Path,
    output_dir: Path,
    top_k: int = 5
) -> Dict[str, Any]:
    """Run ablation study."""
    logger.info("🚀 Starting Ablation Study")
    
    with open(test_dataset_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    logger.info(f"✅ Loaded {len(test_data)} test queries")
    
    retrievers = AblationRetrievers()
    
    configurations = {
        "Hybrid RRF Only": retrievers.hybrid_only,
        "Hybrid + Morphology": retrievers.hybrid_plus_morphology,
        "Hybrid + Adaptive": retrievers.hybrid_plus_adaptive,
        "Full Proposed": retrievers.full_proposed,
    }
    
    results = {}
    
    for config_name, retriever_func in configurations.items():
        logger.info(f"\n{'='*70}")
        logger.info(f"🚀 Running: {config_name}")
        logger.info(f"{'='*70}")
        
        retrieval_results = []
        start_time = time.time()
        
        for i, item in enumerate(test_data, 1):
            query = item["query"]
            gold_chunks = item["gold_chunks"]
            
            if i % 10 == 0 or i == 1:
                logger.info(f"   [{i}/{len(test_data)}] {query[:40]}...")
            
            retrieved = retriever_func(query, top_k=top_k)
            retrieved_ids = [c["chunk_id"] for c in retrieved]
            
            retrieval_results.append({
                "retrieved_ids": retrieved_ids,
                "relevant_ids": gold_chunks,
            })
        
        elapsed = time.time() - start_time
        
        metrics = evaluate_retrieval_batch(retrieval_results, k_values=[1, 3, 5, 10])
        metrics["total_time_seconds"] = round(elapsed, 2)
        
        results[config_name] = metrics
        
        logger.info(
            f"✅ {config_name}: "
            f"Recall@5={metrics['Recall@5_mean']:.4f}, "
            f"MRR={metrics['MRR_mean']:.4f}, "
            f"Time={elapsed:.1f}s"
        )
    
    # Generate reports
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Markdown
    lines = ["# Ablation Study Results\n\n"]
    lines.append("| Configuration | Recall@5 | MRR | NDCG@5 | MAP |")
    lines.append("|---------------|----------|-----|--------|-----|")
    
    for config_name, metrics in results.items():
        recall = metrics.get("Recall@5_mean", 0.0)
        mrr = metrics.get("MRR_mean", 0.0)
        ndcg = metrics.get("NDCG@5_mean", 0.0)
        map_val = metrics.get("MAP_mean", 0.0)
        
        bold = "**" if config_name == "Full Proposed" else ""
        lines.append(
            f"| {bold}{config_name}{bold} | "
            f"{bold}{recall:.4f}{bold} | "
            f"{bold}{mrr:.4f}{bold} | "
            f"{bold}{ndcg:.4f}{bold} | "
            f"{bold}{map_val:.4f}{bold} |"
        )
    
    md_path = output_dir / "ablation_study.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    
    logger.info(f"✅ Ablation report: {md_path}")
    
    # JSON
    json_path = output_dir / "ablation_study.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✅ JSON report: {json_path}")
    
    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    TEST_DATASET_PATH = EVAL_DATASET_DIR / "test.json"
    OUTPUT_DIR = EVAL_REPORTS_DIR / "ablation"
    
    if not TEST_DATASET_PATH.exists():
        logger.error(f"❌ Test dataset not found: {TEST_DATASET_PATH}")
        sys.exit(1)
    
    results = run_ablation_study(TEST_DATASET_PATH, OUTPUT_DIR, top_k=5)
    
    print("\n" + "=" * 80)
    print("✅ ABLATION STUDY COMPLETE")
    print("=" * 80)
    print(f"📁 Reports saved to: {OUTPUT_DIR}")
    print("=" * 80)