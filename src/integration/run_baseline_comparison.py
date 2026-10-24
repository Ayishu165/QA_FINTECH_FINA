"""
Baseline Comparison: Proposed Method vs. Standard Retrieval Methods.

Compares 4 retrieval strategies on the test dataset:
  1. BM25-only (Sparse baseline)
  2. Dense-only (Vector baseline)
  3. Hybrid RRF (Dense + Sparse fusion baseline)
  4. Proposed: Query-Adaptive Morphology-Aware RAG (your method)

Generates Table 1 for your PhD paper.
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
from src.evaluation.report_generator import EvaluationReportGenerator

logger = logging.getLogger(__name__)


# ============================================================
# BASELINE RETRIEVERS
# ============================================================

class BaselineRetrievers:
    """
    Container for all baseline retrieval strategies.
    Each method returns a list of chunk dicts with 'chunk_id' and 'document'.
    """
    
    def __init__(self):
        logger.info("🔧 Initializing baseline retrievers...")
        
        # Import real components
        from src.retrieval.vector_store import VectorStore
        from src.retrieval.dense_retriever import DenseRetriever
        from src.retrieval.sparse_retriever import SparseRetriever
        from src.retrieval.hybrid_retriever import HybridRetriever
        from src.retrieval.adaptive_router import AdaptiveRouter
        from src.retrieval.reranker import MorphologyAwareReranker
        from src.retrieval.retrieval_pipeline import RetrievalPipeline
        from src.morphology.query_analyzer import QueryAnalyzer
        from src.embeddings.embedding_model import get_embedding_model
        
        # 1. Vector Store
        self.vector_store = VectorStore(
            persist_directory=CHROMA_DB_PATH,
            collection_name="amharic_fintech_chunks",
            metadata_path=EMBEDDINGS_DIR / "embedding_metadata.json",
        )
        logger.info(f"   ✅ VectorStore: {self.vector_store.count()} vectors")
        
        # 2. Embedding Model
        self.embedding_model = get_embedding_model(**EMBEDDING_CONFIG)
        
        # 3. Dense Retriever
        self.dense_retriever = DenseRetriever(
            vector_store=self.vector_store,
            local_model_path=EMBEDDING_CONFIG["local_model_path"],
            device=EMBEDDING_CONFIG["device"],
            use_fp16=EMBEDDING_CONFIG["use_fp16"],
            top_k=10,
        )
        
        # 4. Sparse Retriever (BM25)
        self.sparse_retriever = SparseRetriever(
            chunks_path=CHUNKED_DOCS_PATH,
            index_cache_path=PROJECT_ROOT / "data" / "retrieval" / "bm25_index.pkl",
        )
        
        # 5. Hybrid Retriever (RRF)
        self.hybrid_retriever = HybridRetriever(rrf_k=RETRIEVAL_CONFIG["hybrid_rrf_k"])
        
        # 6. Query Analyzer
        self.query_analyzer = QueryAnalyzer()
        
        # 7. Adaptive Router
        self.adaptive_router = AdaptiveRouter(
            dense_retriever=self.dense_retriever,
            sparse_retriever=self.sparse_retriever,
            hybrid_retriever=self.hybrid_retriever,
            low_threshold=RETRIEVAL_CONFIG["low_threshold"],
            high_threshold=RETRIEVAL_CONFIG["high_threshold"],
            morpho_override_threshold=RETRIEVAL_CONFIG["morpho_override_threshold"],
        )
        
        # 8. Reranker
        self.reranker = MorphologyAwareReranker()
        
        # 9. Full Retrieval Pipeline (Proposed Method)
        self.proposed_pipeline = RetrievalPipeline(
            embedding_model_name=EMBEDDING_CONFIG["model_name"],
            local_model_path=EMBEDDING_CONFIG["local_model_path"],
            device=EMBEDDING_CONFIG["device"],
            use_fp16=EMBEDDING_CONFIG["use_fp16"],
            query_analyzer=self.query_analyzer,
            adaptive_router=self.adaptive_router,
            reranker=self.reranker,
        )
        
        logger.info("✅ All baseline retrievers initialized")
    
    def bm25_only(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Baseline 1: BM25 (Sparse) only."""
        results = self.sparse_retriever.retrieve(query_text=query, top_k=top_k)
        return [{"chunk_id": r.chunk_id, "document": r.document} for r in results]
    
    def dense_only(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Baseline 2: Dense (Vector) only."""
        results = self.dense_retriever.retrieve(query_text=query, top_k=top_k)
        return [{"chunk_id": r.chunk_id, "document": r.document} for r in results]
    
    def hybrid_rrf_only(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Baseline 3: Hybrid RRF (Dense + Sparse, no morphology)."""
        # Overfetch for fusion
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
    
    def proposed_method(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Proposed: Full Query-Adaptive Morphology-Aware RAG."""
        # Use the full pipeline (includes morphology analysis, adaptive routing, reranking)
        report = self.proposed_pipeline.retrieve(query_text=query)
        
        # Extract final chunks
        chunks = []
        for r in report.final_context[:top_k]:
            chunks.append({
                "chunk_id": r.chunk_id,
                "document": r.document,
                "score": r.score,
            })
        return chunks


# ============================================================
# BASELINE COMPARISON RUNNER
# ============================================================

class BaselineComparison:
    """
    Runs all baseline strategies on the test dataset and compares results.
    """
    
    def __init__(self, retrievers: BaselineRetrievers, top_k: int = 5):
        self.retrievers = retrievers
        self.top_k = top_k
        self.strategies = {
            "BM25 Only": retrievers.bm25_only,
            "Dense Only": retrievers.dense_only,
            "Hybrid RRF": retrievers.hybrid_rrf_only,
            "Proposed (Morphology-Aware)": retrievers.proposed_method,
        }
    
    def run_on_dataset(self, test_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Run all strategies on the test dataset.
        
        Returns:
            Dict mapping strategy name to evaluation results
        """
        results = {}
        
        for strategy_name, retriever_func in self.strategies.items():
            logger.info(f"\n{'='*70}")
            logger.info(f"🚀 Running: {strategy_name}")
            logger.info(f"{'='*70}")
            
            retrieval_results = []
            start_time = time.time()
            
            for i, item in enumerate(test_data, 1):
                query = item["query"]
                gold_chunks = item["gold_chunks"]
                
                if i % 10 == 0 or i == 1:
                    logger.info(f"   [{i}/{len(test_data)}] {query[:40]}...")
                
                # Run retrieval
                retrieved = retriever_func(query, top_k=self.top_k)
                retrieved_ids = [c["chunk_id"] for c in retrieved]
                
                retrieval_results.append({
                    "retrieved_ids": retrieved_ids,
                    "relevant_ids": gold_chunks,
                })
            
            elapsed = time.time() - start_time
            
            # Calculate metrics
            metrics = evaluate_retrieval_batch(
                retrieval_results,
                k_values=[1, 3, 5, 10]
            )
            metrics["total_time_seconds"] = round(elapsed, 2)
            metrics["avg_time_per_query_ms"] = round((elapsed / len(test_data)) * 1000, 2)
            
            results[strategy_name] = metrics
            
            logger.info(
                f"✅ {strategy_name}: "
                f"Recall@5={metrics['Recall@5_mean']:.4f}, "
                f"MRR={metrics['MRR_mean']:.4f}, "
                f"NDCG@5={metrics['NDCG@5_mean']:.4f}, "
                f"Time={elapsed:.1f}s"
            )
        
        return results
    
    def generate_comparison_table(self, results: Dict[str, Dict[str, Any]]) -> str:
        """
        Generate a publication-ready Markdown table.
        """
        lines = []
        lines.append("# Baseline Comparison Results\n")
        lines.append(f"**Test Dataset Size:** {len(next(iter(results.values())).get('num_queries', 'N/A'))} queries\n")
        lines.append(f"**Top-K:** {self.top_k}\n\n")
        
        # Table header
        strategies = list(results.keys())
        header = "| Metric | " + " | ".join(strategies) + " |"
        separator = "|" + "|".join(["---"] * (len(strategies) + 1)) + "|"
        
        lines.append(header)
        lines.append(separator)
        
        # Key metrics
        key_metrics = [
            ("Recall@1", "Recall@1_mean"),
            ("Recall@3", "Recall@3_mean"),
            ("Recall@5", "Recall@5_mean"),
            ("Precision@5", "Precision@5_mean"),
            ("MRR", "MRR_mean"),
            ("NDCG@5", "NDCG@5_mean"),
            ("MAP", "MAP_mean"),
            ("Hit Rate@5", "HitRate@5_mean"),
        ]
        
        for display_name, metric_key in key_metrics:
            values = []
            for strategy in strategies:
                value = results[strategy].get(metric_key, 0.0)
                values.append(f"**{value:.4f}**" if strategy == "Proposed (Morphology-Aware)" else f"{value:.4f}")
            
            lines.append(f"| {display_name} | " + " | ".join(values) + " |")
        
        # Performance metrics
        lines.append("| --- | --- | --- | --- | --- |")
        lines.append("| Total Time (s) | " + " | ".join(
            [f"{results[s]['total_time_seconds']:.1f}" for s in strategies]
        ) + " |")
        lines.append("| Avg Time/Query (ms) | " + " | ".join(
            [f"{results[s]['avg_time_per_query_ms']:.1f}" for s in strategies]
        ) + " |")
        
        return "\n".join(lines)
    
    def generate_improvement_analysis(self, results: Dict[str, Dict[str, Any]]) -> str:
        """
        Generate analysis showing improvement over baselines.
        """
        lines = []
        lines.append("\n## Improvement Analysis\n")
        
        proposed = results["Proposed (Morphology-Aware)"]
        baselines = ["BM25 Only", "Dense Only", "Hybrid RRF"]
        
        key_metrics = ["Recall@5", "MRR", "NDCG@5"]
        
        for baseline in baselines:
            lines.append(f"\n### Proposed vs. {baseline}\n")
            lines.append("| Metric | Baseline | Proposed | Improvement |")
            lines.append("|--------|----------|----------|-------------|")
            
            for metric in key_metrics:
                baseline_val = results[baseline].get(f"{metric}_mean", 0.0)
                proposed_val = proposed.get(f"{metric}_mean", 0.0)
                
                if baseline_val > 0:
                    improvement = ((proposed_val - baseline_val) / baseline_val) * 100
                    lines.append(
                        f"| {metric} | {baseline_val:.4f} | {proposed_val:.4f} | "
                        f"**+{improvement:.1f}%** |"
                    )
                else:
                    lines.append(f"| {metric} | {baseline_val:.4f} | {proposed_val:.4f} | N/A |")
        
        return "\n".join(lines)


# ============================================================
# MAIN EXECUTION
# ============================================================

def run_baseline_comparison(
    test_dataset_path: Path,
    output_dir: Path,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Main function to run baseline comparison.
    """
    logger.info("🚀 Starting Baseline Comparison")
    logger.info(f"📂 Dataset: {test_dataset_path}")
    logger.info(f"📁 Output: {output_dir}")
    
    # Load test dataset
    with open(test_dataset_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    logger.info(f"✅ Loaded {len(test_data)} test queries")
    
    # Initialize retrievers
    retrievers = BaselineRetrievers()
    
    # Run comparison
    comparison = BaselineComparison(retrievers, top_k=top_k)
    results = comparison.run_on_dataset(test_data)
    
    # Generate reports
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Markdown report
    md_content = comparison.generate_comparison_table(results)
    md_content += comparison.generate_improvement_analysis(results)
    
    md_path = output_dir / "baseline_comparison.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    logger.info(f"✅ Markdown report: {md_path}")
    
    # 2. JSON report
    json_path = output_dir / "baseline_comparison.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ JSON report: {json_path}")
    
    # 3. LaTeX table for paper
    latex_path = output_dir / "baseline_comparison.tex"
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write("% Table 1: Baseline Comparison\n")
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Retrieval Performance: Proposed Method vs. Baselines}\n")
        f.write("\\label{tab:baseline_comparison}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Method} & \\textbf{Recall@5} & \\textbf{MRR} & \\textbf{NDCG@5} & \\textbf{MAP} \\\\\n")
        f.write("\\hline\n")
        
        for strategy in ["BM25 Only", "Dense Only", "Hybrid RRF", "Proposed (Morphology-Aware)"]:
            r = results[strategy]
            recall = r.get("Recall@5_mean", 0.0)
            mrr = r.get("MRR_mean", 0.0)
            ndcg = r.get("NDCG@5_mean", 0.0)
            map_val = r.get("MAP_mean", 0.0)
            
            # Bold the proposed method
            if "Proposed" in strategy:
                f.write(f"\\textbf{{{strategy}}} & \\textbf{{{recall:.4f}}} & \\textbf{{{mrr:.4f}}} & \\textbf{{{ndcg:.4f}}} & \\textbf{{{map_val:.4f}}} \\\\\n")
            else:
                f.write(f"{strategy} & {recall:.4f} & {mrr:.4f} & {ndcg:.4f} & {map_val:.4f} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    logger.info(f"✅ LaTeX table: {latex_path}")
    
    return results


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    TEST_DATASET_PATH = EVAL_DATASET_DIR / "test.json"
    OUTPUT_DIR = EVAL_REPORTS_DIR / "baseline"
    
    if not TEST_DATASET_PATH.exists():
        logger.error(f"❌ Test dataset not found: {TEST_DATASET_PATH}")
        logger.info("💡 Please create test.json first (see Phase 1 instructions)")
        sys.exit(1)
    
    results = run_baseline_comparison(
        test_dataset_path=TEST_DATASET_PATH,
        output_dir=OUTPUT_DIR,
        top_k=5,
    )
    
    print("\n" + "=" * 80)
    print("✅ BASELINE COMPARISON COMPLETE")
    print("=" * 80)
    print(f"\n📁 Reports saved to: {OUTPUT_DIR}")
    print(f"   - baseline_comparison.md   (human-readable)")
    print(f"   - baseline_comparison.json (structured data)")
    print(f"   - baseline_comparison.tex  (LaTeX for paper)")
    print("=" * 80)