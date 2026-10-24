"""
Full Pipeline Integration Test
Connects: Retrieval → Generation → Evaluation
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.generation.generation_pipeline import GenerationPipeline
from src.evaluation.evaluator import RAGEvaluator
from src.evaluation.report_generator import EvaluationReportGenerator

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class FullPipelineIntegration:
    """
    Orchestrates the complete RAG pipeline from query to evaluation.
    """
    
    def __init__(self, config_path: Path = None):
        self.config_path = config_path
        self.retrieval_pipeline = None
        self.generation_pipeline = None
        self.evaluator = None
        self.report_generator = None
        
    def initialize(self):
        """Initialize all pipeline components."""
        logger.info("🔧 Initializing Full Pipeline...")
        
        # 1. Initialize Retrieval Pipeline
        # NOTE: You need to provide your actual initialization code here
        # This is a placeholder - replace with your real setup
        from src.retrieval.vector_store import VectorStore
        from src.retrieval.dense_retriever import DenseRetriever
        from src.retrieval.sparse_retriever import SparseRetriever
        from src.retrieval.hybrid_retriever import HybridRetriever
        from src.retrieval.adaptive_router import AdaptiveRouter
        from src.retrieval.reranker import MorphologyAwareReranker
        from src.morphology.query_analyzer import QueryAnalyzer
        from src.embeddings.embedding_model import get_embedding_model
        
        PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
        EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
        CHROMA_DB_PATH = EMBEDDINGS_DIR / "chroma_db"
        CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "chunked_documents.json"
        LOCAL_MODEL_PATH = r"D:\models\multilingual-e5-large"
        
        # Initialize components
        vector_store = VectorStore(
            persist_directory=CHROMA_DB_PATH,
            collection_name="amharic_fintech_chunks",
            metadata_path=EMBEDDINGS_DIR / "embedding_metadata.json",
        )
        
        dense_retriever = DenseRetriever(
            vector_store=vector_store,
            local_model_path=LOCAL_MODEL_PATH,
            device="cpu",
            use_fp16=False,
            top_k=10,
        )
        
        sparse_retriever = SparseRetriever(chunks_path=CORPUS_PATH)
        hybrid_retriever = HybridRetriever(rrf_k=60)
        query_analyzer = QueryAnalyzer()
        
        adaptive_router = AdaptiveRouter(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            hybrid_retriever=hybrid_retriever,
            low_threshold=0.30,
            high_threshold=0.60,
            morpho_override_threshold=0.65,
        )
        
        reranker = MorphologyAwareReranker()
        
        self.retrieval_pipeline = RetrievalPipeline(
            embedding_model_name="intfloat/multilingual-e5-large",
            local_model_path=LOCAL_MODEL_PATH,
            device="cpu",
            use_fp16=False,
            query_analyzer=query_analyzer,
            adaptive_router=adaptive_router,
            reranker=reranker,
        )
        
        # 2. Initialize Generation Pipeline
        from src.generation.prompt_builder import PromptBuilder
        from src.generation.llm_client import LLMClient, LLMConfig
        from src.generation.citation_generator import CitationGenerator
        from src.generation.quality_checker import QualityChecker
        
        # Use Mock LLM for testing (replace with Ollama later)
        llm_client = LLMClient(config=LLMConfig(backend="mock", model="mock-amharic-v2"))
        
        self.generation_pipeline = GenerationPipeline(
            retrieval_pipeline=self.retrieval_pipeline,
            prompt_builder=PromptBuilder(max_chunks=5, include_scores=True),
            llm_client=llm_client,
            citation_generator=CitationGenerator(extract_context=True),
            quality_checker=QualityChecker(min_support_threshold=0.3),
        )
        
        # 3. Initialize Evaluator
        self.evaluator = RAGEvaluator(k_values=[1, 3, 5, 10])
        
        # 4. Initialize Report Generator
        output_dir = PROJECT_ROOT / "src" / "evaluation" / "reports" / "full_pipeline"
        self.report_generator = EvaluationReportGenerator(output_dir=output_dir)
        
        logger.info("✅ Full Pipeline initialized successfully")
    
    def run_single_query(self, query: str) -> Dict[str, Any]:
        """Run the full pipeline on a single query."""
        logger.info(f"🚀 Processing query: {query[:50]}...")
        
        # Step 1: Generation (includes retrieval)
        generation_report = self.generation_pipeline.generate(query)
        
        return generation_report.to_dict()
    
    def run_evaluation(self, test_dataset_path: Path) -> Dict[str, Any]:
        """Run the full pipeline on the test dataset and evaluate."""
        logger.info(f"📊 Loading test dataset: {test_dataset_path}")
        
        with open(test_dataset_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        logger.info(f"✅ Loaded {len(test_data)} test queries")
        
        # Run pipeline on all queries
        generation_reports = []
        retrieval_ground_truth = []
        morphology_data = []
        
        for i, item in enumerate(test_data, 1):
            query = item["query"]
            gold_chunks = item["gold_chunks"]
            gold_answer = item.get("gold_answer", "")
            
            logger.info(f"[{i}/{len(test_data)}] Processing: {query[:40]}...")
            
            # Run full pipeline
            report = self.run_single_query(query)
            generation_reports.append(report)
            
            # Prepare retrieval ground truth
            retrieval_report = report.get("retrieval", {})
            retrieved_chunks = [c.get("chunk_id", "") for c in retrieval_report.get("final_context", [])]
            
            retrieval_ground_truth.append({
                "retrieved_ids": retrieved_chunks,
                "relevant_ids": gold_chunks
            })
            
            # Prepare morphology data
            query_profile = retrieval_report.get("query_profile", {})
            morphology_data.append({
                "predicted_complexity": query_profile.get("morphology", {}),
                "gold_complexity": item.get("complexity", {}),
                "routing_decision": {"strategy": retrieval_report.get("strategy", "")},
                "expected_routing": {"expected_strategy": "hybrid_morpho_expanded" if item.get("complexity", {}).get("morphology_score", 0) >= 0.65 else "any"},
                "query": query,
                "retrieved_chunks": retrieved_chunks,
                "gold_chunks": gold_chunks,
                "retrieved_chunks_text": [c.get("document", "") for c in retrieval_report.get("final_context", [])]
            })
        
        # Run evaluation
        logger.info("📊 Running comprehensive evaluation...")
        results = self.evaluator.evaluate(
            generation_reports=generation_reports,
            retrieval_ground_truth=retrieval_ground_truth,
            morphology_data=morphology_data
        )
        
        # Generate reports
        logger.info("📝 Generating publication reports...")
        generated_files = self.report_generator.generate_all_reports(
            evaluation_results=results,
            base_filename="full_pipeline_evaluation"
        )
        
        results["generated_files"] = {k: str(v) for k, v in generated_files.items()}
        
        return results


# ==========================================================
# MAIN EXECUTION
# ==========================================================

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    TEST_DATASET_PATH = PROJECT_ROOT / "src" / "evaluation" / "dataset" / "test.json"
    
    # Check if test dataset exists
    if not TEST_DATASET_PATH.exists():
        logger.error(f"❌ Test dataset not found: {TEST_DATASET_PATH}")
        logger.info("💡 Please create test.json first (see Phase 1 instructions)")
        exit(1)
    
    # Initialize and run
    pipeline = FullPipelineIntegration()
    pipeline.initialize()
    
    results = pipeline.run_evaluation(TEST_DATASET_PATH)
    
    print("\n" + "=" * 80)
    print("✅ FULL PIPELINE EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\n📁 Generated Reports:")
    for format_name, file_path in results.get("generated_files", {}).items():
        print(f"   {format_name.upper()}: {file_path}")
    
    print("\n" + "=" * 80)