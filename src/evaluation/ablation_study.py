"""
Ablation Study Framework for Amharic FinTech RAG.

Compares different system configurations to prove each component's contribution.

Configurations to compare:
1. Baseline BM25 Only
2. Baseline Dense Only
3. Hybrid RRF (Dense + Sparse)
4. Hybrid + Morphology Reranking
5. Full Proposed Method (Adaptive + Morphology + Trust)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

import numpy as np

from src.evaluation.retrieval_metrics import evaluate_retrieval_batch
from src.evaluation.generation_metrics import evaluate_generation_batch
from src.evaluation.trustworthiness_metrics import evaluate_trustworthiness_batch
from src.evaluation.morphology_metrics import evaluate_morphology_batch

logger = logging.getLogger(__name__)


class AblationStudy:
    """
    Runs ablation experiments to validate each component's contribution.
    """
    
    def __init__(self, dataset: List[Dict[str, Any]]):
        """
        Initialize ablation study.
        
        Args:
            dataset: List of test queries with gold annotations
        """
        self.dataset = dataset
        self.results = {}
        
        logger.info(f"✅ AblationStudy initialized with {len(dataset)} queries")
    
    def run_configuration(
        self,
        config_name: str,
        retriever_func: Callable,
        generator_func: Callable = None,
        quality_checker_func: Callable = None
    ) -> Dict[str, Any]:
        """
        Run a specific configuration and collect metrics.
        
        Args:
            config_name: Name of the configuration (e.g., "Baseline BM25")
            retriever_func: Function that takes query and returns retrieved chunks
            generator_func: Optional function that generates answers
            quality_checker_func: Optional function that checks quality
            
        Returns:
            Dict with all metrics for this configuration
        """
        logger.info(f"🚀 Running configuration: {config_name}")
        
        # Prepare data structures
        retrieval_results = []
        generation_results = []
        trustworthiness_results = []
        morphology_results = []
        
        for item in self.dataset:
            query = item.get("query", "")
            gold_chunks = item.get("gold_chunks", [])
            gold_answer = item.get("gold_answer", "")
            gold_complexity = item.get("complexity", {})
            
            # Step 1: Retrieval
            retrieved = retriever_func(query)
            retrieved_ids = [c.get("chunk_id", "") for c in retrieved]
            retrieved_text = [c.get("document", "") for c in retrieved]
            
            retrieval_results.append({
                "retrieved_ids": retrieved_ids,
                "relevant_ids": gold_chunks
            })
            
            # Step 2: Generation (if provided)
            if generator_func:
                answer = generator_func(query, retrieved)
                
                generation_results.append({
                    "answer": answer,
                    "context": retrieved_text,
                    "query": query,
                    "gold_answer": gold_answer
                })
                
                # Step 3: Quality Checking (if provided)
                if quality_checker_func:
                    quality_report = quality_checker_func(answer, retrieved, query)
                    
                    trustworthiness_results.append({
                        "quality_report": quality_report,
                        "citation_report": quality_report.get("citation_report")
                    })
            
            # Step 4: Morphology evaluation
            morphology_results.append({
                "predicted_complexity": item.get("predicted_complexity", {}),
                "gold_complexity": gold_complexity,
                "routing_decision": item.get("routing_decision", {}),
                "expected_routing": item.get("expected_routing", {}),
                "query": query,
                "retrieved_chunks": retrieved_ids,
                "gold_chunks": gold_chunks,
                "retrieved_chunks_text": retrieved_text
            })
        
        # Calculate metrics
        config_metrics = {
            "retrieval": evaluate_retrieval_batch(retrieval_results, k_values=[1, 3, 5, 10]),
            "morphology": evaluate_morphology_batch(morphology_results)
        }
        
        if generation_results:
            config_metrics["generation"] = evaluate_generation_batch(generation_results)
        
        if trustworthiness_results:
            config_metrics["trustworthiness"] = evaluate_trustworthiness_batch(trustworthiness_results)
        
        self.results[config_name] = config_metrics
        
        logger.info(f"✅ Configuration {config_name} complete")
        
        return config_metrics
    
    def compare_configurations(self) -> Dict[str, Any]:
        """
        Compare all configurations and generate comparison table.
        
        Returns:
            Dict with comparison metrics
        """
        if not self.results:
            logger.warning("No configurations have been run yet")
            return {}
        
        comparison = {}
        
        # Get all unique metric names
        all_metrics = set()
        for config_metrics in self.results.values():
            for dimension_metrics in config_metrics.values():
                if isinstance(dimension_metrics, dict):
                    all_metrics.update(dimension_metrics.keys())
        
        # Build comparison table
        for metric in sorted(all_metrics):
            if metric.endswith(("_std", "_min", "_max")) or metric == "num_queries":
                continue
            
            comparison[metric] = {}
            
            for config_name, config_metrics in self.results.items():
                value = None
                
                # Search for metric in all dimensions
                for dimension_metrics in config_metrics.values():
                    if isinstance(dimension_metrics, dict) and metric in dimension_metrics:
                        value = dimension_metrics[metric]
                        break
                
                comparison[metric][config_name] = value
        
        return comparison
    
    def generate_ablation_table(self) -> str:
        """
        Generate publication-ready ablation study table.
        
        Returns:
            Markdown-formatted table
        """
        comparison = self.compare_configurations()
        
        if not comparison:
            return "No data available"
        
        lines = []
        lines.append("# Ablation Study Results\n")
        lines.append("| Metric | " + " | ".join(self.results.keys()) + " |")
        lines.append("|" + "|".join(["---"] * (len(self.results) + 1)) + "|")
        
        for metric, config_values in comparison.items():
            values = []
            for config_name in self.results.keys():
                value = config_values.get(config_name)
                if isinstance(value, float):
                    values.append(f"{value:.4f}")
                elif value is not None:
                    values.append(str(value))
                else:
                    values.append("-")
            
            lines.append(f"| {metric} | " + " | ".join(values) + " |")
        
        return "\n".join(lines)


# ============================================================
# Demo / Test
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    
    print("=" * 70)
    print("ABLATION STUDY DEMO")
    print("=" * 70)
    
    # Mock dataset
    dataset = [
        {
            "query": "የCBE የቁጠባ ሂሳብ እንዴት መክፈት እችላለሁ?",
            "gold_chunks": ["chunk_001", "chunk_002"],
            "gold_answer": "መታወቂያ ካርድ ያስፈልጋል።",
            "complexity": {"morphology_score": 0.68, "level": "high"},
            "predicted_complexity": {"morphology_score": 0.65, "level": "high"},
            "routing_decision": {"strategy": "hybrid_morpho_expanded"},
            "expected_routing": {"expected_strategy": "hybrid_morpho_expanded"}
        },
        {
            "query": "የብድር ወለድ ስንት ነው?",
            "gold_chunks": ["chunk_003"],
            "gold_answer": "12% ነው።",
            "complexity": {"morphology_score": 0.45, "level": "medium"},
            "predicted_complexity": {"morphology_score": 0.42, "level": "medium"},
            "routing_decision": {"strategy": "hybrid"},
            "expected_routing": {"expected_strategy": "hybrid"}
        }
    ]
    
    # Mock retriever functions
    def mock_bm25_retriever(query: str) -> List[Dict[str, Any]]:
        return [{"chunk_id": "chunk_001", "document": "..."}]
    
    def mock_dense_retriever(query: str) -> List[Dict[str, Any]]:
        return [{"chunk_id": "chunk_001", "document": "..."}, {"chunk_id": "chunk_002", "document": "..."}]
    
    def mock_hybrid_retriever(query: str) -> List[Dict[str, Any]]:
        return [{"chunk_id": "chunk_001", "document": "..."}, {"chunk_id": "chunk_002", "document": "..."}]
    
    def mock_proposed_retriever(query: str) -> List[Dict[str, Any]]:
        return [{"chunk_id": "chunk_001", "document": "..."}, {"chunk_id": "chunk_002", "document": "..."}]
    
    # Run ablation study
    study = AblationStudy(dataset)
    
    study.run_configuration("Baseline BM25", mock_bm25_retriever)
    study.run_configuration("Baseline Dense", mock_dense_retriever)
    study.run_configuration("Hybrid RRF", mock_hybrid_retriever)
    study.run_configuration("Proposed Method", mock_proposed_retriever)
    
    # Generate table
    table = study.generate_ablation_table()
    
    print("\n" + table)
    
    print("\n" + "=" * 70)
    print("✅ ABLATION STUDY DEMO COMPLETED")
    print("=" * 70)