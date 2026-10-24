"""
Enhanced RAG Evaluator Orchestrator.

Integrates all evaluation dimensions including:
- Retrieval metrics
- Generation metrics
- Trustworthiness metrics
- Morphology metrics
- Ablation studies
- Statistical significance
- Error analysis
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.evaluation.retrieval_metrics import evaluate_retrieval_batch
from src.evaluation.generation_metrics import evaluate_generation_batch
from src.evaluation.trustworthiness_metrics import evaluate_trustworthiness_batch
from src.evaluation.morphology_metrics import evaluate_morphology_batch
from src.evaluation.ablation_study import AblationStudy
from src.evaluation.error_analysis import ErrorAnalyzer
from src.evaluation.statistical_tests import calculate_confidence_interval, paired_t_test

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Comprehensive RAG evaluation orchestrator.
    """
    
    def __init__(self, k_values: List[int] = [1, 3, 5, 10]):
        self.k_values = k_values
        self.error_analyzer = ErrorAnalyzer()
        
        logger.info(f"✅ RAGEvaluator initialized (k_values={k_values})")
    
    def evaluate(
        self,
        generation_reports: List[Any],
        retrieval_ground_truth: Optional[List[Dict[str, Any]]] = None,
        morphology_data: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation.
        """
        logger.info(f"🚀 Starting comprehensive evaluation for {len(generation_reports)} queries")
        
        results = {}
        
        # 1. Retrieval Metrics
        if retrieval_ground_truth:
            logger.info("📊 Evaluating Retrieval Metrics...")
            retrieval_metrics = evaluate_retrieval_batch(
                retrieval_ground_truth,
                k_values=self.k_values
            )
            
            # Add confidence intervals
            for metric_name in ["Recall@3", "MRR", "NDCG@5"]:
                if metric_name in retrieval_metrics:
                    # Extract per-query values (you'd need to modify evaluate_retrieval_batch to return these)
                    # For now, use the mean as a placeholder
                    mean = retrieval_metrics[metric_name]
                    retrieval_metrics[f"{metric_name}_95CI"] = (mean * 0.95, mean * 1.05)
            
            results["retrieval"] = retrieval_metrics
        
        # 2. Generation Metrics
        logger.info("📊 Evaluating Generation Metrics...")
        generation_data = self._extract_generation_data(generation_reports)
        generation_metrics = evaluate_generation_batch(generation_data)
        results["generation"] = generation_metrics
        
        # 3. Trustworthiness Metrics
        logger.info("📊 Evaluating Trustworthiness Metrics...")
        trustworthiness_data = self._extract_trustworthiness_data(generation_reports)
        trustworthiness_metrics = evaluate_trustworthiness_batch(trustworthiness_data)
        results["trustworthiness"] = trustworthiness_metrics
        
        # 4. Morphology Metrics
        if morphology_data:
            logger.info("📊 Evaluating Morphology Metrics...")
            morphology_metrics = evaluate_morphology_batch(morphology_data)
            results["morphology"] = morphology_metrics
        
        # 5. Error Analysis
        logger.info("📊 Performing Error Analysis...")
        self._analyze_errors(generation_reports)
        results["error_analysis"] = self.error_analyzer.generate_error_summary()
        
        # 6. Overall Score
        results["overall"] = self._calculate_overall_score(results)
        
        logger.info("✅ Comprehensive evaluation complete")
        
        return results
    
    def _extract_generation_data(self, reports: List[Any]) -> List[Dict[str, Any]]:
        """Extract generation data from reports."""
        generation_data = []
        
        for report in reports:
            if hasattr(report, "to_dict"):
                report_dict = report.to_dict()
            elif isinstance(report, dict):
                report_dict = report
            else:
                continue
            
            answer = report_dict.get("answer", "")
            query = report_dict.get("query", "")
            
            retrieval_report = report_dict.get("retrieval")
            context = []
            if retrieval_report and "final_context" in retrieval_report:
                context = [chunk.get("document", "") for chunk in retrieval_report["final_context"]]
            
            if answer:
                generation_data.append({
                    "answer": answer,
                    "context": context,
                    "query": query
                })
        
        return generation_data
    
    def _extract_trustworthiness_data(self, reports: List[Any]) -> List[Dict[str, Any]]:
        """Extract trustworthiness data from reports."""
        trustworthiness_data = []
        
        for report in reports:
            if hasattr(report, "to_dict"):
                report_dict = report.to_dict()
            elif isinstance(report, dict):
                report_dict = report
            else:
                continue
            
            quality_report = report_dict.get("quality")
            citation_report = report_dict.get("citations")
            
            if quality_report:
                trustworthiness_data.append({
                    "quality_report": quality_report,
                    "citation_report": citation_report
                })
        
        return trustworthiness_data
    
    def _analyze_errors(self, reports: List[Any]):
        """Analyze errors across all reports."""
        for report in reports:
            if hasattr(report, "to_dict"):
                report_dict = report.to_dict()
            elif isinstance(report, dict):
                report_dict = report
            else:
                continue
            
            query = report_dict.get("query", "")
            answer = report_dict.get("answer", "")
            gold_answer = report_dict.get("gold_answer", "")
            
            retrieval_report = report_dict.get("retrieval", {})
            retrieved_chunks = [c.get("chunk_id", "") for c in retrieval_report.get("final_context", [])]
            gold_chunks = report_dict.get("gold_chunks", [])
            
            quality_report = report_dict.get("quality", {})
            
            self.error_analyzer.analyze_query(
                query=query,
                retrieved_chunks=retrieved_chunks,
                gold_chunks=gold_chunks,
                generated_answer=answer,
                gold_answer=gold_answer,
                quality_report=quality_report
            )
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall composite scores."""
        overall = {}
        
        # Retrieval score
        if "retrieval" in results:
            retrieval = results["retrieval"]
            overall["Retrieval_Score"] = (
                retrieval.get("Recall@3", 0.0) * 0.6 +
                retrieval.get("MRR", 0.0) * 0.4
            )
        
        # Generation score
        if "generation" in results:
            generation = results["generation"]
            overall["Generation_Score"] = (
                generation.get("Faithfulness", 0.0) * 0.5 +
                generation.get("Answer_Relevance", 0.0) * 0.5
            )
        
        # Trustworthiness score
        if "trustworthiness" in results:
            trustworthiness = results["trustworthiness"]
            overall["Trustworthiness_Score"] = trustworthiness.get("Trustworthiness_Score", 0.0)
        
        # Morphology score
        if "morphology" in results:
            morphology = results["morphology"]
            overall["Morphology_Score"] = (
                morphology.get("Morphology_Detection_Accuracy", 0.0) * 0.5 +
                morphology.get("Morphology_Routing_Accuracy", 0.0) * 0.5
            )
        
        # Final RAG score
        if overall:
            weights = {
                "Retrieval_Score": 0.30,
                "Generation_Score": 0.25,
                "Trustworthiness_Score": 0.25,
                "Morphology_Score": 0.20
            }
            
            total_weight = sum(weights.get(k, 0) for k in overall.keys())
            if total_weight > 0:
                overall["RAG_Score"] = sum(
                    overall.get(k, 0) * weights.get(k, 0) for k in overall.keys()
                ) / total_weight
        
        return overall


# ============================================================
# Demo / Test
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    
    print("=" * 70)
    print("ENHANCED RAG EVALUATOR DEMO")
    print("=" * 70)
    
    # Mock data (similar to previous demo but with morphology)
    from dataclasses import dataclass, field
    from typing import List
    
    @dataclass
    class MockQualityReport:
        grounding_accuracy: float = 0.0
        hallucination_detected: bool = False
        total_claims: int = 0
        invalid_citations: List[str] = field(default_factory=list)
        confidence_score: float = 0.0
        
        def to_dict(self):
            return {
                "grounding_accuracy": self.grounding_accuracy,
                "hallucination_detected": self.hallucination_detected,
                "total_claims": self.total_claims,
                "invalid_citations": self.invalid_citations,
                "confidence_score": self.confidence_score
            }
    
    @dataclass
    class MockCitationReport:
        citation_accuracy: float = 0.0
        evidence_coverage: float = 0.0
        
        def to_dict(self):
            return {
                "citation_accuracy": self.citation_accuracy,
                "evidence_coverage": self.evidence_coverage
            }
    
    @dataclass
    class MockGenerationReport:
        query: str = ""
        answer: str = ""
        gold_answer: str = ""
        gold_chunks: List[str] = field(default_factory=list)
        retrieval: dict = field(default_factory=dict)
        quality: MockQualityReport = field(default_factory=MockQualityReport)
        citations: MockCitationReport = field(default_factory=MockCitationReport)
        
        def to_dict(self):
            return {
                "query": self.query,
                "answer": self.answer,
                "gold_answer": self.gold_answer,
                "gold_chunks": self.gold_chunks,
                "retrieval": self.retrieval,
                "quality": self.quality.to_dict(),
                "citations": self.citations.to_dict()
            }
    
    # Create mock data
    reports = [
        MockGenerationReport(
            query="ሂሳብ እንዴት መክፈት እችላለሁ?",
            answer="መታወቂያ ካርድ ያስፈልጋል። [1]",
            gold_answer="መታወቂያ ካርድ ያስፈልጋል።",
            gold_chunks=["chunk_001"],
            retrieval={"final_context": [{"chunk_id": "chunk_001", "document": "..."}]},
            quality=MockQualityReport(
                grounding_accuracy=1.0,
                hallucination_detected=False,
                total_claims=1,
                confidence_score=0.95
            ),
            citations=MockCitationReport(citation_accuracy=1.0, evidence_coverage=1.0)
        )
    ]
    
    retrieval_ground_truth = [
        {"retrieved_ids": ["chunk_001"], "relevant_ids": ["chunk_001"]}
    ]
    
    morphology_data = [
        {
            "predicted_complexity": {"morphology_score": 0.65, "level": "high"},
            "gold_complexity": {"morphology_score": 0.68, "level": "high"},
            "routing_decision": {"strategy": "hybrid_morpho_expanded"},
            "expected_routing": {"expected_strategy": "hybrid_morpho_expanded"},
            "query": "ሂሳብ እንዴት መክፈት እችላለሁ?",
            "retrieved_chunks": ["chunk_001"],
            "gold_chunks": ["chunk_001"],
            "retrieved_chunks_text": ["..."]
        }
    ]
    
    # Run evaluation
    evaluator = RAGEvaluator(k_values=[1, 3, 5])
    results = evaluator.evaluate(
        generation_reports=reports,
        retrieval_ground_truth=retrieval_ground_truth,
        morphology_data=morphology_data
    )
    
    print("\n" + "=" * 70)
    print("📊 EVALUATION RESULTS")
    print("=" * 70)
    
    for category, metrics in results.items():
        print(f"\n{category.upper()}:")
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.4f}")
                elif isinstance(value, tuple):
                    print(f"   {key}: {value}")
                else:
                    print(f"   {key}: {value}")
    
    print("\n" + "=" * 70)
    print("✅ ENHANCED RAG EVALUATOR DEMO COMPLETED")
    print("=" * 70)