"""
Error Analysis Module for Qualitative Evaluation.

Identifies and categorizes failure cases for deeper understanding.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ErrorAnalyzer:
    """
    Analyzes failure cases and categorizes error types.
    """
    
    ERROR_CATEGORIES = {
        "retrieval_failure": "Relevant chunks not retrieved",
        "hallucination": "Generated unsupported claims",
        "citation_error": "Invalid or missing citations",
        "morphology_failure": "Failed to handle complex morphology",
        "intent_misclassification": "Wrong intent detected",
        "low_confidence": "System uncertain about answer"
    }
    
    def __init__(self):
        self.errors = []
        
        logger.info("✅ ErrorAnalyzer initialized")
    
    def analyze_query(
        self,
        query: str,
        retrieved_chunks: List[str],
        gold_chunks: List[str],
        generated_answer: str,
        gold_answer: str,
        quality_report: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze a single query for errors.
        
        Returns:
            Dict with error categories and details
        """
        errors = []
        
        # Check retrieval failure
        if not set(retrieved_chunks) & set(gold_chunks):
            errors.append({
                "category": "retrieval_failure",
                "severity": "high",
                "details": f"No gold chunks retrieved. Gold: {gold_chunks}, Retrieved: {retrieved_chunks}"
            })
        
        # Check hallucination
        if quality_report.get("hallucination_detected", False):
            errors.append({
                "category": "hallucination",
                "severity": "high",
                "details": "Generated answer contains unsupported claims"
            })
        
        # Check citation errors
        citation_accuracy = quality_report.get("citation_accuracy", 1.0)
        if citation_accuracy < 0.8:
            errors.append({
                "category": "citation_error",
                "severity": "medium",
                "details": f"Citation accuracy: {citation_accuracy:.2%}"
            })
        
        # Check low confidence
        confidence = quality_report.get("confidence_score", 1.0)
        if confidence < 0.5:
            errors.append({
                "category": "low_confidence",
                "severity": "low",
                "details": f"Low confidence: {confidence:.2%}"
            })
        
        error_record = {
            "query": query,
            "errors": errors,
            "error_count": len(errors)
        }
        
        self.errors.append(error_record)
        
        return error_record
    
    def generate_error_summary(self) -> Dict[str, int]:
        """
        Generate summary of error categories.
        
        Returns:
            Dict mapping error category to count
        """
        summary = {category: 0 for category in self.ERROR_CATEGORIES.keys()}
        
        for error_record in self.errors:
            for error in error_record.get("errors", []):
                category = error.get("category")
                if category in summary:
                    summary[category] += 1
        
        return summary
    
    def generate_error_report(self) -> str:
        """
        Generate human-readable error report.
        
        Returns:
            Formatted error report string
        """
        summary = self.generate_error_summary()
        
        lines = []
        lines.append("# Error Analysis Report\n")
        lines.append("## Error Category Distribution\n")
        lines.append("| Category | Count | Description |")
        lines.append("|----------|-------|-------------|")
        
        for category, count in summary.items():
            description = self.ERROR_CATEGORIES.get(category, "Unknown")
            lines.append(f"| {category} | {count} | {description} |")
        
        lines.append(f"\n**Total Queries with Errors:** {len(self.errors)}")
        
        return "\n".join(lines)


# ============================================================
# Demo / Test
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    
    print("=" * 70)
    print("ERROR ANALYSIS DEMO")
    print("=" * 70)
    
    analyzer = ErrorAnalyzer()
    
    # Analyze some queries
    analyzer.analyze_query(
        query="የCBE የቁጠባ ሂሳብ እንዴት መክፈት እችላለሁ?",
        retrieved_chunks=["chunk_001", "chunk_002"],
        gold_chunks=["chunk_001", "chunk_002"],
        generated_answer="መታወቂያ ካርድ ያስፈልጋል። [1]",
        gold_answer="መታወቂያ ካርድ ያስፈልጋል።",
        quality_report={"hallucination_detected": False, "citation_accuracy": 1.0, "confidence_score": 0.95}
    )
    
    analyzer.analyze_query(
        query="የብድር ወለድ ስንት ነው?",
        retrieved_chunks=["chunk_003"],
        gold_chunks=["chunk_004"],  # Wrong chunk retrieved
        generated_answer="ወለድ 15% ነው። [99]",  # Hallucination + bad citation
        gold_answer="ወለድ 12% ነው።",
        quality_report={"hallucination_detected": True, "citation_accuracy": 0.0, "confidence_score": 0.3}
    )
    
    # Generate report
    report = analyzer.generate_error_report()
    
    print("\n" + report)
    
    print("\n" + "=" * 70)
    print("✅ ERROR ANALYSIS DEMO COMPLETED")
    print("=" * 70)