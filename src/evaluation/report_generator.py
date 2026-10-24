"""
Enhanced Evaluation Report Generator.

Generates publication-ready reports in:
- JSON: Full structured data
- CSV: Tabular format
- Markdown: Human-readable
- LaTeX: Publication tables
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class EvaluationReportGenerator:
    """
    Enhanced report generator with LaTeX support.
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✅ EvaluationReportGenerator initialized (output_dir={output_dir})")
    
    def generate_all_reports(
        self,
        evaluation_results: Dict[str, Any],
        base_filename: str = "evaluation_report"
    ) -> Dict[str, Path]:
        """Generate all report formats."""
        logger.info("📝 Generating evaluation reports...")
        
        generated_files = {}
        
        # JSON
        json_path = self._generate_json_report(evaluation_results, base_filename)
        generated_files["json"] = json_path
        
        # CSV
        csv_path = self._generate_csv_report(evaluation_results, base_filename)
        generated_files["csv"] = csv_path
        
        # Markdown
        md_path = self._generate_markdown_report(evaluation_results, base_filename)
        generated_files["markdown"] = md_path
        
        # LaTeX
        latex_path = self._generate_latex_report(evaluation_results, base_filename)
        generated_files["latex"] = latex_path
        
        logger.info("✅ All reports generated successfully")
        
        return generated_files
    
    def _generate_json_report(self, results: Dict[str, Any], base_filename: str) -> Path:
        """Generate JSON report."""
        output_path = self.output_dir / f"{base_filename}.json"
        
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"   ✅ JSON report: {output_path}")
        return output_path
    
    def _generate_csv_report(self, results: Dict[str, Any], base_filename: str) -> Path:
        """Generate CSV report."""
        output_path = self.output_dir / f"{base_filename}.csv"
        
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Category", "Metric", "Value", "Std", "Min", "Max"])
            
            for category, metrics in results.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            std = metrics.get(f"{metric_name}_std", "")
                            min_val = metrics.get(f"{metric_name}_min", "")
                            max_val = metrics.get(f"{metric_name}_max", "")
                            
                            writer.writerow([
                                category,
                                metric_name,
                                f"{value:.4f}" if isinstance(value, float) else value,
                                f"{std:.4f}" if isinstance(std, float) else std,
                                f"{min_val:.4f}" if isinstance(min_val, float) else min_val,
                                f"{max_val:.4f}" if isinstance(max_val, float) else max_val,
                            ])
        
        logger.info(f"   ✅ CSV report: {output_path}")
        return output_path
    
    def _generate_markdown_report(self, results: Dict[str, Any], base_filename: str) -> Path:
        """Generate Markdown report."""
        output_path = self.output_dir / f"{base_filename}.md"
        
        with output_path.open("w", encoding="utf-8") as f:
            f.write("# RAG Evaluation Report\n\n")
            f.write(f"**Generated:** {self._get_timestamp()}\n\n")
            
            # Overall Score
            if "overall" in results:
                f.write("## Overall Performance\n\n")
                f.write("| Metric | Score |\n")
                f.write("|--------|-------|\n")
                for metric, value in results["overall"].items():
                    if isinstance(value, float):
                        f.write(f"| {metric} | {value:.2%} |\n")
                f.write("\n")
            
            # Retrieval Metrics
            if "retrieval" in results:
                f.write("## Retrieval Metrics\n\n")
                f.write("| Metric | Mean | Std | Min | Max |\n")
                f.write("|--------|------|-----|-----|-----|\n")
                self._write_metrics_table(f, results["retrieval"])
                f.write("\n")
            
            # Generation Metrics
            if "generation" in results:
                f.write("## Generation Metrics\n\n")
                f.write("| Metric | Mean | Std | Min | Max |\n")
                f.write("|--------|------|-----|-----|-----|\n")
                self._write_metrics_table(f, results["generation"])
                f.write("\n")
            
            # Trustworthiness Metrics
            if "trustworthiness" in results:
                f.write("## Trustworthiness Metrics\n\n")
                f.write("| Metric | Mean | Std | Min | Max |\n")
                f.write("|--------|------|-----|-----|-----|\n")
                self._write_metrics_table(f, results["trustworthiness"])
                f.write("\n")
            
            # Morphology Metrics
            if "morphology" in results:
                f.write("## Morphology Metrics\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                for metric, value in results["morphology"].items():
                    if isinstance(value, float) and not metric.endswith(("_std", "_min", "_max")):
                        f.write(f"| {metric} | {value:.4f} |\n")
                f.write("\n")
            
            # Error Analysis
            if "error_analysis" in results:
                f.write("## Error Analysis\n\n")
                f.write("| Error Category | Count |\n")
                f.write("|----------------|-------|\n")
                for category, count in results["error_analysis"].items():
                    f.write(f"| {category} | {count} |\n")
                f.write("\n")
            
            # Summary
            f.write("## Summary\n\n")
            f.write(f"- **Total Queries Evaluated:** {results.get('generation', {}).get('num_queries', 'N/A')}\n")
            if "overall" in results and "RAG_Score" in results["overall"]:
                f.write(f"- **Overall RAG Score:** {results['overall']['RAG_Score']:.2%}\n")
        
        logger.info(f"   ✅ Markdown report: {output_path}")
        return output_path
    
    def _generate_latex_report(self, results: Dict[str, Any], base_filename: str) -> Path:
        """Generate LaTeX tables for publication."""
        output_path = self.output_dir / f"{base_filename}.tex"
        
        with output_path.open("w", encoding="utf-8") as f:
            f.write("% Auto-generated LaTeX tables for RAG evaluation\n\n")
            
            # Table 1: Retrieval Performance
            if "retrieval" in results:
                f.write("\\begin{table}[ht]\n")
                f.write("\\centering\n")
                f.write("\\caption{Retrieval Performance Comparison}\n")
                f.write("\\label{tab:retrieval}\n")
                f.write("\\begin{tabular}{lcccc}\n")
                f.write("\\hline\n")
                f.write("\\textbf{Metric} & \\textbf{Mean} & \\textbf{Std} & \\textbf{Min} & \\textbf{Max} \\\\\n")
                f.write("\\hline\n")
                
                retrieval = results["retrieval"]
                for metric in ["Recall@3", "Precision@3", "MRR", "NDCG@5", "MAP"]:
                    if metric in retrieval:
                        mean = retrieval[metric]
                        std = retrieval.get(f"{metric}_std", 0.0)
                        min_val = retrieval.get(f"{metric}_min", 0.0)
                        max_val = retrieval.get(f"{metric}_max", 0.0)
                        
                        f.write(f"{metric} & {mean:.4f} & {std:.4f} & {min_val:.4f} & {max_val:.4f} \\\\\n")
                
                f.write("\\hline\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n\n")
            
            # Table 2: Generation Quality
            if "generation" in results:
                f.write("\\begin{table}[ht]\n")
                f.write("\\centering\n")
                f.write("\\caption{Generation Quality Metrics}\n")
                f.write("\\label{tab:generation}\n")
                f.write("\\begin{tabular}{lcc}\n")
                f.write("\\hline\n")
                f.write("\\textbf{Metric} & \\textbf{Mean} & \\textbf{Std} \\\\\n")
                f.write("\\hline\n")
                
                generation = results["generation"]
                for metric in ["Faithfulness", "Answer_Relevance", "Context_Relevance", "Completeness"]:
                    if metric in generation:
                        mean = generation[metric]
                        std = generation.get(f"{metric}_std", 0.0)
                        f.write(f"{metric.replace('_', '\\_')} & {mean:.4f} & {std:.4f} \\\\\n")
                
                f.write("\\hline\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n\n")
            
            # Table 3: Trustworthiness
            if "trustworthiness" in results:
                f.write("\\begin{table}[ht]\n")
                f.write("\\centering\n")
                f.write("\\caption{Trustworthiness Metrics}\n")
                f.write("\\label{tab:trustworthiness}\n")
                f.write("\\begin{tabular}{lcc}\n")
                f.write("\\hline\n")
                f.write("\\textbf{Metric} & \\textbf{Mean} & \\textbf{Std} \\\\\n")
                f.write("\\hline\n")
                
                trust = results["trustworthiness"]
                for metric in ["Grounding_Accuracy", "Citation_Accuracy", "Hallucination_Rate", "Evidence_Coverage"]:
                    if metric in trust:
                        mean = trust[metric]
                        std = trust.get(f"{metric}_std", 0.0)
                        f.write(f"{metric.replace('_', '\\_')} & {mean:.4f} & {std:.4f} \\\\\n")
                
                f.write("\\hline\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n\n")
            
            # Table 4: Morphology Metrics
            if "morphology" in results:
                f.write("\\begin{table}[ht]\n")
                f.write("\\centering\n")
                f.write("\\caption{Morphology-Aware Metrics}\n")
                f.write("\\label{tab:morphology}\n")
                f.write("\\begin{tabular}{lc}\n")
                f.write("\\hline\n")
                f.write("\\textbf{Metric} & \\textbf{Value} \\\\\n")
                f.write("\\hline\n")
                
                morphology = results["morphology"]
                for metric in ["Morphology_Detection_Accuracy", "Morphology_Routing_Accuracy", "Affix_Handling_Score", "Compound_Word_Preservation"]:
                    if metric in morphology:
                        value = morphology[metric]
                        f.write(f"{metric.replace('_', '\\_')} & {value:.4f} \\\\\n")
                
                f.write("\\hline\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n\n")
        
        logger.info(f"   ✅ LaTeX report: {output_path}")
        return output_path
    
    def _write_metrics_table(self, f, metrics: Dict[str, Any]):
        """Write metrics table rows."""
        for metric_name, value in metrics.items():
            if isinstance(value, float) and not metric_name.endswith(("_std", "_min", "_max")):
                std = metrics.get(f"{metric_name}_std", 0.0)
                min_val = metrics.get(f"{metric_name}_min", 0.0)
                max_val = metrics.get(f"{metric_name}_max", 0.0)
                
                f.write(
                    f"| {metric_name} | {value:.4f} | {std:.4f} | {min_val:.4f} | {max_val:.4f} |\n"
                )
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# Demo / Test
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    
    print("=" * 70)
    print("ENHANCED REPORT GENERATOR DEMO")
    print("=" * 70)
    
    # Mock results
    mock_results = {
        "retrieval": {
            "Recall@3": 0.85, "Recall@3_std": 0.08, "Recall@3_min": 0.70, "Recall@3_max": 0.95,
            "MRR": 0.78, "MRR_std": 0.12, "MRR_min": 0.55, "MRR_max": 0.92,
            "NDCG@5": 0.82, "NDCG@5_std": 0.09, "NDCG@5_min": 0.68, "NDCG@5_max": 0.93,
            "num_queries": 50
        },
        "generation": {
            "Faithfulness": 0.92, "Faithfulness_std": 0.05,
            "Answer_Relevance": 0.88, "Answer_Relevance_std": 0.07,
            "num_queries": 50
        },
        "trustworthiness": {
            "Grounding_Accuracy": 0.95, "Grounding_Accuracy_std": 0.03,
            "Citation_Accuracy": 0.98, "Citation_Accuracy_std": 0.02,
            "Hallucination_Rate": 0.05, "Hallucination_Rate_std": 0.02,
            "num_queries": 50
        },
        "morphology": {
            "Morphology_Detection_Accuracy": 0.89,
            "Morphology_Routing_Accuracy": 0.92,
            "Affix_Handling_Score": 0.85,
            "num_queries": 50
        },
        "overall": {
            "Retrieval_Score": 0.83,
            "Generation_Score": 0.90,
            "Trustworthiness_Score": 0.93,
            "Morphology_Score": 0.90,
            "RAG_Score": 0.89
        },
        "error_analysis": {
            "retrieval_failure": 3,
            "hallucination": 2,
            "citation_error": 1,
            "morphology_failure": 0
        }
    }
    
    # Generate reports
    output_dir = Path(__file__).parent / "reports"
    generator = EvaluationReportGenerator(output_dir=output_dir)
    
    generated_files = generator.generate_all_reports(
        evaluation_results=mock_results,
        base_filename="enhanced_evaluation"
    )
    
    print("\n" + "=" * 70)
    print("📁 GENERATED FILES")
    print("=" * 70)
    for format_name, file_path in generated_files.items():
        print(f"   {format_name.upper()}: {file_path}")
    
    print("\n" + "=" * 70)
    print("✅ ENHANCED REPORT GENERATOR DEMO COMPLETED")
    print("=" * 70)