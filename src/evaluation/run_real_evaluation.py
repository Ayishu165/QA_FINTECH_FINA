import json
import logging
from pathlib import Path
from src.evaluation.evaluator import RAGEvaluator
from src.evaluation.report_generator import EvaluationReportGenerator

# 1. Load your real test dataset
with open("src/evaluation/dataset/test.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

# 2. Initialize Evaluator
evaluator = RAGEvaluator(k_values=[1, 3, 5, 10])

# 3. Prepare data for the evaluator
# (You will need to run your actual RetrievalPipeline and GenerationPipeline 
# on each query in test_data to get the 'generation_reports' and 'retrieval_ground_truth')

# For now, this is the structure the evaluator expects:
generation_reports = [...] # Output from your GenerationPipeline
retrieval_ground_truth = [...] # List of {"retrieved_ids": [...], "relevant_ids": [...]}
morphology_data = [...] # List of morphology predictions vs gold

# 4. Run Evaluation
results = evaluator.evaluate(
    generation_reports=generation_reports,
    retrieval_ground_truth=retrieval_ground_truth,
    morphology_data=morphology_data
)

# 5. Generate Publication Reports
report_dir = Path("src/evaluation/reports/final_results")
generator = EvaluationReportGenerator(output_dir=report_dir)
generator.generate_all_reports(results, base_filename="thesis_evaluation")

print("✅ Real evaluation complete! Check src/evaluation/reports/final_results/")