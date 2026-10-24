#!/usr/bin/env python
"""
app.py

Unified Entry Point for Amharic FinTech RAG System.

Usage:
    python app.py interactive --query "የCBE ካርድ እንዴት ማግኘት እችላለሁ?"
    python app.py evaluate --dataset data/test_queries.json --output outputs/evaluation/run1
    python app.py serve --port 8000
    python app.py benchmark --queries 100 --output outputs/benchmark

Modes:
    interactive  : Ask a single question interactively
    evaluate     : Run evaluation on a test dataset
    serve        : Start a web API server (if implemented)
    benchmark    : Run benchmark on multiple test queries
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ============================================================
# Configure logging early
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# Imports (lazy to keep startup fast)
# ============================================================

def get_pipeline():
    """Lazy import the generation pipeline."""
    from src.generation.generation_pipeline import GenerationPipeline
    return GenerationPipeline

def get_evaluator():
    """Lazy import the evaluator."""
    from src.evaluation.evaluator import Evaluator
    return Evaluator

def get_report_generator():
    """Lazy import the report generator."""
    from src.evaluation.report_generator import generate_full_report
    return generate_full_report

def load_config() -> Dict[str, Any]:
    """Load configuration from config.py or environment."""
    try:
        from config import config
        return config
    except ImportError:
        logger.warning("config.py not found. Using defaults.")
        return {
            "max_chunks": 5,
            "llm_backend": "mock",
            "llm_model": "mock-model",
            "embedding_model": "intfloat/multilingual-e5-large",
            "device": "cpu",
            "use_real_retrieval": False,
            "use_real_llm": False,
            "output_dir": "outputs",
            "log_level": "INFO",
        }


# ============================================================
# Interactive Mode
# ============================================================

def interactive_mode(
    query: str,
    max_chunks: int = 5,
    output_json: Optional[Path] = None,
    verbose: bool = False,
) -> None:
    """
    Run the pipeline interactively for a single query.

    Args:
        query: The user's question.
        max_chunks: Maximum number of chunks to retrieve.
        output_json: Optional path to save the result as JSON.
        verbose: Print full prompt and detailed logs.
    """
    logger.info(f"🔍 Interactive Mode: {query[:60]}...")

    # Load pipeline
    Pipeline = get_pipeline()
    pipeline = Pipeline.from_config(max_chunks=max_chunks)

    # Run generation
    start_time = time.time()
    report = pipeline.generate(query=query)
    elapsed_ms = (time.time() - start_time) * 1000

    # Display results
    print("\n" + "=" * 80)
    print("📊 GENERATION REPORT")
    print("=" * 80)

    print(f"\n🎯 Query: {report.query}")
    print(f"🎭 Intent: {report.intent}")
    print(f"🤖 LLM: {report.llm_backend}")
    print(f"📦 Chunks Used: {report.chunks_used}")
    print(f"✅ Success: {report.success}")
    print(f"⏱️  Time: {elapsed_ms:.1f}ms")

    print(f"\n📝 Answer:")
    print("-" * 80)
    print(report.answer)
    print("-" * 80)

    print(f"\n🛡️ Quality:")
    print(f"   Confidence: {report.final_confidence:.1%}")
    print(f"   Quality Level: {report.quality_level}")
    print(f"   Grounding: {report.grounding_accuracy:.1%}")
    print(f"   Citations: {report.citation_accuracy:.1%}")
    print(f"   Hallucination: {report.is_hallucinated}")

    if report.errors:
        print(f"\n⚠️ Errors: {report.errors}")

    if verbose:
        print(f"\n📄 Full Prompt:")
        print("-" * 80)
        print(report.prompt_used)
        print("-" * 80)

    # Save JSON output
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Report saved to: {output_json}")


# ============================================================
# Evaluation Mode
# ============================================================

def evaluate_mode(
    dataset_path: Path,
    output_dir: Path,
    max_chunks: int = 5,
    show_progress: bool = True,
) -> None:
    """
    Run evaluation on a test dataset.

    Args:
        dataset_path: Path to JSON file with test queries.
        output_dir: Directory to save evaluation results.
        max_chunks: Maximum chunks per query.
        show_progress: Show progress bar.
    """
    logger.info(f"📊 Evaluation Mode: dataset={dataset_path}")

    # Load dataset
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    with dataset_path.open("r", encoding="utf-8") as f:
        test_data = json.load(f)

    if isinstance(test_data, dict) and "queries" in test_data:
        test_data = test_data["queries"]

    logger.info(f"📄 Loaded {len(test_data)} test queries")

    # Load pipeline and evaluator
    Pipeline = get_pipeline()
    Evaluator = get_evaluator()

    pipeline = Pipeline.from_config(max_chunks=max_chunks)
    evaluator = Evaluator(
        generation_pipeline=pipeline,
        test_data=test_data,
        output_dir=output_dir,
    )

    # Run evaluation
    results = evaluator.run(show_progress=show_progress)

    # Generate reports
    ReportGenerator = get_report_generator()
    aggregate = evaluator._compute_aggregate()
    ReportGenerator(results, output_dir, aggregate)

    # Print summary
    print("\n" + "=" * 80)
    print("📊 EVALUATION SUMMARY")
    print("=" * 80)
    print(f"✅ Completed: {len(results)} queries")
    print(f"📁 Reports saved to: {output_dir}")
    print("=" * 80)


# ============================================================
# Benchmark Mode
# ============================================================

def benchmark_mode(
    num_queries: int,
    output_dir: Path,
    max_chunks: int = 5,
) -> None:
    """
    Run benchmark on synthetic/generated queries.

    Args:
        num_queries: Number of queries to generate/run.
        output_dir: Directory to save results.
        max_chunks: Maximum chunks per query.
    """
    logger.info(f"⏱️ Benchmark Mode: {num_queries} queries")

    # Generate or load benchmark queries
    # For now, use a set of predefined query templates
    query_templates = [
        "የCBE የቁጠባ ሂሳብ እንዴት መክፈት እችላለሁ?",
        "የብድር ወለድ መጠን ስንት ነው?",
        "የባንክ ሂሳብ ምንድን ነው?",
        "የCBE ካርድ እንዴት ማግኘት እችላለሁ?",
        "የቁጠባ ሂሳብ ወለድ ስንት ነው?",
        "በየኢትዮጵያ ንግድ ባንክ የሞባይል ባንክ አገልግሎት ምንድን ነው?",
        "የCBE ካርድ ገደብ ስንት ነው?",
        "የብድር አሰራር ሂደት እንዴት ነው?",
        "የባንክ ቅርንጫፍ አድራሻ ምንድን ነው?",
        "የሂሳብ ቁጥር እንዴት ማግኘት እችላለሁ?",
    ]

    # Repeat templates to reach num_queries
    queries = []
    for i in range(num_queries):
        queries.append(query_templates[i % len(query_templates)])

    # Create dataset
    dataset_path = output_dir / "benchmark_dataset.json"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("w", encoding="utf-8") as f:
        json.dump([{"query": q, "expected_intent": "general"} for q in queries], f, ensure_ascii=False, indent=2)

    logger.info(f"📄 Generated {len(queries)} benchmark queries")

    # Run evaluation
    evaluate_mode(
        dataset_path=dataset_path,
        output_dir=output_dir / "results",
        max_chunks=max_chunks,
        show_progress=True,
    )


# ============================================================
# Serve Mode (Web API) – Stub for future implementation
# ============================================================

def serve_mode(host: str = "127.0.0.1", port: int = 8000) -> None:
    """
    Start a web API server (FastAPI implementation placeholder).

    Args:
        host: Host to bind to.
        port: Port to listen on.
    """
    logger.info(f"🌐 Starting web server on {host}:{port}")
    logger.info("ℹ️  Web API mode is under development.")
    logger.info("💡 Use interactive mode for now: python app.py interactive --query 'your question'")


# ============================================================
# Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Amharic FinTech RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive query
  python app.py interactive --query "የCBE ካርድ እንዴት ማግኘት እችላለሁ?"

  # Interactive with JSON output
  python app.py interactive --query "..." --output results/query.json --verbose

  # Evaluation on dataset
  python app.py evaluate --dataset data/test_queries.json --output outputs/evaluation/run1

  # Benchmark with 100 queries
  python app.py benchmark --queries 100 --output outputs/benchmark

  # Start web server (future)
  python app.py serve --port 8080
        """
    )

    subparsers = parser.add_subparsers(dest="mode", help="Mode to run")

    # Interactive mode
    interactive_parser = subparsers.add_parser("interactive", help="Interactive query mode")
    interactive_parser.add_argument("--query", "-q", required=True, help="The question to ask")
    interactive_parser.add_argument("--max-chunks", "-c", type=int, default=5, help="Max chunks to retrieve")
    interactive_parser.add_argument("--output", "-o", type=Path, help="Save result as JSON")
    interactive_parser.add_argument("--verbose", "-v", action="store_true", help="Print full prompt")

    # Evaluate mode
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluation mode")
    evaluate_parser.add_argument("--dataset", "-d", type=Path, required=True, help="Path to test dataset JSON")
    evaluate_parser.add_argument("--output", "-o", type=Path, default=Path("outputs/evaluation"), help="Output directory")
    evaluate_parser.add_argument("--max-chunks", "-c", type=int, default=5, help="Max chunks to retrieve")
    evaluate_parser.add_argument("--no-progress", action="store_true", help="Hide progress bar")

    # Benchmark mode
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark mode")
    benchmark_parser.add_argument("--queries", "-n", type=int, default=50, help="Number of queries to run")
    benchmark_parser.add_argument("--output", "-o", type=Path, default=Path("outputs/benchmark"), help="Output directory")
    benchmark_parser.add_argument("--max-chunks", "-c", type=int, default=5, help="Max chunks to retrieve")

    # Serve mode (future)
    serve_parser = subparsers.add_parser("serve", help="Start web server (under development)")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    serve_parser.add_argument("--port", "-p", type=int, default=8000, help="Port to listen on")

    # Parse arguments
    args = parser.parse_args()

    if not args.mode:
        parser.print_help()
        sys.exit(0)

    # Execute mode
    if args.mode == "interactive":
        interactive_mode(
            query=args.query,
            max_chunks=args.max_chunks,
            output_json=args.output,
            verbose=args.verbose,
        )

    elif args.mode == "evaluate":
        evaluate_mode(
            dataset_path=args.dataset,
            output_dir=args.output,
            max_chunks=args.max_chunks,
            show_progress=not args.no_progress,
        )

    elif args.mode == "benchmark":
        benchmark_mode(
            num_queries=args.queries,
            output_dir=args.output,
            max_chunks=args.max_chunks,
        )

    elif args.mode == "serve":
        serve_mode(host=args.host, port=args.port)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()


# ---------------------
# WHAT IT SOLVES:
# ---------------------
# This module provides a unified entry point for the entire RAG system.
# It supports:
#   1. Interactive query mode (ask a question, get answer + citations + quality)
#   2. Evaluation mode (run on test dataset, generate reports)
#   3. Benchmark mode (run on synthetic queries for performance testing)
#   4. Web API mode (planned for future)
#
# It integrates all modules:
#   - Generation pipeline (src.generation)
#   - Evaluation metrics (src.evaluation)
#   - Configuration (config.py)
#   - Logging and debugging
#
# ---------------------
# INPUT REQUIRED:
# ---------------------
# - For interactive: query string
# - For evaluate: dataset JSON file
# - For benchmark: number of queries
#
# All modes accept optional configuration via command-line arguments.
#
# ---------------------
# EXPECTED OUTPUT:
# ---------------------
# - Interactive: Printed answer with citations and quality metrics
# - Evaluate: JSON, CSV, Markdown reports in output directory
# - Benchmark: Evaluation reports with timing statistics
# ---------------------