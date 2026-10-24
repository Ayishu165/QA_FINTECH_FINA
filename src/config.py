# src/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# ============================================================
# PROJECT ROOT & DIRECTORIES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data subdirectories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_DIR = DATA_DIR / "chunks"
QUERIES_DIR = DATA_DIR / "queries"
EVALUATION_DIR = DATA_DIR / "evaluation"

# Morphology intermediate file assets
MORPHOLOGY_DOC_FEATURES_PATH = PROCESSED_DIR / "morphology_doc_features.json"
MORPHOLOGY_DOC_COMPLEXITY_PATH = PROCESSED_DIR / "morphology_doc_complexity.json"
QUERY_ANALYSIS_EXAMPLES_PATH = PROCESSED_DIR / "query_analysis_examples.json"

# Persistent storage engine spaces
STORAGE_DIR = PROJECT_ROOT / "storage"
VECTOR_DB_DIR = STORAGE_DIR / "vector_db"

# Processed pipeline output files
CLEANED_DOCS_PATH = PROCESSED_DIR / "cleaned_docs.json"
NORMALIZED_DOCS_PATH = PROCESSED_DIR / "normalized_docs.json"
TOKENIZED_DOCS_PATH = PROCESSED_DIR / "tokenized_docs.json"
CHUNKED_DOCUMENTS_PATH = PROCESSED_DIR / "chunked_documents.json"
LEXICON_DIR = DATA_DIR / "lexicons"


# Benchmark / evaluation outputs
BENCHMARK_QUERIES_PATH = QUERIES_DIR / "benchmark_queries.json"
BENCHMARK_RESULTS_PATH = EVALUATION_DIR / "benchmark_results.json"
ABLATION_RESULTS_PATH = EVALUATION_DIR / "ablation_results.json"

# Generation outputs
GENERATION_DEMO_RESULTS_PATH = EVALUATION_DIR / "generation_demo_results.json"
GENERATION_SUMMARY_PATH = EVALUATION_DIR / "generation_summary.json"


# ============================================================
# DOMAIN RESOURCES
# ============================================================

AMHARIC_QUESTION_WORDS: List[str] = [
    "ምን", "ምንድን", "እንዴት", "መቼ", "የት", "ለምን", "ማን", "ስንት"
]

FINTECH_KEYWORDS: List[str] = [
    "ባንክ",
    "ሂሳብ",
    "ቁጠባ",
    "ብድር",
    "ክፍያ",
    "ወለድ",
    "ዝውውር",
    "ማስተላለፍ",
    "ሲቢኢ",
    "CBE",
    "CBE Birr",
    "ሞባይል ባንኪንግ",
    "ATM",
    "PIN",
    "ካርድ",
    "አፕ",
    "መመዝገብ",
    "ኮሚሽን",
    "ክፍያ ገደብ",
    "ተቀማጭ",
    "ማውጣት",
]


# ============================================================
# CONFIGURATION CLASSES
# ============================================================

@dataclass
class PreprocessingConfig:
    min_clean_char_length: int = 30
    chunk_size_tokens: int = 120
    chunk_overlap_tokens: int = 30
    min_chunk_char_length: int = 50


@dataclass
class MorphologyConfig:
    """
    Configuration for morphology-aware feature extraction and
    query/document complexity estimation.
    """

    # --------------------------------------------------------
    # Token-length / complexity thresholds
    # --------------------------------------------------------
    long_token_threshold: int = 8

    query_complexity_low_threshold: float = 0.33
    query_complexity_high_threshold: float = 0.66
    query_complexity_multi_hop_threshold: float = 0.80

    # --------------------------------------------------------
    # Grammatical prefixes
    # --------------------------------------------------------
    prefix_patterns: List[str] = field(default_factory=lambda: [
        "በ",
        "ለ",
        "ከ",
        "ወደ",
        "የ",
        "እንደ",
        "እየ",
        "ተ",
        "ማ",
        "ለማ",
    ])

    # --------------------------------------------------------
    # Common grammatical suffixes
    # --------------------------------------------------------
    suffix_patterns: List[str] = field(default_factory=lambda: [
        "ነው",
        "ናቸው",
        "ሁ",
        "ህ",
        "ሽ",
        "ኝ",
        "ው",
        "ዋ",
        "ች",
        "ዎች",
        "ታል",
        "ችላለሁ",
        "ችላል",
        "ም",
        "ን",
    ])

    # --------------------------------------------------------
    # Financial terminology
    # --------------------------------------------------------
    financial_terms: List[str] = field(default_factory=lambda: [
        "ባንክ",
        "ሂሳብ",
        "ገንዘብ",
        "ብድር",
        "ወለድ",
        "ቁጠባ",
        "ክፍያ",
        "ማስተላለፍ",
        "ATM",
        "PIN",
        "ካርድ",
        "ቴሌብር",
        "ሞባይል",
        "ባንኪንግ",
        "CBE",
        "CBE Birr",
    ])

    # --------------------------------------------------------
    # Interrogative words
    # --------------------------------------------------------
    question_words: List[str] = field(default_factory=lambda: [
        "ምን",
        "ምንድን",
        "እንዴት",
        "መቼ",
        "የት",
        "ለምን",
        "ማን",
        "ስንት",
    ])

    # --------------------------------------------------------
    # Constraint words
    # --------------------------------------------------------
    constraint_words: List[str] = field(default_factory=lambda: [
        "ገደብ",
        "ከፍተኛ",
        "ዝቅተኛ",
        "ቢያንስ",
        "ቢበዛ",
    ])

    # --------------------------------------------------------
    # Temporal expressions
    # --------------------------------------------------------
    temporal_words: List[str] = field(default_factory=lambda: [
        "ዛሬ",
        "ትላንት",
        "ነገ",
        "በየቀኑ",
        "በወር",
        "ወር",
        "ቀን",
        "ዓመታዊ",
    ])

    # --------------------------------------------------------
    # Error / troubleshooting terms
    # --------------------------------------------------------
    error_words: List[str] = field(default_factory=lambda: [
        "ስህተት",
        "error",
        "አልሰራም",
        "403",
        "404",
        "500",
    ])

    # --------------------------------------------------------
    # Multi-hop query indicators
    # --------------------------------------------------------
    multi_hop_connectors: List[str] = field(default_factory=lambda: [
        "እና",
        "ወይም",
        "ከዚያ",
        "በተጨማሪ",
        "እንዲሁም",
        "ለምን",
        "እንዴት",
    ])

    multi_hop_intent_terms: List[str] = field(default_factory=lambda: [
        "መስፈርት",
        "መስፈርቶች",
        "ሂደት",
        "ሂደቶች",
        "ሰነድ",
        "ሰነዶች",
        "ክፍያ",
        "ክፍያዎች",
        "ጊዜ",
        "እርምጃ",
        "እርምጃዎች",
        "ልዩነት",
        "አይነቶች",
    ])


@dataclass
class DenseRetrievalConfig:
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    chroma_collection_name: str = "amharic_fintech_chunks"
    top_k: int = 10


@dataclass
class SparseRetrievalConfig:
    top_k: int = 10


@dataclass
class HybridRetrievalConfig:
    top_k: int = 10
    dense_weight: float = 0.5
    sparse_weight: float = 0.5


@dataclass
class RerankerConfig:
    top_k_final: int = 5
    weight_dense: float = 0.30
    weight_sparse: float = 0.20
    weight_keyword_overlap: float = 0.25
    weight_morphology_bonus: float = 0.15
    weight_query_type_bonus: float = 0.10


@dataclass
class GenerationConfig:
    backend: str = "ollama"  # "ollama" or "mock"
    ollama_model: str = "qwen2.5:7b"
    temperature: float = 0.0  # Set to 0.0 for stable, deterministic RAG evaluation
    timeout_seconds: int = 180
    max_context_chunks: int = 5
    max_context_chars_per_chunk: int = 1800
    require_json_output: bool = True
    default_answer_language: str = "amharic"
    strict_grounding: bool = True


@dataclass
class ValidationConfig:
    grounding_overlap_threshold_high: float = 0.50
    grounding_overlap_threshold_medium: float = 0.25
    min_grounding_overlap: float = 0.12
    min_citation_count: int = 1
    unsupported_phrase_penalty: float = 0.10


@dataclass
class EvaluationConfig:
    retrieval_ks: List[int] = field(default_factory=lambda: [1, 3, 5, 10])


# ============================================================
# UNIFIED APPLICATION CONFIG
# ============================================================

@dataclass
class AppConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    morphology: MorphologyConfig = field(default_factory=MorphologyConfig)
    dense: DenseRetrievalConfig = field(default_factory=DenseRetrievalConfig)
    sparse: SparseRetrievalConfig = field(default_factory=SparseRetrievalConfig)
    hybrid: HybridRetrievalConfig = field(default_factory=HybridRetrievalConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


CONFIG = AppConfig()


# ============================================================
# RUNTIME DIRECTORY INITIALIZATION
# ============================================================

def ensure_project_directories() -> None:
    dirs = [
        DATA_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        CHUNKS_DIR,
        QUERIES_DIR,
        EVALUATION_DIR,
        STORAGE_DIR,
        VECTOR_DB_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


ensure_project_directories()