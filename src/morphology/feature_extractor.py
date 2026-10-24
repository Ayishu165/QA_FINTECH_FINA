# src/morphology/feature_extractor.py

"""
Morphology-Aware Feature Extractor for Amharic FinTech RAG.

This module extracts linguistic, morphological, lexical, and domain-specific 
features from pre-processed document chunks. It is designed to be highly 
efficient by leveraging the already-tokenized output from the preprocessing 
pipeline, avoiding redundant computation.

Design note on Morphological Analysis (Critical Fix)
----------------------------------------------------------------------
Previous iterations used arbitrary length thresholds (e.g., len >= prefix+5) 
to detect affixes, which failed on short but highly inflected Amharic words 
(e.g., "ለባንክ"). This module uses a compiled, linguistically grounded regex 
pattern that matches known Amharic derivational and inflectional affixes 
regardless of stem length, ensuring accurate morpheme counting for the 
complexity estimator.

Feature Groups Extracted:
- Statistical: token counts, TTR, hapax ratio, sentence stats
- Language: Amharic/Latin/digit ratios, mixed script detection
- Morphological: affix densities, morpheme counts, root extraction
- Lexical: entropy, stopword ratio, vocabulary size
- Domain: banking keywords, financial entities, currency mentions
- Structural: paragraphs, headings, list items
- Quality: confidence, OCR noise, clean text ratio
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Set, Optional

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts morphology-aware and domain-specific features from Amharic text chunks.
    """

    def __init__(
        self,
        stopwords_path: Path,
        banking_keywords_path: Path,
        financial_entities_path: Path,
        amharic_roots_path: Optional[Path] = None,
    ):
        """
        Initialize the feature extractor with lexicon paths.

        Parameters:
            stopwords_path: Path to stopwords.txt
            banking_keywords_path: Path to banking_keywords.txt
            financial_entities_path: Path to financial_entities.txt
            amharic_roots_path: Optional path to amharic_roots.txt
        """
        # 1. Load Lexicons
        self.stopwords: Set[str] = self._load_word_list(stopwords_path)
        self.banking_keywords: Set[str] = self._load_word_list(banking_keywords_path)
        self.financial_entities: Set[str] = self._load_word_list(financial_entities_path)
        self.amharic_roots: Set[str] = self._load_word_list(amharic_roots_path) if amharic_roots_path else set()

        # 2. Amharic Morphological Rules (Compiled for Performance)
        # Matches common Amharic prefixes (derivational & clitics)
        self._prefix_pattern = re.compile(r"^(መ|ተ|አ|ይ|አይ|የ|ለ|በ|ከ|እ|ወ|ስለ)")
        # Matches common Amharic suffixes (inflectional & pluralizers)
        self._suffix_pattern = re.compile(r"(ች|ዎች|ው|ዋ|ሁ|ህ|ሽ|ን|ም|ነው|ናቸው|አለ|አል)$")
        # Matches a full Amharic word to analyze
        self._amharic_word_pattern = re.compile(r"[\u1200-\u135F\u1369-\u137F]+")

        # 3. Domain Lexicons
        self.currency_terms = {"ብር", "ETB", "USD", "ዶላር", "Birr", "EUR", "GBP"}

        # 4. Statistical parameters
        self.long_word_threshold = 8
        self.short_word_threshold = 3

        # 5. Internal vocabulary for OOV detection
        self.corpus_vocabulary: Optional[Set[str]] = None

        logger.info(
            f"FeatureExtractor initialized with {len(self.stopwords)} stopwords, "
            f"{len(self.banking_keywords)} banking keywords, "
            f"{len(self.financial_entities)} financial entities, "
            f"{len(self.amharic_roots)} Amharic roots"
        )

    # ==========================================================
    # RESOURCE LOADING
    # ==========================================================

    def _load_word_list(self, path: Optional[Path]) -> Set[str]:
        """Loads a newline-separated text file into a set of strings."""
        if path is None or not path.exists():
            return set()
        with path.open("r", encoding="utf-8") as f:
            return {line.strip().lower() for line in f if line.strip()}

    # ==========================================================
    # TEXT UTILITIES
    # ==========================================================

    def _clean_invisible(self, text: str) -> str:
        """Remove all invisible format characters (Cf) and ASCII controls."""
        # Remove Unicode format characters (Cf)
        text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Cf')
        # Remove ASCII control characters except newline, tab, carriage return
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+', '', text)
        return text

    def _is_amharic(self, char: str) -> bool:
        """Check if a character is an Ethiopic letter."""
        return 0x1200 <= ord(char) <= 0x137F

    def _is_latin(self, char: str) -> bool:
        """Check if a character is a Latin letter."""
        return char.isascii() and char.isalpha()

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using Amharic punctuation."""
        sentences = re.split(r"[።!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    # ==========================================================
    # STATISTICAL FEATURES
    # ==========================================================

    def _extract_statistical_features(self, text: str, tokens: List[str]) -> Dict[str, Any]:
        """Extract statistical features from text and tokens."""
        token_count = len(tokens)
        unique_tokens = set(tokens)
        unique_count = len(unique_tokens)
        character_count = len(text)

        sentences = self._split_sentences(text)
        sentence_count = len(sentences)

        average_token_length = sum(len(t) for t in tokens) / token_count if token_count else 0
        average_sentence_length = token_count / sentence_count if sentence_count else 0

        type_token_ratio = unique_count / token_count if token_count else 0

        # Frequency calculation for hapax
        frequency = {}
        for token in tokens:
            frequency[token] = frequency.get(token, 0) + 1

        hapax_count = sum(1 for value in frequency.values() if value == 1)
        hapax_ratio = hapax_count / token_count if token_count else 0

        long_words = sum(1 for token in tokens if len(token) >= self.long_word_threshold)
        short_words = sum(1 for token in tokens if len(token) <= self.short_word_threshold)

        punctuation_count = len(re.findall(r"[።፣፤፦,.;:!?]", text))

        return {
            "token_count": token_count,
            "unique_token_count": unique_count,
            "character_count": character_count,
            "average_token_length": round(average_token_length, 3),
            "sentence_count": sentence_count,
            "average_sentence_length": round(average_sentence_length, 3),
            "type_token_ratio": round(type_token_ratio, 4),
            "lexical_diversity": round(type_token_ratio, 4),
            "hapax_ratio": round(hapax_ratio, 4),
            "long_word_ratio": round(long_words / token_count, 4) if token_count else 0,
            "short_word_ratio": round(short_words / token_count, 4) if token_count else 0,
            "punctuation_count": punctuation_count,
        }

    # ==========================================================
    # LANGUAGE FEATURES
    # ==========================================================

    def _extract_language_features(self, text: str, tokens: List[str]) -> Dict[str, Any]:
        """Extract language features from text and tokens."""
        total_chars = len(text)

        if total_chars == 0:
            return {"language": "am"}

        amharic_chars = sum(1 for c in text if self._is_amharic(c))
        latin_chars = sum(1 for c in text if self._is_latin(c))
        digits = sum(1 for c in text if c.isdigit())
        punctuation = len(re.findall(r"[።፣፤፦,.;:!?]", text))

        mixed_script_tokens = 0
        digit_tokens = 0

        for token in tokens:
            has_amharic = any(self._is_amharic(c) for c in token)
            has_latin = any(self._is_latin(c) for c in token)

            if has_amharic and has_latin:
                mixed_script_tokens += 1

            if any(c.isdigit() for c in token):
                digit_tokens += 1

        return {
            "language": "am",
            "amharic_character_ratio": round(amharic_chars / total_chars, 4),
            "latin_character_ratio": round(latin_chars / total_chars, 4),
            "mixed_script_ratio": round(mixed_script_tokens / len(tokens), 4) if tokens else 0,
            "digit_ratio": round(digits / total_chars, 4),
            "digit_token_ratio": round(digit_tokens / len(tokens), 4) if tokens else 0,
            "numeral_ratio": round(digits / total_chars, 4),
            "punctuation_ratio": round(punctuation / total_chars, 4),
        }

    # ==========================================================
    # MORPHOLOGICAL FEATURES (The Core Novelty - Using Regex)
    # ==========================================================

    def _extract_root(self, word: str) -> Optional[str]:
        """
        Extract the root of an Amharic word by stripping prefixes and suffixes.

        Returns the root if found, else None.
        """
        root = word

        # Strip prefixes using regex
        prefix_match = self._prefix_pattern.search(root)
        if prefix_match and len(root) > len(prefix_match.group()) + 1:
            root = root[len(prefix_match.group()):]

        # Strip suffixes using regex
        suffix_match = self._suffix_pattern.search(root)
        if suffix_match and len(root) > len(suffix_match.group()) + 1:
            root = root[:-len(suffix_match.group())]

        # Remove leftover punctuation
        root = re.sub(r'[።፣፤፥፦፧.,!?;:()"\']', '', root)

        # Check if root is in our root lexicon
        if root in self.amharic_roots:
            return root

        # If root length is at least 2 and consists of Ethiopic characters, return it
        if len(root) >= 2 and all(self._is_amharic(c) for c in root):
            return root

        return None

    def _compute_root_features(self, tokens: List[str]) -> Dict[str, Any]:
        """Compute root extraction metrics."""
        if not tokens:
            return {
                "root_extraction_ratio": 0.0,
                "root_diversity": 0.0,
                "average_root_length": 0.0,
                "unique_roots": 0,
            }

        roots = []
        for token in tokens:
            root = self._extract_root(token)
            if root:
                roots.append(root)

        if not roots:
            return {
                "root_extraction_ratio": 0.0,
                "root_diversity": 0.0,
                "average_root_length": 0.0,
                "unique_roots": 0,
            }

        unique_roots = set(roots)
        root_extraction_ratio = len(roots) / len(tokens) if tokens else 0.0
        root_diversity = len(unique_roots) / len(roots) if roots else 0.0
        avg_root_length = sum(len(r) for r in roots) / len(roots) if roots else 0.0

        return {
            "root_extraction_ratio": round(root_extraction_ratio, 4),
            "root_diversity": round(root_diversity, 4),
            "average_root_length": round(avg_root_length, 3),
            "unique_roots": len(unique_roots),
        }

    def _extract_morphological_features(self, text: str, tokens: List[str]) -> Dict[str, Any]:
        """
        Analyzes the morphological complexity of the text using regex-based 
        affix detection, avoiding the pitfalls of arbitrary length thresholds.
        """
        if not tokens:
            return self._empty_morpho_features()

        total_prefixes = 0
        total_suffixes = 0
        total_amharic_words = 0

        # Find all contiguous Amharic words in the text for accurate morphological analysis
        # We use the text directly to avoid missing words that might not be in tokens
        amharic_words = self._amharic_word_pattern.findall(text)

        for word in amharic_words:
            total_amharic_words += 1
            
            # Check for prefix using regex
            if self._prefix_pattern.search(word):
                total_prefixes += 1
                
            # Check for suffix using regex
            if self._suffix_pattern.search(word):
                total_suffixes += 1

        # Calculate densities
        affix_count = total_prefixes + total_suffixes
        avg_morphemes = (total_amharic_words + affix_count) / total_amharic_words if total_amharic_words > 0 else 1.0
        prefix_density = total_prefixes / total_amharic_words if total_amharic_words > 0 else 0.0
        suffix_density = total_suffixes / total_amharic_words if total_amharic_words > 0 else 0.0
        affix_density = affix_count / total_amharic_words if total_amharic_words > 0 else 0.0

        # Root features
        root_features = self._compute_root_features(tokens)

        return {
            "total_amharic_words": total_amharic_words,
            "total_prefixes": total_prefixes,
            "total_suffixes": total_suffixes,
            "average_morphemes_per_word": round(avg_morphemes, 3),
            "prefix_density": round(prefix_density, 4),
            "suffix_density": round(suffix_density, 4),
            "affix_density": round(affix_density, 4),
            "analyzer_source": "regex_heuristic_v2",
            # Root features
            "root_extraction_ratio": root_features["root_extraction_ratio"],
            "root_diversity": root_features["root_diversity"],
            "average_root_length": root_features["average_root_length"],
            "unique_roots": root_features["unique_roots"],
        }

    def _empty_morpho_features(self) -> Dict[str, Any]:
        return {
            "total_amharic_words": 0,
            "total_prefixes": 0,
            "total_suffixes": 0,
            "average_morphemes_per_word": 0.0,
            "prefix_density": 0.0,
            "suffix_density": 0.0,
            "affix_density": 0.0,
            "analyzer_source": "regex_heuristic_v2",
            "root_extraction_ratio": 0.0,
            "root_diversity": 0.0,
            "average_root_length": 0.0,
            "unique_roots": 0,
        }

    # ==========================================================
    # LEXICAL FEATURES
    # ==========================================================

    def _extract_lexical_features(self, tokens: List[str]) -> Dict[str, Any]:
        """Calculates lexical diversity, entropy, and stopword ratios."""
        if not tokens:
            return {
                "vocabulary_size": 0,
                "type_token_ratio": 0.0,
                "lexical_entropy": 0.0,
                "stopword_ratio": 0.0,
                "content_word_ratio": 0.0,
                "rare_word_ratio": 0.0,
            }

        vocab = set(tokens)
        vocab_size = len(vocab)
        token_count = len(tokens)

        # Stopword ratio
        stop_count = sum(1 for t in tokens if t.lower() in self.stopwords)
        
        # Rare word ratio (hapax)
        frequency = {}
        for token in tokens:
            frequency[token] = frequency.get(token, 0) + 1
        rare_count = sum(1 for value in frequency.values() if value == 1)
        rare_word_ratio = rare_count / token_count if token_count else 0

        # Lexical Entropy (Shannon Entropy)
        entropy = 0.0
        for count in frequency.values():
            p = count / token_count
            entropy -= p * math.log2(p) if p > 0 else 0

        return {
            "vocabulary_size": vocab_size,
            "type_token_ratio": round(vocab_size / token_count, 4) if token_count else 0,
            "lexical_entropy": round(entropy, 4),
            "stopword_ratio": round(stop_count / token_count, 4) if token_count else 0,
            "content_word_ratio": round((token_count - stop_count) / token_count, 4) if token_count else 0,
            "rare_word_ratio": round(rare_word_ratio, 4),
        }

    # ==========================================================
    # DOMAIN FEATURES (FinTech Specific)
    # ==========================================================

    def _extract_domain_features(self, tokens: List[str]) -> Dict[str, Any]:
        """Identifies FinTech-specific entities, keywords, and currency mentions."""
        if not tokens:
            return self._empty_domain_features()

        token_count = len(tokens)
        lower_tokens = [t.lower() for t in tokens]

        banking_hits = sum(1 for t in lower_tokens if t in self.banking_keywords)
        entity_hits = sum(1 for t in lower_tokens if t in self.financial_entities)
        currency_hits = sum(1 for t in tokens if t in self.currency_terms)
        
        # Count tokens that contain digits (e.g., "100,000", "5%")
        number_count = sum(1 for t in tokens if any(c.isdigit() for c in t))

        # Abbreviation count: uppercase Latin tokens with length <= 5
        abbreviation_count = sum(1 for t in tokens if t.isascii() and t.isupper() and len(t) <= 5)

        return {
            "domain": "fintech",
            "document_type": "banking",
            "banking_keyword_ratio": round(banking_hits / token_count, 4) if token_count else 0,
            "financial_entity_count": entity_hits,
            "entity_density": round(entity_hits / token_count, 4) if token_count else 0,
            "currency_mentions": currency_hits,
            "number_count": number_count,
            "abbreviation_ratio": round(abbreviation_count / token_count, 4) if token_count else 0,
            "domain_relevance_score": round((banking_hits + entity_hits) / token_count, 4) if token_count else 0,
        }

    def _empty_domain_features(self) -> Dict[str, Any]:
        return {
            "domain": "fintech",
            "document_type": "banking",
            "banking_keyword_ratio": 0.0,
            "financial_entity_count": 0,
            "entity_density": 0.0,
            "currency_mentions": 0,
            "number_count": 0,
            "abbreviation_ratio": 0.0,
            "domain_relevance_score": 0.0,
        }

    # ==========================================================
    # STRUCTURAL FEATURES
    # ==========================================================

    def _extract_structural_features(self, text: str) -> Dict[str, Any]:
        """Extract structural features from text."""
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        heading_count = 0
        list_item_count = 0

        for line in lines:
            if line.startswith("#"):
                heading_count += 1
            if re.match(r"^(\d+\.|-|\*)", line):
                list_item_count += 1

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        paragraph_count = len(paragraphs)

        sentences = self._split_sentences(text)
        sentence_count = len(sentences)

        word_count = len(re.findall(r"[A-Za-z]+|[\u1200-\u137F]+", text))
        average_paragraph_length = word_count / paragraph_count if paragraph_count else 0

        if word_count < 150:
            length_category = "short"
        elif word_count < 400:
            length_category = "medium"
        else:
            length_category = "long"

        return {
            "heading_count": heading_count,
            "paragraph_count": paragraph_count,
            "list_item_count": list_item_count,
            "sentence_count": sentence_count,
            "average_paragraph_length": round(average_paragraph_length, 2),
            "chunk_length_category": length_category,
        }

    # ==========================================================
    # QUALITY FEATURES
    # ==========================================================

    def _extract_quality_features(self, text: str) -> Dict[str, Any]:
        """Extract quality features to assess text cleanliness."""
        total_chars = len(text)

        if total_chars == 0:
            return {
                "feature_confidence": 0.0,
                "missing_feature_ratio": 1.0,
                "ocr_noise_ratio": 0.0,
                "clean_text_ratio": 0.0,
            }

        replacement_count = text.count("\ufffd")
        ocr_noise_ratio = replacement_count / total_chars

        return {
            "feature_confidence": round(1 - ocr_noise_ratio, 4),
            "missing_feature_ratio": 0.0,
            "ocr_noise_ratio": round(ocr_noise_ratio, 4),
            "clean_text_ratio": round(1 - ocr_noise_ratio, 4),
        }

    # ==========================================================
    # COMPLETE FEATURE EXTRACTION
    # ==========================================================

    def extract_complete_features(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrates the extraction of all feature groups for a single chunk.

        Parameters:
            chunk: Dictionary containing chunk_id, document_title, chunk_text, tokens

        Returns:
            Dictionary with chunk_id, metadata, features, timing, and quality.
        """
        start_time = time.time()

        text = chunk.get("chunk_text", "")
        # Apply defensive cleaning to remove invisible characters
        text = self._clean_invisible(text)

        # Leverage pre-computed tokens from chunker.py for efficiency
        tokens = chunk.get("tokens", [])

        # If tokens are missing, extract them from text
        if not tokens:
            tokens = re.findall(r"[A-Za-z]+|[\u1200-\u137F]+|[0-9]+", text)

        features = {}

        # Statistical
        t1 = time.time()
        features["statistical"] = self._extract_statistical_features(text, tokens)
        statistical_time = time.time() - t1

        # Language
        t1 = time.time()
        features["language"] = self._extract_language_features(text, tokens)
        language_time = time.time() - t1

        # Morphological (using regex-based analysis + root extraction)
        t1 = time.time()
        features["morphological"] = self._extract_morphological_features(text, tokens)
        morphology_time = time.time() - t1

        # Lexical
        t1 = time.time()
        features["lexical"] = self._extract_lexical_features(tokens)
        lexical_time = time.time() - t1

        # Domain
        t1 = time.time()
        features["domain"] = self._extract_domain_features(tokens)
        domain_time = time.time() - t1

        # Structural
        t1 = time.time()
        features["structural"] = self._extract_structural_features(text)
        structural_time = time.time() - t1

        # Quality
        quality = self._extract_quality_features(text)

        total_time_ms = (time.time() - start_time) * 1000

        return {
            "chunk_id": chunk.get("chunk_id"),
            "metadata": {
                "document_title": chunk.get("document_title", "unknown"),
                "language": "am",
                "feature_version": "1.2",
            },
            "features": features,
            "timing": {
                "statistical_time_ms": round(statistical_time * 1000, 3),
                "language_time_ms": round(language_time * 1000, 3),
                "morphology_time_ms": round(morphology_time * 1000, 3),
                "lexical_time_ms": round(lexical_time * 1000, 3),
                "domain_time_ms": round(domain_time * 1000, 3),
                "structural_time_ms": round(structural_time * 1000, 3),
                "total_processing_time_ms": round(total_time_ms, 3),
            },
            "quality": quality,
        }


# ==========================================================
# PIPELINE BUILDER
# ==========================================================

def build_morphology_features(
    input_path: Path,
    output_path: Path,
    stopwords_path: Path,
    banking_keywords_path: Path,
    financial_entities_path: Path,
    amharic_roots_path: Optional[Path] = None,
) -> None:
    """
    Processes the entire chunked corpus and saves morphological features.

    Parameters:
        input_path: Path to chunked_documents.json
        output_path: Path to save morphology_features.json
        stopwords_path: Path to stopwords.txt
        banking_keywords_path: Path to banking_keywords.txt
        financial_entities_path: Path to financial_entities.txt
        amharic_roots_path: Optional path to amharic_roots.txt
    """
    # Configure logging if not already configured
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    logger.info(f"Loading chunks from: {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        documents = json.load(f)

    if isinstance(documents, dict):
        documents = [documents]
    elif not isinstance(documents, list):
        raise ValueError(f"Expected list or dict, got {type(documents).__name__}")

    logger.info(f"Loaded {len(documents)} chunks")

    extractor = FeatureExtractor(
        stopwords_path=stopwords_path,
        banking_keywords_path=banking_keywords_path,
        financial_entities_path=financial_entities_path,
        amharic_roots_path=amharic_roots_path,
    )

    results = []
    total = len(documents)

    for idx, doc in enumerate(documents, start=1):
        try:
            result = extractor.extract_complete_features(doc)
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing chunk {idx}: {e}")
            continue

        if idx % 500 == 0:
            logger.info(f"Processed {idx}/{total} chunks")

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(results)} feature records to: {output_path}")


# ==========================================================
# MAIN EXECUTION
# ==========================================================

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    MORPHOLOGY_DIR = PROJECT_ROOT / "data" / "morphology"
    LEXICON_DIR = PROJECT_ROOT / "data" / "lexicons"

    # Ensure directories exist
    MORPHOLOGY_DIR.mkdir(parents=True, exist_ok=True)

    INPUT_FILE = PROCESSED_DIR / "chunked_documents.json"
    OUTPUT_FILE = MORPHOLOGY_DIR / "morphology_features.json"

    build_morphology_features(
        input_path=INPUT_FILE,
        output_path=OUTPUT_FILE,
        stopwords_path=LEXICON_DIR / "stopwords.txt",
        banking_keywords_path=LEXICON_DIR / "banking_keywords.txt",
        financial_entities_path=LEXICON_DIR / "financial_entities.txt",
        amharic_roots_path=LEXICON_DIR / "amharic_roots.txt",  # optional
    )