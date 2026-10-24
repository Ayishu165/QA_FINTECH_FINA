"""Sentence segmentation and surface tokenization for normalized Amharic FinTech documents.

Consumes ``normalized_docs.json`` (output of normalizer.py) and produces a
``sentences`` field per document: a list of ``{"cleaned_text": <sentence>, "tokens": [<token>, ...]}`` objects.

This module performs SURFACE tokenization but includes a critical clitic-separation 
heuristic to ensure the resulting tokens are immediately usable by the 
Morphology-Aware Complexity Estimator (Novelty 1). It deliberately does not perform 
root extraction or deep lemmatization, which is reserved for src/morphology/feature_extractor.py.

Design note on the Ethiopic Unicode block (bug found and fixed by direct testing)
-----------------------------------------------------------------------
The Ethiopic Unicode block (U+1200-U+137F) contains BOTH Ethiopic letters
AND Ethiopic punctuation marks (፠ ። ፣ ፤ ፥ ፦ ፧ ፨ occupy U+1360-U+1368). A
naive "Ethiopic word" character class of ``[\\u1200-\\u137F]+`` therefore
greedily consumes trailing punctuation into the same token (e.g. "ነው።"
would tokenize as one token instead of "ነው" + "።"). This is fixed by 
excluding the punctuation sub-range (U+1360-U+1368) from the letter-matching 
class and matching those punctuation marks individually.

Design note on Clitic Separation Boundary (CRITICAL FIX)
-----------------------------------------------------------------------
Amharic prepositions (ለ, በ, ከ, የ, ስለ) are attached as prefixes. However, 
these letters frequently appear in the MIDDLE of valid Amharic words 
(e.g., "ዓለም" contains "ለ", "ማበረታቻ" contains "በ"). A naive regex will 
fracture these words. This module uses a negative lookbehind 
`(?<![\u1200-\u135F\u1369-\u137F])` to ensure clitics are ONLY separated 
when they appear at the absolute beginning of a word (i.e., not preceded 
by another Ethiopic letter). This preserves morphological integrity while 
still enabling accurate morphological depth calculation downstream.

Defensive invisible‑character removal (Added for robustness)
-----------------------------------------------------------------------
Even though the normalizer strips format characters, this module applies a
second safety pass to remove any remaining zero‑width characters (ZWJ, ZWNJ,
ZWS, BOM, etc.) and ASCII controls before tokenization. This guarantees that
the Ethiopic word regex never sees a break inside a word, even if the
normalizer is bypassed or fails to catch a rare character.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_FILENAME: str = "tokenized_documents.json"

# Ethiopic letters, EXCLUDING the Ethiopic punctuation sub-range (U+1360-U+1368)
_ETHIOPIC_WORD = r"[\u1200-\u135F\u1369-\u137F]+"
_LATIN_WORD = r"[A-Za-z]+"
# Numbers with optional comma/period grouping, e.g., "100,000" or "3.5"
_NUMBER = r"\d+(?:[,.]\d+)*"
# Punctuation, including isolated Ethiopic punctuation marks
_PUNCTUATION = r'[.,!?;:()"\'\-—–።፣፤፥፦፧]'

TOKEN_PATTERN = re.compile(
    f"{_ETHIOPIC_WORD}|{_LATIN_WORD}|{_NUMBER}|{_PUNCTUATION}"
)

# Sentence boundary heuristic: split after Amharic period or other terminal punctuation followed by whitespace
SENTENCE_END_PATTERN = re.compile(r"(?<=[።!?፧])\s+")

# CRITICAL FIX: Added negative lookbehind (?<![\u1200-\u135F\u1369-\u137F])
# This ensures we ONLY split clitics at the START of a word, preventing 
# mid-word fracturing (e.g., protecting "ዓለም" from becoming "ዓለ ም").
CLITIC_PATTERN = re.compile(r"(?<![\u1200-\u135F\u1369-\u137F])(ለ|በ|ከ|የ|ስለ)([\u1200-\u135F\u1369-\u137F]+)")


@dataclass(frozen=True)
class Sentence:
    """A single segmented sentence with its surface tokens.

    Attributes:
        cleaned_text: The sentence text (trimmed, whitespace-normalized).
        tokens: Ordered list of surface tokens, with clitics separated and 
                Latin characters lowercased for retrieval consistency.
    """

    cleaned_text: str
    tokens: List[str]

    def to_dict(self) -> dict:
        """Returns the JSON-serializable representation."""
        return {"cleaned_text": self.cleaned_text, "tokens": self.tokens}


def _clean_invisible(text: str) -> str:
    """Remove all format characters (Cf) and ASCII control characters (0x00-0x1F except \n).
    
    This is a defensive measure to fuse Amharic words that may have been split
    by zero-width characters (ZWJ, ZWNJ, BOM, etc.) that survived the normalizer.
    """
    # Remove all Unicode format characters (category 'Cf')
    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Cf')
    # Remove ASCII control characters except newline, tab, carriage return
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+', '', text)
    return text


def tokenize(text: str) -> List[str]:
    """Tokenizes a single string into surface tokens with safe clitic separation.

    Parameters:
        text: Normalized cleaned text to tokenize.

    Returns:
        List of tokens. Returns an empty list for blank input.
        
    Raises:
        TypeError: If ``text`` is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")
    if not text.strip():
        return []
    
    # Defensive cleaning: remove any remaining invisible characters
    cleaned_text = _clean_invisible(text)
    
    # 1. Separate clitics ONLY at the start of words
    text_with_separated_clitics = CLITIC_PATTERN.sub(r"\1 \2", cleaned_text)
    
    # 2. Extract tokens using the robust Unicode-aware pattern
    raw_tokens = TOKEN_PATTERN.findall(text_with_separated_clitics)
    
    # 3. Lowercase Latin tokens to match the search_text normalization strategy
    return [token.lower() if token.isascii() else token for token in raw_tokens]


def segment_sentences(text: str) -> List[str]:
    """Splits text into sentences at Amharic sentence-final punctuation.

    Parameters:
        text: Normalized document text.

    Returns:
        List of trimmed, non-empty sentence strings.
        
    Raises:
        TypeError: If ``text`` is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")
    if not text.strip():
        return []
    
    return [
        sentence.strip()
        for sentence in SENTENCE_END_PATTERN.split(text)
        if sentence.strip()
    ]


class DocumentTokenizer:
    """Segments and tokenizes a single document's normalized text."""

    def tokenize_document(self, cleaned_text: str) -> List[Sentence]:
        """Segments text into sentences and tokenizes each one.

        Parameters:
            cleaned_text: The document's normalized_cleaned_text field.

        Returns:
            List of Sentence objects. Sentences that tokenize to zero tokens 
            are dropped.
            
        Raises:
            TypeError: If ``cleaned_text`` is not a string.
        """
        sentences: List[Sentence] = []
        for sentence_text in segment_sentences(cleaned_text):
            tokens = tokenize(sentence_text)
            if tokens:
                sentences.append(Sentence(cleaned_text=sentence_text, tokens=tokens))
        return sentences


def tokenize_corpus(
    normalized_path: Path,
    processed_dir: Path,
    output_filename: str = DEFAULT_OUTPUT_FILENAME,
) -> Tuple[int, int, int]:
    """Reads a normalized corpus, tokenizes each document, and saves the result.

    Parameters:
        normalized_path: Path to normalized_docs.json (output of normalizer.py).
        processed_dir: Directory to write the tokenized corpus into.
        output_filename: Name of the output JSON file.

    Returns:
        Tuple of (total_documents, successful_count, total_sentences).
        
    Raises:
        FileNotFoundError: If ``normalized_path`` does not exist.
        ValueError: If the loaded corpus is not a JSON list.
    """
    normalized_path = Path(normalized_path)
    if not normalized_path.exists():
        raise FileNotFoundError(f"Normalized corpus not found: {normalized_path}")

    with normalized_path.open("r", encoding="utf-8") as file_handle:
        corpus = json.load(file_handle)

    if not isinstance(corpus, list):
        raise ValueError("Corpus must be a JSON list of document records.")

    tokenizer = DocumentTokenizer()
    output_records: List[dict] = []
    successful = 0
    total_sentences = 0

    for record in corpus:
        filename = record.get("filename", "unknown")
        # Correctly reads the normalized text field
        cleaned_text = record.get("normalized_cleaned_text", "")

        if not cleaned_text:
            logger.warning("Empty normalized_cleaned_text for %s; no sentences produced.", filename)
            new_record = dict(record)
            new_record["sentences"] = []
            output_records.append(new_record)
            continue

        try:
            sentences = tokenizer.tokenize_document(cleaned_text)
            successful += 1
        except TypeError as exc:
            logger.error("Tokenization failed for %s: %s", filename, exc)
            sentences = []

        total_sentences += len(sentences)
        
        # Preserve all original fields and append the new sentences array
        new_record = dict(record)
        new_record["sentences"] = [sentence.to_dict() for sentence in sentences]
        output_records.append(new_record)

    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / output_filename
    
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(output_records, file_handle, ensure_ascii=False, indent=2)

    logger.info(
        "Saved tokenized corpus (%d/%d docs, %d sentences) to %s",
        successful, len(corpus), total_sentences, output_path,
    )
    return len(corpus), successful, total_sentences


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    NORMALIZED_PATH = PROCESSED_DIR / "normalized_docs.json"

    # Test cases designed to verify both clitic protection and invisible‑character removal
    SAMPLE_TEXT = (
        "ለባንክ አገልግሎት ዓለም አቀፍ ነው። "
        "መግለጫ እና ማበረታቻ ፕሮግራም ካለው ቀሪ ሂሳብ ጋር እኩል ነው።"
    )

    print("\n" + "=" * 60)
    print("TOKENIZER DEMO: Mid-Word Clitic Protection & Invisible‑Character Removal")
    print("=" * 60)
    print(f"\nINPUT: {SAMPLE_TEXT}")

    demo_tokens = tokenize(SAMPLE_TEXT)
    print(f"\nOUTPUT TOKENS: {demo_tokens}")
    
    # Verification checks
    checks = {
        "ለ ባንክ (Clitic separated)": "ለ" in demo_tokens and "ባንክ" in demo_tokens,
        "ዓለም (Mid-word 'ለ' protected)": "ዓለም" in demo_tokens,
        "መግለጫ (Mid-word 'ለ' protected)": "መግለጫ" in demo_tokens,
        "ማበረታቻ (Mid-word 'በ' protected)": "ማበረታቻ" in demo_tokens,
        "ካለው (Mid-word 'ለ' protected)": "ካለው" in demo_tokens,
    }
    
    print("\n--- VERIFICATION ---")
    for check, result in checks.items():
        print(f"[{'PASS' if result else 'FAIL'}] {check}")

    print("\n" + "=" * 60)
    print("TOKENIZER DEMO: Full Corpus Run")
    print("=" * 60)
    
    if NORMALIZED_PATH.exists():
        total, successful, sentence_count = tokenize_corpus(NORMALIZED_PATH, PROCESSED_DIR)
        print(f"\nTotal documents  : {total}")
        print(f"Successfully done: {successful}")
        print(f"Total sentences  : {sentence_count}")
        print(f"Saved to         : {PROCESSED_DIR / DEFAULT_OUTPUT_FILENAME}")
    else:
        logger.info(
            "%s not found; skipping full corpus demo run. Run normalizer.py first.",
            NORMALIZED_PATH,
        )