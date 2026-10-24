"""Text normalization for cleaned Amharic FinTech documents.

Applies, in order: Zero-width character removal (to fuse broken words),
newline collapsing, Ge'ez numeral conversion, Unicode normalization (NFC),
optional Amharic homophone merging, FinTech entity standardization,
punctuation standardization, optional Arabic-Indic to Western digit
conversion, optional standalone currency token normalization (ብር → ETB),
and whitespace cleanup.

Design note on Zero-Width Characters (Critical Fix for Tokenization)
----------------------------------------------------------------------
PDF and web extraction frequently introduces Zero-Width Joiners (ZWJ) or
Zero-Width Spaces (ZWS) inside Amharic words (e.g., "ዓለ" + ZWJ + "ም").
Replacing these with a space causes catastrophic token fragmentation.
This module now removes them entirely (replaces with ""), fusing the word
back into its correct, single-token form.

To be absolutely robust, we use unicodedata.category() to strip ALL
characters in the 'Cf' (Format) category, which includes ZWJ, ZWNJ, BOM,
direction marks, and any other invisible formatting marks. This guarantees
that no stray marker survives into the tokenizer.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Match, Tuple

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_FILENAME: str = "normalized_docs.json"


@dataclass(frozen=True)
class NormalizedDocument:
    """A single normalized document record, preserving prior pipeline fields.

    Attributes:
        source_record: The original document dict, preserved for reproducibility.
        normalized_cleaned_text: Normalized text optimized for LLM prompting.
        normalized_search_text: Normalized text optimized for BM25/Embedding.
        normalized_size: Character length of normalized_cleaned_text.
    """

    source_record: dict
    normalized_cleaned_text: str
    normalized_search_text: str
    normalized_size: int

    def to_dict(self) -> dict:
        """Merges the source record with the new normalization fields."""
        merged = dict(self.source_record)
        merged["normalized_cleaned_text"] = self.normalized_cleaned_text
        merged["normalized_search_text"] = self.normalized_search_text
        merged["normalized_size"] = self.normalized_size
        return merged


class DocumentNormalizer:
    """Normalizes document text with Amharic and FinTech-specific rules.

    All regex patterns and translation tables are compiled/built once as
    class attributes to avoid recompilation cost across a large corpus.
    """

    # 1a. ASCII control characters (except newline, carriage return, tab) - remove
    CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+")
    # 1b. Non-breaking space - replace with regular space
    NBSP_PATTERN = re.compile(r"\u00a0")
    # 1c. Newlines - replace with space to ensure clean, continuous chunk_text
    NEWLINE_PATTERN = re.compile(r"\n+")

    # 2. Ge'ez Numeral to Western Arabic Numeral Mapping
    GEEZ_NUMERAL_MAP: Dict[str, str] = {
        "፩": "1", "፪": "2", "፫": "3", "፬": "4", "፭": "5",
        "፮": "6", "፯": "7", "፰": "8", "፱": "9",
        "፲": "10", "፳": "20", "፴": "30", "፵": "40", "፶": "50",
        "፷": "60", "፸": "70", "፹": "80", "፺": "90",
        "፻": "100", "፼": "10000"
    }
    GEEZ_TRANS_TABLE = str.maketrans(GEEZ_NUMERAL_MAP)

    # 3. Amharic Homophone Mapping (Optional)
    AMHARIC_HOMOPHONE_MAP: Dict[str, str] = {
        "ሐ": "ሀ", "ሑ": "ሁ", "ሒ": "ሂ", "ሓ": "ሃ", "ሔ": "ሄ", "ሕ": "ህ", "ሖ": "ሆ",
        "ኀ": "ሀ", "ኁ": "ሁ", "ኂ": "ሂ", "ኃ": "ሃ", "ኄ": "ሄ", "ኅ": "ህ", "ኆ": "ሆ",
        "ሠ": "ሰ", "ሡ": "ሱ", "ሢ": "ሲ", "ሣ": "ሳ", "ሤ": "ሴ", "ሥ": "ስ", "ሦ": "ሶ",
        "ዐ": "አ", "ዑ": "ኡ", "ዒ": "ኢ", "ዓ": "ኣ", "ዔ": "ኤ", "ዕ": "እ", "ዖ": "ኦ",
        "ጸ": "ፀ", "ጹ": "ፁ", "ጺ": "ፂ", "ጻ": "ፃ", "ጼ": "ፄ", "ጽ": "ፅ", "ጾ": "ፆ",
    }
    AMHARIC_TRANS_TABLE = str.maketrans(AMHARIC_HOMOPHONE_MAP)

    # 4. FinTech Entity Standardization (Sorted by length desc to prevent partial matches)
    ENTITY_NORMALIZATION_MAP: Dict[str, str] = {
        "የኢትዮጵያ ንግድ ባንክ": "Commercial Bank of Ethiopia",
        "ንግድ ባንክ": "Commercial Bank of Ethiopia",
        "ሲቢኢ": "CBE",
        "ብሔራዊ ባንክ": "National Bank of Ethiopia",
        "ቴሌብር": "Telebirr",
        "ኢትዮ ቴሌኮም": "Ethio Telecom"
    }
    _ENTITY_PATTERN = re.compile(
        r"|".join(re.escape(key) for key in sorted(ENTITY_NORMALIZATION_MAP.keys(), key=len, reverse=True)),
        flags=re.IGNORECASE
    )

    # 5. Punctuation & Formatting
    QUOTE_MAP: Dict[str, str] = {
        "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
        "\u2039": "<", "\u203a": ">", "\u00ab": '"', "\u00bb": '"',
    }
    QUOTE_PATTERN = re.compile("|".join(re.escape(key) for key in QUOTE_MAP))
    ELLIPSIS_PATTERN = re.compile(r"\u2026")
    MULTI_SPACE = re.compile(r"[ ]{2,}")

    # 6. Arabic-Indic Digits
    ARABIC_INDIC_MAP: Dict[str, str] = {
        "\u0660": "0", "\u0661": "1", "\u0662": "2", "\u0663": "3", "\u0664": "4",
        "\u0665": "5", "\u0666": "6", "\u0667": "7", "\u0668": "8", "\u0669": "9",
    }
    ARABIC_INDIC_PATTERN = re.compile("|".join(ARABIC_INDIC_MAP))

    # 7. Standalone Currency Token
    CURRENCY_PATTERN = re.compile(r"\bብር\b")

    def __init__(
        self,
        unicode_form: str = "NFC",
        standardize_punctuation: bool = True,
        normalize_numbers: bool = True,
        normalize_currency: bool = True,
        normalize_amharic_homophones: bool = False,
    ) -> None:
        """Configures normalization behavior."""
        if unicode_form not in ("NFC", "NFKC"):
            raise ValueError(f"unicode_form must be 'NFC' or 'NFKC', got {unicode_form!r}.")
        
        self.unicode_form = unicode_form
        self.standardize_punctuation = standardize_punctuation
        self.normalize_numbers = normalize_numbers
        self.normalize_currency = normalize_currency
        self.normalize_amharic_homophones = normalize_amharic_homophones

    @staticmethod
    def _strip_format_characters(text: str) -> str:
        """Remove all Unicode characters of category 'Cf' (Format)."""
        return ''.join(ch for ch in text if unicodedata.category(ch) != 'Cf')

    def normalize(self, text: str) -> str:
        """Applies all configured normalization steps to the input text."""
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")
        if not text.strip():
            return ""

        norm = text

        # Step 1: Remove all format characters (Cf) – this catches ZWJ, ZWNJ, BOM, etc.
        norm = self._strip_format_characters(norm)

        # Step 2: Remove other ASCII control characters (except newline/tab)
        norm = self.CONTROL_CHARS.sub("", norm)

        # Step 3: Normalize NBSP and newlines
        norm = self.NBSP_PATTERN.sub(" ", norm)
        norm = self.NEWLINE_PATTERN.sub(" ", norm)

        # Step 4: Unicode normalization
        norm = unicodedata.normalize(self.unicode_form, norm)

        # Step 5: Ge'ez numeral conversion (Always applied for FinTech)
        norm = norm.translate(self.GEEZ_TRANS_TABLE)

        # Step 6: Amharic homophone merging (Optional)
        if self.normalize_amharic_homophones:
            norm = norm.translate(self.AMHARIC_TRANS_TABLE)

        # Step 7: FinTech entity standardization
        norm = self._ENTITY_PATTERN.sub(self._replace_entity, norm)

        # Step 8: Punctuation standardization
        if self.standardize_punctuation:
            norm = self.QUOTE_PATTERN.sub(self._quote_replacement, norm)
            norm = self.ELLIPSIS_PATTERN.sub("...", norm)

        # Step 9: Number standardization (Arabic-Indic)
        if self.normalize_numbers:
            norm = self.ARABIC_INDIC_PATTERN.sub(self._digit_replacement, norm)

        # Step 10: Currency standardization
        if self.normalize_currency:
            norm = self.CURRENCY_PATTERN.sub("ETB", norm)

        # Step 11: Final whitespace cleanup
        norm = self.MULTI_SPACE.sub(" ", norm)
        
        return norm.strip()

    @staticmethod
    def _replace_entity(match: Match) -> str:
        """Callback for entity replacement to ensure exact dictionary lookup."""
        return DocumentNormalizer.ENTITY_NORMALIZATION_MAP.get(match.group(0), match.group(0))

    @staticmethod
    def _quote_replacement(match: Match) -> str:
        """Replacement callback mapping a curly quote to its ASCII form."""
        return DocumentNormalizer.QUOTE_MAP.get(match.group(0), match.group(0))

    @staticmethod
    def _digit_replacement(match: Match) -> str:
        """Replacement callback mapping an Arabic-Indic digit to Western."""
        return DocumentNormalizer.ARABIC_INDIC_MAP.get(match.group(0), match.group(0))


def normalize_corpus(
    cleaned_path: Path,
    processed_dir: Path,
    output_filename: str = DEFAULT_OUTPUT_FILENAME,
    **normalizer_kwargs,
) -> Tuple[int, int]:
    """Reads a cleaned corpus, normalizes each document, and saves the result."""
    cleaned_path = Path(cleaned_path)
    if not cleaned_path.exists():
        raise FileNotFoundError(f"Cleaned corpus not found: {cleaned_path}")

    with cleaned_path.open("r", encoding="utf-8") as file_handle:
        corpus = json.load(file_handle)

    if not isinstance(corpus, list):
        raise ValueError("Corpus must be a JSON list of document records.")

    normalizer = DocumentNormalizer(**normalizer_kwargs)
    normalized_records: List[dict] = []
    successful = 0

    for record in corpus:
        filename = record.get("filename", "unknown")
        cleaned_text = record.get("cleaned_text", "")
        search_text = record.get("search_text", "")

        if not cleaned_text and not search_text:
            logger.warning("Empty text for %s; skipping normalization.", filename)
            document = NormalizedDocument(record, "", "", 0)
            normalized_records.append(document.to_dict())
            continue

        try:
            norm_cleaned = normalizer.normalize(cleaned_text) if cleaned_text else ""
            norm_search = normalizer.normalize(search_text) if search_text else ""
            
            document = NormalizedDocument(
                source_record=record,
                normalized_cleaned_text=norm_cleaned,
                normalized_search_text=norm_search,
                normalized_size=len(norm_cleaned)
            )
            successful += 1
        except TypeError as exc:
            logger.error("Normalization failed for %s: %s; keeping original text.", filename, exc)
            document = NormalizedDocument(record, cleaned_text, search_text, len(cleaned_text))

        normalized_records.append(document.to_dict())

    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / output_filename
    
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(normalized_records, file_handle, ensure_ascii=False, indent=2)

    logger.info("Saved normalized corpus (%d/%d succeeded) to %s", successful, len(corpus), output_path)
    return len(corpus), successful


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    CLEANED_PATH = PROCESSED_DIR / "cleaned_docs.json"

    # Sample containing ZWJ inside a word and newlines
    SAMPLE_CLEANED = "ደንበኛው ፭ ሺህ ብር በሲቢኢ አካውንት አስገብቷል።\n\nብርሃን ማብራት ያስፈልጋል\u2026"
    SAMPLE_SEARCH = "ደንበኛው ፭ ሺህ ብር በ ሲቢኢ አካውንት አስገብቷል ።\n\nብርሃን ማብራት ያስፈልጋል ።"

    print("\n" + "=" * 60)
    print("NORMALIZER DEMO: Single Document Transformation")
    print("=" * 60)
    
    demo_normalizer = DocumentNormalizer(
        unicode_form="NFC",
        standardize_punctuation=True,
        normalize_numbers=True,
        normalize_currency=True,
        normalize_amharic_homophones=False,
    )
    
    norm_c = demo_normalizer.normalize(SAMPLE_CLEANED)
    norm_s = demo_normalizer.normalize(SAMPLE_SEARCH)
    
    print(f"\nOriginal Cleaned : {repr(SAMPLE_CLEANED)}")
    print(f"Normalized Cleaned: {repr(norm_c)}")
    print(
        "\nCheck: Newlines removed "
        f"({'PASS' if '\\n' not in norm_c else 'FAIL'}), "
        f"Ge'ez '፭' became '5' ({'PASS' if '5' in norm_c else 'FAIL'}), "
        f"'ሲቢኢ' became 'CBE' ({'PASS' if 'CBE' in norm_c else 'FAIL'})."
    )

    print("\n" + "=" * 60)
    print("NORMALIZER DEMO: Full Corpus Execution")
    print("=" * 60)
    
    if CLEANED_PATH.exists():
        total, successful = normalize_corpus(
            CLEANED_PATH,
            PROCESSED_DIR,
            unicode_form="NFC",
            standardize_punctuation=True,
            normalize_numbers=True,
            normalize_currency=True,
            normalize_amharic_homophones=False,
        )
        print(f"\nTotal documents  : {total}")
        print(f"Successfully done: {successful}")
        print(f"Skipped/Failed   : {total - successful}")
        print(f"Saved to         : {PROCESSED_DIR / DEFAULT_OUTPUT_FILENAME}")
    else:
        logger.info(
            "%s not found; skipping full corpus demo run. Run cleaner.py first.",
            CLEANED_PATH,
        )