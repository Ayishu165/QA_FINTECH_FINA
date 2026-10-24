"""Markdown-to-plaintext cleaning for the raw Amharic FinTech corpus.

Strips markdown syntax while preserving Amharic script, Latin script, digits, 
and standard/Amharic-specific punctuation. 

This module now implements the 8 critical preprocessing rules required for 
the Morphology-Aware RAG framework, including dual-text generation 
(cleaned_text for LLM, search_text for BM25).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Match, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MIN_LENGTH: int = 10
DEFAULT_OUTPUT_FILENAME: str = "cleaned_docs.json"


@dataclass(frozen=True)
class CleanedDocument:
    """A single cleaned document record with dual-text support.

    Attributes:
        filename: Path of the source file, relative to the raw data root.
        cleaned_text: Markdown-stripped, orthographically normalized plaintext (for LLM).
        search_text: Further normalized text optimized for sparse retrieval (BM25).
        original_size: Character length of the raw source text.
        cleaned_size: Character length of the cleaned text.
    """

    filename: str
    cleaned_text: str
    search_text: str
    original_size: int
    cleaned_size: int

    def to_dict(self) -> dict:
        """Returns a JSON-serializable representation matching the project schema."""
        return {
            "filename": self.filename,
            "cleaned_text": self.cleaned_text,
            "search_text": self.search_text,
            "original_size": self.original_size,
            "cleaned_size": self.cleaned_size,
        }


class DocumentCleaner:
    """Cleans raw markdown documents into retrieval-ready dual-text formats.

    All regex patterns are compiled once as class attributes for performance.
    """

    HEADING = re.compile(r"^#{1,6}\s*", re.MULTILINE)
    HRULE = re.compile(r"^[-_*]{3,}\s*$", re.MULTILINE)
    EMOJI = re.compile(
        r"[\U0001F000-\U0001FFFF]|[\u2600-\u27BF]|[\u2700-\u27BF]|"
        r"[\uFE00-\uFEFF]|[\u25A0-\u25FF]|[\u2300-\u23FF]|[\u2B00-\u2BFF]",
        flags=re.UNICODE,
    )
    BULLET = re.compile(
        r"^[ \t]*[•·▪▫◦➢➣➤➥➦➧➨➩➪➫➬➭➮➯➰➱➲➳➴➵➶➷➸➹➺➻➼➽➾➿✦✧✩✪✫✬✭✮✯✰✱✲✳✴✵✶✷✸✹✺✻✼✽✾✿"
        r"❀❁❂❃❄❅❆❇❈❉❊❋❌❍❎❏❐❑❒❓❔❕❖❗❘❙❚❛❜❝❞❟❠❡❢❣❤❥❦❧❨❩❪❫❬❭❮❯❰❱❲❳❴❵➔➘➙➚➛➜➝➞➟➠➡➢➣➤➥➦➧➨➩➪➫➬➭➮➯➰➱➲➳➴➵➶➷➸➹➺➻➼➽➾➿▶►▼▲]",
        flags=re.MULTILINE,
    )
    FORMAT = re.compile(r"(\*\*|__)(.*?)\1|(\*|_)(.*?)\3")
    LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")

    MULTI_SPACE = re.compile(r"[ ]{2,}")
    MULTI_NEWLINE = re.compile(r"\n{3,}")

    EXTRANEOUS = re.compile(r"[*#@_`~^\\|]")
    
    # RULE 2: Removed '\-' from allowed punctuation so hyphens are replaced by spaces.
    # '/' is already excluded, so it will also be replaced by spaces.
    ALLOWED_PUNCTUATION = r".,!?;:()\"'—–"
    AMHARIC_PUNCTUATION = r"።፣፤፥፦፧"
    KEEP_PATTERN = re.compile(
        r"[^"
        r"\u1200-\u137F"
        r"A-Za-z"
        r"0-9"
        rf"{re.escape(ALLOWED_PUNCTUATION)}"
        rf"{re.escape(AMHARIC_PUNCTUATION)}"
        r"\s"
        r"]",
        flags=re.UNICODE,
    )
    CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

    # --- NEW RULES ADDED FOR AMHARIC NLP & RAG ---
    
    # Rule 5: Foreign punctuation correction
    RE_FOREIGN_PERIOD = re.compile(r"[。．]")
    
    # Rule 5: Latin artifact removal (1-3 Latin chars attached to Amharic words)
    AMHARIC_CHAR_CLASS = r"[\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF]"
    RE_LATIN_ARTIFACT = re.compile(
        rf"({AMHARIC_CHAR_CLASS}+?)([a-zA-Z]{{1,3}})(?=[^{AMHARIC_CHAR_CLASS}a-zA-Z]|$)"
    )

    # Rule 6: Amharic Orthographic Normalization (Homophone mapping to modern standard)
    ORTHOGRAPHY_MAP = str.maketrans({
        "ሐ": "ሀ", "ኀ": "ሀ",
        "ሠ": "ሰ",
        "ፀ": "ጸ"
    })

    # Rule 3 & 4: Search text specific normalizations
    RE_PARENS = re.compile(r"[()]")
    RE_NUMBER_COMMA = re.compile(r"(\d),(\d)")
    RE_SEARCH_PUNCT = re.compile(r"[።፣፤፥፦፧.,!?;:\"'—–]+")

    # Rule 7: Clitic Separation
    # Note: 'እ' and 'ወ' are intentionally excluded to prevent fragmenting 
    # common standalone conjunctions like 'እና' (and) and 'ወይም' (or).
    RE_CLITICS = re.compile(r"(ለ|በ|ከ|የ|ስለ)([\u1200-\u137F])")

    def __init__(self, keep_headings: bool = True, collapse_newlines: bool = False) -> None:
        """Configures cleaning behavior."""
        self.keep_headings = keep_headings
        self.collapse_newlines = collapse_newlines

    def clean(self, text: str) -> Tuple[str, str]:
        """Cleans raw markdown text and returns dual-text formats.

        Parameters:
            text: Raw markdown text of one document.

        Returns:
            Tuple of (cleaned_text for LLM, search_text for BM25).
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")
        if not text.strip():
            return "", ""

        cleaned = text
        cleaned = self.HRULE.sub("", cleaned)
        if self.keep_headings:
            cleaned = self.HEADING.sub("", cleaned)
        cleaned = self.EMOJI.sub("", cleaned)
        cleaned = self.BULLET.sub("", cleaned)
        cleaned = self.FORMAT.sub(self._format_replacement, cleaned)
        cleaned = self.LINK.sub(self._link_replacement, cleaned)

        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = self.MULTI_SPACE.sub(" ", cleaned)
        cleaned = self.MULTI_NEWLINE.sub("\n\n", cleaned)
        cleaned = self.EXTRANEOUS.sub("", cleaned)
        
        # Rule 2: Hyphens and slashes are replaced with spaces here 
        # because '-' was removed from ALLOWED_PUNCTUATION.
        cleaned = self.KEEP_PATTERN.sub(" ", cleaned)
        cleaned = self.MULTI_SPACE.sub(" ", cleaned)

        # --- APPLY NEW LINGUISTIC RULES ---
        # Rule 5: Foreign punctuation
        cleaned = self.RE_FOREIGN_PERIOD.sub("።", cleaned)
        # Rule 5: Latin artifacts
        cleaned = self.RE_LATIN_ARTIFACT.sub(r"\1", cleaned)
        # Rule 6: Orthographic normalization
        cleaned = cleaned.translate(self.ORTHOGRAPHY_MAP)
        # ----------------------------------

        cleaned = self._normalize_paragraphs(cleaned)
        cleaned = self.CONTROL_CHARS.sub("", cleaned)
        cleaned_text = cleaned.strip()

        # Generate the search-optimized variant
        search_text = self._prepare_for_search(cleaned_text)

        return cleaned_text, search_text

    @staticmethod
    def _format_replacement(match: Match) -> str:
        return match.group(2) or match.group(4) or ""

    @staticmethod
    def _link_replacement(match: Match) -> str:
        return match.group(1)

    def _normalize_paragraphs(self, text: str) -> str:
        paragraphs = re.split(r"\n\s*\n", text)
        clean_paragraphs: List[str] = []
        for paragraph in paragraphs:
            normalized = " ".join(paragraph.split())
            if normalized:
                clean_paragraphs.append(normalized)

        if self.collapse_newlines:
            return " ".join(clean_paragraphs)
        return "\n\n".join(clean_paragraphs)

    def _prepare_for_search(self, text: str) -> str:
        """Applies Rules 3, 4, 7, and 8 to optimize text for sparse retrieval (BM25)."""
        # Rule 8: English Lowercasing (Amharic is unaffected by .lower())
        text = text.lower()
        
        # Rule 4: Remove commas inside numbers (e.g., 100,000 -> 100000)
        text = self.RE_NUMBER_COMMA.sub(r"\1\2", text)
        
        # Rule 3: Parentheses Handling (replace with space for token boundary)
        text = self.RE_PARENS.sub(" ", text)
        
        # Rule 4: Replace remaining punctuation with spaces
        text = self.RE_SEARCH_PUNCT.sub(" ", text)
        
        # Rule 7: Clitic Separation (e.g., 'ለባንክ' -> 'ለ ባንክ')
        text = self.RE_CLITICS.sub(r"\1 \2", text)
        
        # Final whitespace normalization
        text = self.MULTI_SPACE.sub(" ", text)
        return text.strip()


def clean_corpus(
    raw_dir: Path,
    processed_dir: Path,
    min_length: int = DEFAULT_MIN_LENGTH,
    output_filename: str = DEFAULT_OUTPUT_FILENAME,
    collapse_newlines: bool = False,
) -> Tuple[int, int, int, int]:
    """Cleans every ``.md`` file under ``raw_dir`` and writes the dual-text corpus."""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
    processed_dir.mkdir(parents=True, exist_ok=True)

    md_files = sorted(raw_dir.rglob("*.md"))
    total_files = len(md_files)
    logger.info("Found %d .md files in %s", total_files, raw_dir)

    cleaner = DocumentCleaner(collapse_newlines=collapse_newlines)
    documents: List[CleanedDocument] = []
    empty_count = 0
    short_count = 0

    for md_path in md_files:
        try:
            raw_text = md_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            logger.error("Failed to read %s: %s", md_path, exc)
            continue

        try:
            cleaned_text, search_text = cleaner.clean(raw_text)
        except TypeError as exc:
            logger.error("Cleaning failed for %s: %s", md_path, exc)
            continue

        if not cleaned_text:
            empty_count += 1
            continue
        if len(cleaned_text) < min_length:
            short_count += 1
            continue

        relative_path = md_path.relative_to(raw_dir)
        documents.append(
            CleanedDocument(
                filename=str(relative_path),
                cleaned_text=cleaned_text,
                search_text=search_text,
                original_size=len(raw_text),
                cleaned_size=len(cleaned_text),
            )
        )

    output_path = processed_dir / output_filename
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(
            [document.to_dict() for document in documents],
            file_handle,
            ensure_ascii=False,
            indent=2,
        )

    logger.info("Saved cleaned corpus (%d documents) to %s", len(documents), output_path)
    return total_files, len(documents), empty_count, short_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    RAW_DIR = PROJECT_ROOT / "data" / "raw"
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

    SAMPLE_RAW = (
        "## 1. ስማርት የአገር ውስጥ ካርዶች (Smart Domestic Cards)\n\n"
        "ለቀላጠፈ የአገር ውስጥ ክፍያዎች፣ ግብይቶች እና ለዕለት ተዕለት የባንክ አገልግሎት የተነደፉ ፈጣን፣ "
        "ደህንነታቸው የተጠበቀ እና ተመጣጣኝ ካርዶች።\n\n"
        "### 1.1 ጠቅላላ አገልግሎት የሚሰጥ የቅድመ-ክፍያ ስጦታ ካርድ\n"
        "* **መግለጫ፦** ካርዱ በጥሬ ገንዘብ ሊያገለግል ይችላልer。\n"
        "  * በካርዱ ላይ ሊጫን የሚችለው ከፍተኛው የገንዘብ መጠን 100,000 ብር ነው።\n"
    )

    print("\n" + "=" * 60)
    print("CLEANER DEMO: Dual-Text Generation")
    print("=" * 60)
    
    demo_cleaner = DocumentCleaner(collapse_newlines=False)
    cleaned_text, search_text = demo_cleaner.clean(SAMPLE_RAW)
    
    print("\n--- CLEANED TEXT (For LLM Prompting) ---")
    print(cleaned_text)
    
    print("\n--- SEARCH TEXT (For BM25 / Morphology Analysis) ---")
    print(search_text)

    if RAW_DIR.exists():
        print("\n" + "=" * 60)
        print("CLEANER DEMO: Full Corpus Run")
        print("=" * 60)
        total, retained, empty, short = clean_corpus(RAW_DIR, PROCESSED_DIR)
        print(f"\nRaw documents : {total}")
        print(f"Retained      : {retained}")
        print(f"Skipped empty : {empty}")
        print(f"Skipped short : {short}")
        print(f"Saved to      : {PROCESSED_DIR / DEFAULT_OUTPUT_FILENAME}")