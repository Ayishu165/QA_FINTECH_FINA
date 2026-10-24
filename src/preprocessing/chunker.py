"""Passage-sized, overlapping chunking of tokenized Amharic FinTech documents.

Consumes ``tokenized_documents.json`` (output of tokenizer.py) and groups
each document's sentences into retrieval-sized chunks with globally unique 
chunk_id values.

Design note on sizing strategy (disclosed heuristic, not learned/tuned)
---------------------------------------------------------------------------
Sentences are greedily accumulated into a chunk until adding the next
sentence would exceed ``max_chunk_tokens``. If a document's final chunk
ends up below ``min_chunk_tokens``, it is merged into the previous chunk 
rather than left as a near-empty trailing fragment. This merge can cause 
that specific (last) chunk to slightly exceed max_chunk_tokens, which is an 
intentional trade-off (avoiding tiny orphan chunks matters more than a 
strict ceiling on the final chunk of a document). The defaults 
(max_chunk_tokens=220, min_chunk_tokens=60, overlap_sentences=1) are 
heuristic starting points suited to typical Amharic FinTech product 
descriptions, and should be ablated in the evaluation phase.

Design note on overlap
--------------------------
Overlap (carrying the last ``overlap_sentences`` sentences of a chunk into 
the start of the next chunk) is applied AFTER chunk boundaries are decided 
by the token budget. This keeps the sizing decision and the context-continuity 
decision independent, at the cost of chunks routinely running slightly 
larger than max_chunk_tokens once overlap is added (by design).

Design note on RAG Metadata (Added for Trustworthy RAG Novelty)
--------------------------------------------------------------------------
Each chunk is annotated with ``chunk_sequence`` and ``total_chunks_in_doc``. 
This metadata is strictly required downstream by ``citation_formatter.py`` 
to verify the logical ordering of retrieved context and detect temporal 
or structural hallucinations in the LLM's generated answers.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_FILENAME: str = "chunked_documents.json"
DEFAULT_MAX_CHUNK_TOKENS: int = 220
DEFAULT_MIN_CHUNK_TOKENS: int = 60
DEFAULT_OVERLAP_SENTENCES: int = 1

# Characters that are awkward inside a chunk_id (path separators, whitespace)
# are replaced so chunk_id stays a single clean token usable as a
# dictionary key, filename-safe string, and vector-DB ID.
_UNSAFE_ID_CHARS = re.compile(r"[\\/\s]+")


@dataclass(frozen=True)
class Chunk:
    """A single retrieval-sized chunk with RAG-verification metadata.

    Attributes:
        chunk_id: Globally unique identifier (``{sanitized_filename}::chunk_{index:04d}``).
        document_title: Source document filename.
        chunk_sequence: 0-indexed position of this chunk within the document.
        total_chunks_in_doc: Total number of chunks generated for this document.
        chunk_text: Concatenated text of the chunk's sentences.
        tokens: Concatenated surface tokens of the chunk's sentences.
        sentence_count: Number of sentences in this chunk (including overlap).
        token_count: Total token count for this chunk.
    """

    chunk_id: str
    document_title: str
    chunk_sequence: int
    total_chunks_in_doc: int
    chunk_text: str
    tokens: List[str]
    sentence_count: int
    token_count: int

    def to_dict(self) -> dict:
        """Returns the JSON-serializable representation of this chunk."""
        return {
            "chunk_id": self.chunk_id,
            "document_title": self.document_title,
            "chunk_sequence": self.chunk_sequence,
            "total_chunks_in_doc": self.total_chunks_in_doc,
            "chunk_text": self.chunk_text,
            "tokens": self.tokens,
            "sentence_count": self.sentence_count,
            "token_count": self.token_count,
        }


def _sanitize_for_id(filename: str) -> str:
    """Replaces path separators/whitespace in a filename for safe chunk_id use.

    Parameters:
        filename: Raw filename, possibly containing '/', '\\', or spaces.

    Returns:
        Filename with unsafe characters replaced by '_'.
    """
    return _UNSAFE_ID_CHARS.sub("_", filename)


def _compute_boundaries(
    token_counts: List[int], max_chunk_tokens: int, min_chunk_tokens: int
) -> List[Tuple[int, int]]:
    """Computes (start, end) sentence-index boundaries for chunk grouping.

    Parameters:
        token_counts: Token count of each sentence, in document order.
        max_chunk_tokens: Greedy accumulation ceiling per chunk.
        min_chunk_tokens: Minimum size below which a trailing chunk is
            merged into its predecessor.

    Returns:
        List of (start, end) index pairs into the sentence list, covering
        every sentence exactly once. Empty list if token_counts is empty.
    """
    if not token_counts:
        return []

    boundaries: List[Tuple[int, int]] = []
    start = 0
    current_sum = 0
    
    for index, count in enumerate(token_counts):
        if current_sum and current_sum + count > max_chunk_tokens:
            boundaries.append((start, index))
            start = index
            current_sum = 0
        current_sum += count
        
    boundaries.append((start, len(token_counts)))

    # Merge trailing chunk if it's too small
    if len(boundaries) > 1:
        last_start, last_end = boundaries[-1]
        last_sum = sum(token_counts[last_start:last_end])
        if last_sum < min_chunk_tokens:
            prev_start, _ = boundaries[-2]
            boundaries[-2] = (prev_start, last_end)
            boundaries.pop()

    return boundaries


def chunk_document(
    filename: str,
    sentences: List[dict],
    max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
    min_chunk_tokens: int = DEFAULT_MIN_CHUNK_TOKENS,
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
) -> List[Chunk]:
    """Chunks a single document's sentences into passage-sized windows.

    Parameters:
        filename: Source document filename, used to build chunk_id.
        sentences: List of dicts with "tokens" and "cleaned_text" (or "text").
        max_chunk_tokens: Greedy accumulation ceiling per chunk.
        min_chunk_tokens: Minimum size for trailing chunk merge.
        overlap_sentences: Number of trailing sentences from the previous 
            chunk to prepend to each subsequent chunk.

    Returns:
        List of Chunk objects in document order.
    """
    if max_chunk_tokens <= 0:
        raise ValueError(f"max_chunk_tokens must be positive, got {max_chunk_tokens}.")
    if min_chunk_tokens < 0:
        raise ValueError(f"min_chunk_tokens must be non-negative, got {min_chunk_tokens}.")
    if overlap_sentences < 0:
        raise ValueError(f"overlap_sentences must be non-negative, got {overlap_sentences}.")

    if not sentences:
        return []

    # Robust key detection: aligns with tokenizer.py's "cleaned_text", fallback to "text"
    text_key = "cleaned_text" if "cleaned_text" in sentences[0] else "text"
    
    token_counts = [len(sentence["tokens"]) for sentence in sentences]
    boundaries = _compute_boundaries(token_counts, max_chunk_tokens, min_chunk_tokens)

    safe_filename = _sanitize_for_id(filename)
    raw_chunks: List[Chunk] = []

    for chunk_index, (start, end) in enumerate(boundaries):
        context_start = max(0, start - overlap_sentences) if chunk_index > 0 else start
        window = sentences[context_start:end]

        chunk_tokens = [token for sentence in window for token in sentence["tokens"]]
        chunk_text = " ".join(sentence[text_key] for sentence in window)

        raw_chunks.append(
            Chunk(
                chunk_id=f"{safe_filename}::chunk_{chunk_index:04d}", # Temp ID
                document_title=filename,
                chunk_sequence=chunk_index,
                total_chunks_in_doc=0, # Placeholder, updated below
                chunk_text=chunk_text,
                tokens=chunk_tokens,
                sentence_count=len(window),
                token_count=len(chunk_tokens),
            )
        )

    # Finalize total_chunks_in_doc for all chunks in this document
    total_chunks = len(raw_chunks)
    finalized_chunks = [
        Chunk(
            chunk_id=f"{safe_filename}::chunk_{c.chunk_sequence:04d}",
            document_title=c.document_title,
            chunk_sequence=c.chunk_sequence,
            total_chunks_in_doc=total_chunks,
            chunk_text=c.chunk_text,
            tokens=c.tokens,
            sentence_count=c.sentence_count,
            token_count=c.token_count,
        )
        for c in raw_chunks
    ]

    return finalized_chunks


def chunk_corpus(
    tokenized_path: Path,
    processed_dir: Path,
    output_filename: str = DEFAULT_OUTPUT_FILENAME,
    max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
    min_chunk_tokens: int = DEFAULT_MIN_CHUNK_TOKENS,
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
) -> Tuple[int, int]:
    """Reads a tokenized corpus, chunks every document, and saves the result.

    Returns:
        Tuple of (total_documents, total_chunks_produced).
    """
    tokenized_path = Path(tokenized_path)
    if not tokenized_path.exists():
        raise FileNotFoundError(f"Tokenized corpus not found: {tokenized_path}")

    with tokenized_path.open("r", encoding="utf-8") as file_handle:
        corpus = json.load(file_handle)

    if not isinstance(corpus, list):
        raise ValueError("Corpus must be a JSON list of document records.")

    all_chunks: List[dict] = []
    documents_with_no_sentences = 0

    for record in corpus:
        filename = record.get("filename", "unknown")
        sentences = record.get("sentences", [])

        if not sentences:
            documents_with_no_sentences += 1
            logger.warning("No sentences for %s; producing zero chunks.", filename)
            continue

        document_chunks = chunk_document(
            filename,
            sentences,
            max_chunk_tokens=max_chunk_tokens,
            min_chunk_tokens=min_chunk_tokens,
            overlap_sentences=overlap_sentences,
        )
        all_chunks.extend(chunk.to_dict() for chunk in document_chunks)

    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / output_filename
    
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(all_chunks, file_handle, ensure_ascii=False, indent=2)

    if documents_with_no_sentences:
        logger.warning(
            "%d/%d documents had no sentences and produced no chunks.",
            documents_with_no_sentences,
            len(corpus),
        )
        
    logger.info(
        "Saved %d chunks from %d documents to %s",
        len(all_chunks), len(corpus), output_path,
    )
    return len(corpus), len(all_chunks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    TOKENIZED_PATH = PROCESSED_DIR / "tokenized_documents.json"

    # Mock data simulating the EXACT output of tokenizer.py
    DEMO_SENTENCES = [
        {"cleaned_text": "የኢትዮጵያ ንግድ ባንክ የካርድ አገልግሎቶች ናቸው።",
         "tokens": ["የኢትዮጵያ", "ንግድ", "ባንክ", "የ", "ካርድ", "አገልግሎቶች", "ናቸው", "።"]},
        {"cleaned_text": "ካርዱ በጥሬ ገንዘብ ወይም በቼክ ምትክ ሊያገለግል ይችላል።",
         "tokens": ["ካርዱ", "በ", "ጥሬ", "ገንዘብ", "ወይም", "በ", "ቼክ", "ምትክ", "ሊያገለግል", "ይችላል", "።"]},
        {"cleaned_text": "በካርዱ ላይ ሊጫን የሚችለው ከፍተኛው የገንዘብ መጠን 100,000 ብር ነው።",
         "tokens": ["በ", "ካርዱ", "ላይ", "ሊጫን", "የሚችለው", "ከፍተኛው", "የ", "ገንዘብ", "መጠን", "100,000", "ብር", "ነው", "።"]},
        {"cleaned_text": "ካርዱ ከአምስት ዓመት በኋላ ያበቃል።",
         "tokens": ["ካርዱ", "ከ", "አምስት", "ዓመት", "በ", "ኋላ", "ያበቃል", "።"]},
    ]

    print("\n" + "=" * 60)
    print("CHUNKER DEMO: Sentence-Aware Chunking with Overlap & Merge")
    print("=" * 60)
    print(f"\nInput: {len(DEMO_SENTENCES)} sentences, "
          f"token counts = {[len(s['tokens']) for s in DEMO_SENTENCES]}")

    demo_chunks = chunk_document(
        "demo_cbe_cards.md",
        DEMO_SENTENCES,
        max_chunk_tokens=15,
        min_chunk_tokens=5,
        overlap_sentences=1,
    )
    
    print(f"\n--- OUTPUT: {len(demo_chunks)} chunk(s) (max=15, min=5, overlap=1) ---\n")
    for chunk in demo_chunks:
        print(f"CHUNK ID         : {chunk.chunk_id}")
        print(f"SEQ / TOTAL      : {chunk.chunk_sequence} / {chunk.total_chunks_in_doc}")
        print(f"TOKEN COUNT      : {chunk.token_count}")
        print(f"SENTENCE COUNT   : {chunk.sentence_count}")
        print(f"TEXT             : {chunk.chunk_text}")
        print("-" * 40)

    print("\n" + "=" * 60)
    print("CHUNKER DEMO: Full Corpus Run")
    print("=" * 60)
    
    if TOKENIZED_PATH.exists():
        total_docs, total_chunks = chunk_corpus(
            TOKENIZED_PATH, 
            PROCESSED_DIR,
            max_chunk_tokens=DEFAULT_MAX_CHUNK_TOKENS,
            min_chunk_tokens=DEFAULT_MIN_CHUNK_TOKENS,
            overlap_sentences=DEFAULT_OVERLAP_SENTENCES,
        )
        print(f"\nTotal documents : {total_docs}")
        print(f"Total chunks    : {total_chunks}")
        print(f"Saved to        : {PROCESSED_DIR / DEFAULT_OUTPUT_FILENAME}")
    else:
        logger.info(
            "%s not found; skipping full corpus demo run. Run tokenizer.py first.",
            TOKENIZED_PATH,
        )