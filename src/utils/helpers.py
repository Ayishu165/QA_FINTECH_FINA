from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence


# ============================================================
# PRINT / LOG HELPERS
# ============================================================

def print_section(title: str) -> None:
    line = "=" * 70
    print(f"\n{line}\n{title}\n{line}")


# ============================================================
# TEXT CLEANING / NORMALIZATION HELPERS
# ============================================================

CONTROL_CHAR_RE = re.compile(r"[\x00-\x1F\x7F]")
MULTISPACE_RE = re.compile(r"\s+")
REPEATED_PUNCT_RE = re.compile(r"([!?።፣፤፥፦])\1+")


def remove_control_chars(text: str) -> str:
    return CONTROL_CHAR_RE.sub(" ", text)


def normalize_whitespace(text: str) -> str:
    return MULTISPACE_RE.sub(" ", text).strip()


def strip_repeated_punctuation(text: str) -> str:
    return REPEATED_PUNCT_RE.sub(r"\1", text)


def truncate_text(text: str, max_chars: int = 200) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


# Added: Safe word-boundary aware truncation function required by prompt modules
def safe_truncate(text: str, max_chars: int) -> str:
    """
    Safely truncates text to a maximum character length without cutting off mid-word.
    Appends an ellipsis if truncation occurs.
    """
    if not text or len(text) <= max_chars:
        return text
        
    truncated = text[:max_chars]
    
    # Locate the nearest Amharic/English space boundary to prevent split morphemes
    last_space = truncated.rfind(" ")
    if last_space > int(max_chars * 0.8):  
        truncated = truncated[:last_space]
        
    return truncated.rstrip() + "..."


# Added: Safe JSON processing fallbacks required by the generator module
def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    """
    Safely parses strings into dictionaries, cleaning up raw LLM block code wrappers.
    """
    if not text:
        return None
    try:
        # Strip potential markdown formatting if the LLM leaked backticks
        clean_text = text.strip()
        if clean_text.startswith("```"):
            lines = clean_text.splitlines()
            if lines[0].startswith("```json") or lines[0] == "```":
                lines = lines[1:]
            if lines and lines[-1] == "```":
                lines = lines[:-1]
            clean_text = "\n".join(lines).strip()
            
        return json.loads(clean_text)
    except Exception:
        return None


# ============================================================
# TOKENIZATION HELPERS
# ============================================================

def tokenize_whitespace(text: str) -> List[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    return text.split(" ")


def sliding_window_chunks(tokens: List[str], chunk_size: int, overlap: int) -> List[List[str]]:
    """
    Creates overlapping token windows.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if not tokens:
        return []

    windows = []
    step = chunk_size - overlap

    for start in range(0, len(tokens), step):
        end = start + chunk_size
        window = tokens[start:end]
        if not window:
            continue
        windows.append(window)
        if end >= len(tokens):
            break

    return windows


# ============================================================
# RETRIEVAL RESULT HELPERS
# ============================================================

def build_retrieval_result(
    chunk_id: str,
    source_file: str,
    text: str,
    dense_score: Optional[float],
    sparse_score: Optional[float],
    score: float,
    rerank_score: Optional[float],
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "source_file": source_file,
        "text": text,
        "dense_score": dense_score,
        "sparse_score": sparse_score,
        "score": score,
        "rerank_score": rerank_score,
        "metadata": metadata or {}
    }


def min_max_normalize(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax == vmin:
        return [1.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def keyword_overlap_score(query_tokens: List[str], chunk_tokens: List[str]) -> float:
    if not query_tokens or not chunk_tokens:
        return 0.0
    q = set(query_tokens)
    c = set(chunk_tokens)
    return len(q & c) / max(1, len(q))


# ============================================================
# TRUST / GROUNDEDNESS HELPERS
# ============================================================

def grounding_overlap(answer: str, evidence_texts: List[str]) -> float:
    """
    Simple lexical grounding score:
    fraction of unique answer tokens that appear somewhere in evidence.
    """
    answer_tokens = set(tokenize_whitespace(answer))
    if not answer_tokens:
        return 0.0

    evidence_vocab = set()
    for txt in evidence_texts:
        evidence_vocab.update(tokenize_whitespace(txt))

    overlap = len(answer_tokens & evidence_vocab)
    return overlap / max(1, len(answer_tokens))


def build_citation(citation_id: int, chunk_id: str, source_file: str, preview: str) -> Dict[str, Any]:
    return {
        "citation_id": citation_id,
        "chunk_id": chunk_id,
        "source_file": source_file,
        "preview": preview
    }


# ============================================================
# EVALUATION HELPERS
# ============================================================

def recall_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for gid in gold_ids if gid in top_k)
    return hits / len(gold_ids)


def reciprocal_rank(first_relevant_index: Optional[int]) -> float:
    if first_relevant_index is None:
        return 0.0
    return 1.0 / (first_relevant_index + 1)


def exact_match(pred: str, ref: str) -> float:
    return 1.0 if normalize_whitespace(pred) == normalize_whitespace(ref) else 0.0


def token_f1(pred: str, ref: str) -> Dict[str, float]:
    pred_tokens = tokenize_whitespace(pred)
    ref_tokens = tokenize_whitespace(ref)

    if not pred_tokens and not ref_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)

    common = pred_counter & ref_counter
    num_same = sum(common.values())

    if num_same == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}