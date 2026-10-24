"""
src/generation/prompt_builder.py

Ultra-simple prompt builder for Qwen2.5:3B (8GB RAM).

This version is optimized for small models:
- Extremely short system prompt (2 sentences)
- One-shot example in the user prompt
- Direct instruction: "List the items with citations"
- No extra instructions (morphology, confidence, etc.)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BuiltPrompt:
    system_prompt: str
    user_prompt: str
    full_prompt: str
    messages: List[Dict[str, str]] = field(default_factory=list)
    intent: str = "general"
    chunk_ids_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PromptBuilder:
    """
    Ultra-simple prompt builder for small models.
    """

    def __init__(
        self,
        max_chunks: int = 3,
        max_context_chars: int = 3000,
    ) -> None:
        self.max_chunks = max_chunks
        self.max_context_chars = max_context_chars
        logger.info(f"✅ PromptBuilder (Ultra-Simple) initialized: max_chunks={max_chunks}")

    @staticmethod
    def _clean_content(text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def build_prompt(
        self,
        chunks: List[Any],
        query: str,
        analysis: Optional[Dict[str, Any]] = None,
        strategy: str = "",
    ) -> BuiltPrompt:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if not chunks:
            raise ValueError("Cannot build prompt from empty chunks list")

        # Select top chunks
        selected = chunks[:self.max_chunks]
        chunk_ids = []

        # Build evidence with clean [1], [2], [3] numbering
        evidence_lines = []
        for i, chunk in enumerate(selected, start=1):
            cid = getattr(chunk, "chunk_id", f"chunk_{i}")
            text = getattr(chunk, "text", getattr(chunk, "document", ""))
            text = self._clean_content(str(text))
            evidence_lines.append(f"[{i}] {text}")
            chunk_ids.append(cid)

        evidence_block = "\n".join(evidence_lines)

        # ONE-SHOT EXAMPLE: Show exactly the expected format
        one_shot_example = """
Example Question: የቁጠባ ሂሳብ ለመክፈት ምን ያስፈልጋል?

Example Evidence:
[1] አዲስ ሂሳብ ለመክፈት መታወቂያ ካርድ ያስፈልጋል።
[2] 1. መታወቂያ ካርድ 2. የቀበሌ ማረጋገጫ 3. 2 ፓስፖርት ፎቶ
[3] ሂሳብ መክፈት ነፃ ነው።

Example Answer:
የቁጠባ ሂሳብ ለመክፈት ያስፈልጋሉ፡
1. መታወቂያ ካርድ [1]
2. የቀበሌ ማረጋገጫ [2]
3. 2 ፓስፖርት ፎቶ [2]
ሂሳብ መክፈት ነፃ ነው [3]
"""

        # System prompt: VERY short
        system_prompt = """You are an Amharic FinTech assistant. Answer using ONLY the provided evidence. List items with citations like [1], [2], [3]."""

        # User prompt: includes the example and the actual query
        user_prompt = f"""
{one_shot_example}

Now answer this question using the evidence below:

QUESTION:
{query}

EVIDENCE:
{evidence_block}

INSTRUCTIONS:
List the requirements as numbered items (1., 2., 3.) with citations.
"""

        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return BuiltPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            full_prompt=full_prompt,
            messages=messages,
            intent="procedure",
            chunk_ids_used=chunk_ids,
            metadata={
                "chunks_used": len(selected),
                "prompt_tokens": len(full_prompt) // 4,
            },
        )


def build_prompt(
    chunks: List[Any],
    query: str,
    analysis: Optional[Dict[str, Any]] = None,
    strategy: str = "",
    max_chunks: int = 3,
) -> BuiltPrompt:
    builder = PromptBuilder(max_chunks=max_chunks)
    return builder.build_prompt(chunks, query, analysis, strategy)