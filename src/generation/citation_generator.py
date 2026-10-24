"""
src/generation/citation_generator.py

Citation Generator for Amharic FinTech RAG.

Extracts, validates, and structures citations from LLM-generated answers.
This module is the bridge between generation and quality checking, enabling:
  - Grounding verification (does each citation map to a real chunk?)
  - Hallucination detection (are there citations to non-existent chunks?)
  - Evidence coverage analysis (which chunks were actually cited?)
  - Citation accuracy scoring (what % of citations are valid?)

Integration Flow:
    LLMClient → LLMResponse → CitationGenerator → CitationReport → QualityChecker

Design Principles:
- Single Responsibility: Only handles citation extraction and validation
- Zero Dependencies: Uses duck typing for chunk objects
- Robust: Handles malformed citations, missing chunks, edge cases
- Research-Ready: Produces structured output for quality checking
- Explainable: Every citation is traceable to its source chunk
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ============================================================
# DATA STRUCTURES (Enhanced)
# ============================================================

@dataclass
class Citation:
    """
    A single citation linking the LLM's answer to a source chunk.
    
    Attributes:
        citation_id: The citation marker (e.g., "1", "2", "3")
        chunk_id: The unique ID of the source chunk
        chunk_text: The text content of the source chunk
        score: The retrieval score of the source chunk
        document_title: The document title from chunk metadata
        source: Source file/document name
        retriever_name: Which retriever produced this chunk
        verified: Whether this citation maps to a valid chunk
        context_snippet: Short excerpt showing where citation appears in answer
        metadata: Additional metadata
    """
    citation_id: str
    chunk_id: str = ""
    chunk_text: str = ""
    score: float = 0.0
    document_title: str = ""
    source: str = ""
    retriever_name: str = ""
    verified: bool = False
    context_snippet: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "chunk_id": self.chunk_id,
            "chunk_text": self.chunk_text[:200] + "..." if len(self.chunk_text) > 200 else self.chunk_text,
            "score": round(self.score, 4),
            "document_title": self.document_title,
            "source": self.source,
            "retriever_name": self.retriever_name,
            "verified": self.verified,
            "context_snippet": self.context_snippet,
        }


@dataclass
class CitationReport:
    """
    Complete citation analysis report.
    
    Attributes:
        citations: List of all extracted citations
        total_citations: Total number of citations found in the answer
        valid_citations: Number of citations that map to valid chunks
        invalid_citations: Number of citations that don't map to any chunk
        citation_accuracy: Percentage of valid citations (0.0 to 1.0)
        cited_chunk_ids: Set of chunk IDs that were actually cited
        uncited_chunk_ids: Set of chunk IDs that were not cited
        evidence_coverage: Percentage of chunks that were cited (0.0 to 1.0)
        hallucinated_citations: List of citation IDs that don't exist
        references: List of reference strings for output
        markdown: Markdown formatted output
        metadata: Additional analysis metadata
    """
    citations: List[Citation] = field(default_factory=list)
    total_citations: int = 0
    valid_citations: int = 0
    invalid_citations: int = 0
    citation_accuracy: float = 0.0
    cited_chunk_ids: Set[str] = field(default_factory=set)
    uncited_chunk_ids: Set[str] = field(default_factory=set)
    evidence_coverage: float = 0.0
    hallucinated_citations: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    markdown: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "citations": [c.to_dict() for c in self.citations],
            "total_citations": self.total_citations,
            "valid_citations": self.valid_citations,
            "invalid_citations": self.invalid_citations,
            "citation_accuracy": round(self.citation_accuracy, 4),
            "cited_chunk_ids": list(self.cited_chunk_ids),
            "uncited_chunk_ids": list(self.uncited_chunk_ids),
            "evidence_coverage": round(self.evidence_coverage, 4),
            "hallucinated_citations": self.hallucinated_citations,
            "references": self.references,
            "markdown": self.markdown,
            "metadata": self.metadata,
        }
    
    def to_markdown(self) -> str:
        """Get markdown formatted output."""
        return self.markdown


# ============================================================
# CITATION GENERATOR CLASS
# ============================================================

class CitationGenerator:
    """
    Extracts and validates citations from LLM-generated answers.
    
    This module implements the citation extraction pipeline:
      1. Extract all [N] markers from the answer text
      2. Map each marker to its source chunk (using chunk_ids_used)
      3. Validate that each citation corresponds to a real chunk
      4. Produce structured Citation objects with full metadata
      5. Calculate citation accuracy and evidence coverage
    """
    
    # Citation pattern: matches [1], [2], [12], etc.
    CITATION_PATTERN = re.compile(r"\[(\d+)\]")
    
    # Patterns for context extraction (multiple formats)
    CONTEXT_PATTERNS = [
        r"\[(\d+)\]",           # [1]
        r"\[Chunk\s*(\d+)\]",   # [Chunk 1]
        r"\[Source\s*(\d+)\]",  # [Source 1]
        r"\[Evidence\s*(\d+)\]", # [Evidence 1]
    ]
    
    def __init__(
        self,
        citation_style: str = "numeric",
        extract_context: bool = True,
        context_window: int = 50,
        include_text_preview: bool = True,
        preview_length: int = 200,
    ) -> None:
        """
        Initialize the citation generator.
        
        Args:
            citation_style: "numeric" for [1], [2], [3] format
            extract_context: Extract surrounding text for each citation
            context_window: Number of characters before/after citation to capture
            include_text_preview: Include chunk text preview in references
            preview_length: Maximum length of text preview
        """
        self.citation_style = citation_style
        self.extract_context = extract_context
        self.context_window = context_window
        self.include_text_preview = include_text_preview
        self.preview_length = preview_length
        
        logger.info(
            f"✅ CitationGenerator initialized:\n"
            f"   Style: {citation_style}\n"
            f"   Extract context: {extract_context}\n"
            f"   Context window: {context_window} chars\n"
            f"   Include text preview: {include_text_preview}"
        )
    
    def generate_citations(
        self,
        answer_text: str,
        chunks: List[Any],
        chunk_ids_used: List[str],
        query: str = "",
        strategy: str = "",
    ) -> CitationReport:
        """
        Extract and validate citations from the LLM's answer.
        
        Args:
            answer_text: The LLM-generated answer text
            chunks: List of chunk objects used in the prompt (with .chunk_id, .text, etc.)
            chunk_ids_used: List of chunk IDs in the order they appear in the prompt
            query: Original query (for metadata)
            strategy: Retrieval strategy (for metadata)
            
        Returns:
            CitationReport with all extracted and validated citations
        """
        logger.info(f"🔍 Extracting citations from answer ({len(answer_text)} chars)...")
        
        if not answer_text or not answer_text.strip():
            logger.warning("Empty answer received for citation extraction")
            return CitationReport(
                citations=[],
                total_citations=0,
                valid_citations=0,
                invalid_citations=0,
                citation_accuracy=1.0,
                metadata={"error": "Empty answer"},
            )
        
        # Step 1: Extract all citation markers
        citation_markers = self._extract_citation_markers(answer_text)
        logger.info(f"   Found {len(citation_markers)} citation markers: {citation_markers}")
        
        # Step 2: Build chunk lookup (citation_id → chunk object)
        chunk_lookup = self._build_chunk_lookup(chunks, chunk_ids_used)
        
        # Step 3: Map citations to chunks and validate
        citations = []
        valid_count = 0
        invalid_count = 0
        hallucinated = []
        cited_chunk_ids = set()
        references = []
        
        for marker in citation_markers:
            citation_id = marker  # e.g., "1", "2", "3"
            
            # Look up the chunk
            chunk = chunk_lookup.get(citation_id)
            
            if chunk:
                # Valid citation
                chunk_id = self._safe_getattr(chunk, "chunk_id", "")
                chunk_text = self._safe_getattr(chunk, "text", self._safe_getattr(chunk, "document", ""))
                score = self._safe_getattr(chunk, "score", 0.0)
                metadata = self._safe_getattr(chunk, "metadata", {}) or {}
                
                document_title = metadata.get("document_title", metadata.get("filename", "Unknown"))
                source = metadata.get("source", metadata.get("source_file", document_title))
                retriever_name = self._safe_getattr(chunk, "retriever_name", "unknown")
                
                # Extract context snippet
                context_snippet = ""
                if self.extract_context:
                    context_snippet = self._extract_context_snippet(answer_text, f"[{citation_id}]")
                
                citation = Citation(
                    citation_id=citation_id,
                    chunk_id=chunk_id,
                    chunk_text=chunk_text[:self.preview_length] if self.include_text_preview else "",
                    score=score,
                    document_title=document_title,
                    source=source,
                    retriever_name=retriever_name,
                    verified=True,
                    context_snippet=context_snippet,
                    metadata=metadata,
                )
                
                citations.append(citation)
                valid_count += 1
                cited_chunk_ids.add(chunk_id)
                
                # Build reference string
                ref_str = f"[{citation_id}] {source} | {document_title} | Score: {score:.3f}"
                if self.include_text_preview and chunk_text:
                    preview = chunk_text[:self.preview_length]
                    if len(chunk_text) > self.preview_length:
                        preview += "..."
                    ref_str += f"\n    {preview}"
                references.append(ref_str)
                
            else:
                # Invalid/hallucinated citation
                citation = Citation(
                    citation_id=citation_id,
                    verified=False,
                )
                citations.append(citation)
                invalid_count += 1
                hallucinated.append(citation_id)
                logger.warning(f"   ⚠️ Hallucinated citation: [{citation_id}] (no matching chunk)")
        
        # Step 4: Calculate metrics
        total = len(citation_markers)
        accuracy = valid_count / total if total > 0 else 1.0
        
        # Evidence coverage: what % of chunks were actually cited?
        all_chunk_ids = set(chunk_ids_used)
        uncited_chunk_ids = all_chunk_ids - cited_chunk_ids
        coverage = len(cited_chunk_ids) / len(all_chunk_ids) if all_chunk_ids else 0.0
        
        # Step 5: Build markdown output
        markdown = self._build_markdown(answer_text, references, hallucinated)
        
        # Step 6: Build report
        report = CitationReport(
            citations=citations,
            total_citations=total,
            valid_citations=valid_count,
            invalid_citations=invalid_count,
            citation_accuracy=accuracy,
            cited_chunk_ids=cited_chunk_ids,
            uncited_chunk_ids=uncited_chunk_ids,
            evidence_coverage=coverage,
            hallucinated_citations=hallucinated,
            references=references,
            markdown=markdown,
            metadata={
                "answer_length": len(answer_text),
                "total_chunks_provided": len(chunks),
                "citation_style": self.citation_style,
                "query": query,
                "strategy": strategy,
                "has_hallucinations": len(hallucinated) > 0,
                "extract_context": self.extract_context,
            },
        )
        
        logger.info(
            f"✅ Citation analysis complete:\n"
            f"   Total citations: {report.total_citations}\n"
            f"   Valid: {report.valid_citations}\n"
            f"   Invalid: {report.invalid_citations}\n"
            f"   Accuracy: {report.citation_accuracy:.2%}\n"
            f"   Evidence coverage: {report.evidence_coverage:.2%}"
        )
        
        return report
    
    def generate_from_llm_response(
        self,
        llm_response: Any,  # LLMResponse from llm_client
        chunks: List[Any],
        chunk_ids_used: List[str],
        query: str = "",
        strategy: str = "",
    ) -> CitationReport:
        """
        Generate citations directly from an LLMResponse object.
        
        Args:
            llm_response: LLMResponse from llm_client
            chunks: List of chunk objects used in the prompt
            chunk_ids_used: List of chunk IDs in order
            query: Original query (for metadata)
            strategy: Retrieval strategy (for metadata)
            
        Returns:
            CitationReport
        """
        answer_text = getattr(llm_response, "text", "")
        return self.generate_citations(
            answer_text=answer_text,
            chunks=chunks,
            chunk_ids_used=chunk_ids_used,
            query=query,
            strategy=strategy,
        )
    
    # ==========================================================
    # Private Methods
    # ==========================================================
    
    def _safe_getattr(self, obj: Any, attr: str, default: Any = None) -> Any:
        """Safely get attribute from object or dict."""
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return getattr(obj, attr, default)
    
    def _extract_citation_markers(self, text: str) -> List[str]:
        """
        Extract all citation markers from the answer text.
        
        Handles various formats:
          - [1], [2], [3] (standard)
          - [1][2] (multiple citations)
          - [1], [2] (comma-separated)
        
        Returns:
            List of citation IDs (e.g., ["1", "2", "3"])
        """
        # Find all matches
        matches = self.CITATION_PATTERN.findall(text)
        
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for match in matches:
            if match not in seen:
                seen.add(match)
                unique.append(match)
        
        return unique
    
    def _build_chunk_lookup(
        self,
        chunks: List[Any],
        chunk_ids_used: List[str],
    ) -> Dict[str, Any]:
        """
        Build a mapping from citation_id (e.g., "1") to chunk object.
        
        The citation_id corresponds to the position in chunk_ids_used:
          - "1" → chunk_ids_used[0]
          - "2" → chunk_ids_used[1]
          - etc.
        
        Returns:
            Dict mapping citation_id → chunk object
        """
        lookup = {}
        
        # Build chunk ID to chunk object mapping
        chunk_by_id = {}
        for chunk in chunks:
            cid = self._safe_getattr(chunk, "chunk_id", "")
            if cid:
                chunk_by_id[cid] = chunk
        
        # Map citation IDs to chunks
        for i, chunk_id in enumerate(chunk_ids_used, start=1):
            chunk = chunk_by_id.get(chunk_id)
            if chunk:
                lookup[str(i)] = chunk
        
        return lookup
    
    def _extract_context_snippet(self, text: str, citation_marker: str) -> str:
        """
        Extract a short snippet of text surrounding a citation marker.
        
        Args:
            text: The full answer text
            citation_marker: The citation to find (e.g., "[1]")
        
        Returns:
            A short excerpt showing where the citation appears
        """
        # Find the citation in the text
        pattern = re.compile(re.escape(citation_marker))
        match = pattern.search(text)
        
        if not match:
            return ""
        
        # Extract context window
        start = max(0, match.start() - self.context_window)
        end = min(len(text), match.end() + self.context_window)
        
        snippet = text[start:end]
        
        # Clean up whitespace
        snippet = re.sub(r"\s+", " ", snippet).strip()
        
        # Add ellipsis if truncated
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        
        return snippet
    
    def _build_markdown(self, answer_text: str, references: List[str], hallucinated: List[str]) -> str:
        """
        Build markdown formatted output.
        """
        lines = []
        
        # Answer with citations
        lines.append(answer_text)
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # References
        if references:
            lines.append("**References:**")
            lines.append("")
            for ref in references:
                lines.append(ref)
                lines.append("")
        else:
            lines.append("*No references cited.*")
        
        # Hallucination warning
        if hallucinated:
            lines.append("")
            lines.append("⚠️ **Warning:** The following citations do not correspond to any retrieved chunk:")
            lines.append(f"   {', '.join(hallucinated)}")
        
        return "\n".join(lines)


# ============================================================
# Convenience Function
# ============================================================

def generate_citations(
    answer_text: str,
    chunks: List[Any],
    chunk_ids_used: List[str],
    query: str = "",
    strategy: str = "",
    extract_context: bool = True,
    context_window: int = 50,
    include_text_preview: bool = True,
    preview_length: int = 200,
) -> CitationReport:
    """
    Convenience function to generate citations in one call.
    
    Args:
        answer_text: The LLM-generated answer text
        chunks: List of chunk objects used in the prompt
        chunk_ids_used: List of chunk IDs in order
        query: Original query (for metadata)
        strategy: Retrieval strategy (for metadata)
        extract_context: Extract surrounding text for each citation
        context_window: Number of characters before/after citation to capture
        include_text_preview: Include chunk text preview in references
        preview_length: Maximum length of text preview
        
    Returns:
        CitationReport
    """
    generator = CitationGenerator(
        extract_context=extract_context,
        context_window=context_window,
        include_text_preview=include_text_preview,
        preview_length=preview_length,
    )
    return generator.generate_citations(
        answer_text=answer_text,
        chunks=chunks,
        chunk_ids_used=chunk_ids_used,
        query=query,
        strategy=strategy,
    )


# ============================================================
# DEMO / TEST
# ============================================================

if __name__ == "__main__":
    import logging
    from dataclasses import dataclass, field

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    print("=" * 70)
    print("CITATION GENERATOR DEMO (Combined Best of Both)")
    print("=" * 70)

    # ==========================================================
    # Mock Chunk Class
    # ==========================================================
    @dataclass
    class MockChunk:
        chunk_id: str
        text: str
        score: float
        metadata: Dict[str, Any] = field(default_factory=dict)
        retriever_name: str = "hybrid"

    # ==========================================================
    # Test Data
    # ==========================================================
    mock_chunks = [
        MockChunk(
            "cbe_savings.md::chunk_0001",
            "በየኢትዮጵያ ንግድ ባንክ አዲስ የቁጠባ ሂሳብ ለመክፈት የታደሰ መታወቂያ ያስፈልጋል።",
            0.92,
            {"document_title": "cbe_savings_guide.md", "source": "cbe_savings_guide.md"}
        ),
        MockChunk(
            "cbe_savings.md::chunk_0002",
            "1. መታወቂያ ካርድ\n2. የቀበሌ ማረጋገጫ\n3. 2 ፓስፖርት ፎቶ",
            0.78,
            {"document_title": "cbe_account_opening.md", "source": "cbe_account_opening.md"}
        ),
        MockChunk(
            "cbe_savings.md::chunk_0003",
            "የቁጠባ ሂሳብ ክፍት ማድረግ ነፃ ነው።",
            0.55,
            {"document_title": "cbe_fees.md", "source": "cbe_fees.md"}
        ),
    ]

    chunk_ids_used = [c.chunk_id for c in mock_chunks]

    # ==========================================================
    # Test 1: Valid Citations
    # ==========================================================
    print("\n" + "-" * 70)
    print("TEST 1: VALID CITATIONS (All citations map to chunks)")
    print("-" * 70)

    answer_text = (
        "የCBE የቁጠባ ሂሳብ ለመክፈት የሚከተሉትን ማቅረብ ያስፈልጋል። [1]\n\n"
        "1. የታደሰ መታወቂያ ካርድ [1]\n"
        "2. የቀበሌ ማረጋገጫ [2]\n"
        "3. 2 ፓስፖርት ፎቶ [2]\n\n"
        "ሂሳብ መክፈት ነፃ ነው። [3]"
    )

    report = generate_citations(
        answer_text=answer_text,
        chunks=mock_chunks,
        chunk_ids_used=chunk_ids_used,
        query="ሂሳብ እንዴት መክፈት እችላለሁ?",
        strategy="hybrid_rerank",
        extract_context=True,
        context_window=30,
    )

    print(f"\n📊 Citation Metrics:")
    print(f"   Total Citations: {report.total_citations}")
    print(f"   Valid: {report.valid_citations}")
    print(f"   Invalid: {report.invalid_citations}")
    print(f"   Citation Accuracy: {report.citation_accuracy:.2%}")
    print(f"   Evidence Coverage: {report.evidence_coverage:.2%}")
    print(f"   Hallucinated: {report.hallucinated_citations}")

    print(f"\n📌 Citation Details:")
    for c in report.citations:
        print(f"   [{c.citation_id}] → {c.chunk_id}")
        print(f"      Verified: {c.verified}")
        print(f"      Score: {c.score:.3f}")
        print(f"      Source: {c.source}")
        if c.context_snippet:
            print(f"      Context: {c.context_snippet}")

    print(f"\n📝 Markdown Output:")
    print("-" * 70)
    print(report.to_markdown())
    print("-" * 70)

    # ==========================================================
    # Test 2: Hallucinated Citations
    # ==========================================================
    print("\n" + "-" * 70)
    print("TEST 2: HALLUCINATED CITATIONS")
    print("-" * 70)

    answer_hallucinated = (
        "የCBE ሂሳብ ለመክፈት መታወቂያ ያስፈልጋል። [1]\n"
        "እንዲሁም የባንክ ሂሳብ ቁጥር ያስፈልጋል። [99]\n"
        "ክፍያው 100 ብር ነው። [5]"
    )

    report_hallucinated = generate_citations(
        answer_text=answer_hallucinated,
        chunks=mock_chunks,
        chunk_ids_used=chunk_ids_used,
        extract_context=True,
    )

    print(f"\n⚠️ Hallucination Detected:")
    print(f"   Total: {report_hallucinated.total_citations}")
    print(f"   Valid: {report_hallucinated.valid_citations}")
    print(f"   Invalid: {report_hallucinated.invalid_citations}")
    print(f"   Accuracy: {report_hallucinated.citation_accuracy:.2%}")
    print(f"   Hallucinated: {report_hallucinated.hallucinated_citations}")

    # ==========================================================
    # Test 3: No Citations
    # ==========================================================
    print("\n" + "-" * 70)
    print("TEST 3: NO CITATIONS")
    print("-" * 70)

    answer_no_citations = "በቂ መረጃ የለኝም። እባክዎ ከCBE ይወከሉ።"

    report_no_citations = generate_citations(
        answer_text=answer_no_citations,
        chunks=mock_chunks,
        chunk_ids_used=chunk_ids_used,
    )

    print(f"\n📊 No Citations:")
    print(f"   Total: {report_no_citations.total_citations}")
    print(f"   Accuracy: {report_no_citations.citation_accuracy:.2%}")
    print(f"   Evidence Coverage: {report_no_citations.evidence_coverage:.2%}")

    # ==========================================================
    # Test 4: Multiple Citations
    # ==========================================================
    print("\n" + "-" * 70)
    print("TEST 4: MULTIPLE CITATIONS ([1][2])")
    print("-" * 70)

    answer_multi = (
        "ሂሳብ ለመክፈት መታወቂያ [1] እና ማረጋገጫ [2] ያስፈልጋል። "
        "ሁለቱም አስፈላጊ ናቸው። [1][2]"
    )

    report_multi = generate_citations(
        answer_text=answer_multi,
        chunks=mock_chunks,
        chunk_ids_used=chunk_ids_used,
    )

    print(f"\n📌 Multiple Citations:")
    print(f"   Total unique: {report_multi.total_citations}")
    print(f"   Citations found: {[c.citation_id for c in report_multi.citations]}")

    # ==========================================================
    # Test 5: Report Serialization
    # ==========================================================
    print("\n" + "-" * 70)
    print("TEST 5: REPORT SERIALIZATION")
    print("-" * 70)

    report_dict = report.to_dict()
    print(f"\n✅ Report serialized to dict:")
    for key, value in report_dict.items():
        if key == "citations":
            print(f"   {key}: [{len(value)} citations]")
        elif key == "markdown":
            print(f"   {key}: {str(value)[:80]}...")
        else:
            print(f"   {key}: {value}")

    # ==========================================================
    # Summary
    # ==========================================================
    print("\n" + "=" * 70)
    print("✅ CITATION GENERATOR DEMO COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\n📋 Features from Both Versions:")
    print("   ✅ Citation extraction (regex-based, handles [1], [2], [1][2])")
    print("   ✅ Citation validation (maps to actual chunks)")
    print("   ✅ Hallucination detection (identifies invalid citations)")
    print("   ✅ Evidence coverage (tracks which chunks were cited)")
    print("   ✅ Context extraction (shows where citations appear) ← FROM SIMPLE VERSION")
    print("   ✅ Markdown output (human-readable format) ← FROM DETAILED VERSION")
    print("   ✅ Reference building (full reference list) ← FROM DETAILED VERSION")
    print("   ✅ Structured output (CitationReport with full metadata)")
    print("   ✅ Zero dependencies (uses duck typing for chunks)")
    print("   ✅ Convenience function (generate_citations) ← FROM DETAILED VERSION")
    print("=" * 70)


# ---------------------
# WHAT IT SOLVES:
# ---------------------
# This module maps LLM-generated answers back to source chunks and
# produces grounded citations. It:
#   1. Extracts inline citations from LLM output
#   2. Validates citations against provided chunks
#   3. Builds a reference list with source metadata
#   4. Calculates citation accuracy
#   5. Tracks uncited evidence for quality checking
#   6. Provides context snippets showing where citations appear
#   7. Provides markdown output for human-readable results
#
# ---------------------
# INPUT REQUIRED:
# ---------------------
# - answer_text: str – LLM-generated answer with citations
# - chunks: List[Any] – Chunk objects used in the prompt
# - chunk_ids_used: List[str] – Chunk IDs in order they appear
# - query: str (optional) – Original query for metadata
# - strategy: str (optional) – Retrieval strategy for metadata
#
# Configuration (via __init__ or function args):
#   - extract_context: bool (default=True)
#   - context_window: int (default=50)
#   - include_text_preview: bool (default=True)
#   - preview_length: int (default=200)
#
# ---------------------
# EXPECTED OUTPUT:
# ---------------------
# CitationReport dataclass with:
#   - citations: List[Citation] – All extracted citations
#   - total_citations: int – Total citations found
#   - valid_citations: int – Citations mapping to valid chunks
#   - invalid_citations: int – Citations not mapping to any chunk
#   - citation_accuracy: float – Valid / Total
#   - cited_chunk_ids: Set[str] – Chunks actually cited
#   - uncited_chunk_ids: Set[str] – Chunks not cited
#   - evidence_coverage: float – Cited / Total chunks
#   - hallucinated_citations: List[str] – IDs of invalid citations
#   - references: List[str] – Human-readable reference list
#   - markdown: str – Markdown formatted output
#   - metadata: Dict – Citation metrics and statistics
#
# This output is consumed by:
#   - quality_checker.py (for groundedness assessment)
#   - interactive_cli.py (for display)
#   - evaluation scripts (for citation analysis)
# ---------------------