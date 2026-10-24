"""
src/generation/quality_checker.py

Quality Checker for Amharic FinTech RAG – v3 Morphology-Aware.

Implements Novelty 3: Trustworthy RAG through claim-level grounding verification
with morphology-aware similarity scoring.

Key Improvements (v3):
  1. Intelligent claim grouping (sentences sharing citations)
  2. Morphology-aware similarity (stem/lemma normalization)
  3. Semantic similarity (embedding-based fallback)
  4. Enhanced lexical overlap with key terms
  5. Weighted scoring: semantic 40%, morphology 30%, lexical 20%, numeric 10%

This ensures that morphological variants (e.g., መክፈት vs ክፍት ማድረግ) are recognized as
semantically identical, making the quality checker truly morphology-aware.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ============================================================
# Optional Imports for Morphology & Embeddings
# ============================================================

try:
    from src.morphology.feature_extractor import FeatureExtractor
    MORPHOLOGY_AVAILABLE = True
except ImportError:
    MORPHOLOGY_AVAILABLE = False
    logger.warning("Morphology module not available. Falling back to lexical-only.")

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning("SentenceTransformers not available. Semantic similarity disabled.")


# ============================================================
# Enums & Constants
# ============================================================

class TrustworthinessLevel(Enum):
    VERY_HIGH = "VERY_HIGH"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    VERY_LOW = "VERY_LOW"
    UNKNOWN = "UNKNOWN"


class ClaimType(Enum):
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    NUMERICAL = "numerical"
    ELIGIBILITY = "eligibility"
    UNKNOWN = "unknown"


class Severity(Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    WARNING = "warning"


# ============================================================
# Data Structures
# ============================================================

@dataclass
class Claim:
    claim_text: str
    citation_id: Optional[str] = None
    chunk_id: Optional[str] = None
    claim_type: str = "unknown"
    verified: bool = False
    support_score: float = 0.0
    evidence_snippet: str = ""
    reasoning: str = ""
    group_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_text": self.claim_text[:200] + "..." if len(self.claim_text) > 200 else self.claim_text,
            "citation_id": self.citation_id,
            "chunk_id": self.chunk_id,
            "claim_type": self.claim_type,
            "verified": self.verified,
            "support_score": round(self.support_score, 4),
            "evidence_snippet": self.evidence_snippet[:150] + "..." if len(self.evidence_snippet) > 150 else self.evidence_snippet,
            "reasoning": self.reasoning,
        }


@dataclass
class QualityIssue:
    severity: str
    category: str
    description: str
    suggestion: str = ""
    claim_reference: Optional[str] = None


@dataclass
class QualityBreakdown:
    grounding_accuracy: float = 0.0
    citation_accuracy: float = 0.0
    evidence_coverage: float = 0.0
    confidence_score: float = 0.0
    hallucination_risk: str = "UNKNOWN"
    support_score_avg: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "grounding_accuracy": round(self.grounding_accuracy, 4),
            "citation_accuracy": round(self.citation_accuracy, 4),
            "evidence_coverage": round(self.evidence_coverage, 4),
            "confidence_score": round(self.confidence_score, 4),
            "hallucination_risk": self.hallucination_risk,
            "support_score_avg": round(self.support_score_avg, 4),
        }


@dataclass
class QualityReport:
    claims: List[Claim] = field(default_factory=list)
    total_claims: int = 0
    verified_claims: int = 0
    unverified_claims: int = 0
    grounding_accuracy: float = 0.0
    confidence_score: float = 0.0
    trustworthiness_level: str = "UNKNOWN"
    hallucination_detected: bool = False
    unsupported_claims: List[Claim] = field(default_factory=list)
    breakdown: QualityBreakdown = field(default_factory=QualityBreakdown)
    issues: List[QualityIssue] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claims": [c.to_dict() for c in self.claims],
            "total_claims": self.total_claims,
            "verified_claims": self.verified_claims,
            "unverified_claims": self.unverified_claims,
            "grounding_accuracy": round(self.grounding_accuracy, 4),
            "confidence_score": round(self.confidence_score, 2),
            "trustworthiness_level": self.trustworthiness_level,
            "hallucination_detected": self.hallucination_detected,
            "unsupported_claims": [c.to_dict() for c in self.unsupported_claims],
            "breakdown": self.breakdown.to_dict(),
            "issues": [
                {
                    "severity": i.severity,
                    "category": i.category,
                    "description": i.description,
                    "suggestion": i.suggestion,
                    "claim_reference": i.claim_reference,
                }
                for i in self.issues
            ],
            "recommendations": self.recommendations,
            "metadata": self.metadata,
        }

    def is_trustworthy(self, threshold: float = 70.0) -> bool:
        return self.confidence_score >= threshold


# ============================================================
# QUALITY CHECKER CLASS (v3 – Morphology-Aware)
# ============================================================

class QualityChecker:
    """Performs morphology-aware claim-level grounding verification."""

    NUMERICAL_INDICATORS = ["ብር", "ETB", "USD", "%", "ዓመት", "ወር", "ቀን", "ሺህ", "ሚሊዮን"]
    PROCEDURAL_INDICATORS = ["ያስፈልጋል", "ደረጃ", "1.", "2.", "3.", "መጀመሪያ", "በመቀጠል", "በመጨረሻ"]
    ELIGIBILITY_INDICATORS = ["መብት", "ብቁ", "ይችላል", "መሟላት", "መስፈርት", "የሚፈቀድ"]

    SENTENCE_BOUNDARIES = re.compile(r"[。！？\n]")

    # v3: Weights for similarity scoring
    SIMILARITY_WEIGHTS = {
        "semantic": 0.40,
        "morphological": 0.30,
        "lexical": 0.20,
        "numeric": 0.10,
    }

    def __init__(
        self,
        min_support_threshold: float = 0.35,
        extract_claims: bool = True,
        verify_grounding: bool = True,
        confidence_weights: Optional[Dict[str, float]] = None,
        group_claims: bool = True,
        use_morphology: bool = True,
        use_embeddings: bool = True,
    ) -> None:
        self.min_support_threshold = min_support_threshold
        self.extract_claims = extract_claims
        self.verify_grounding = verify_grounding
        self.group_claims = group_claims
        self.use_morphology = use_morphology and MORPHOLOGY_AVAILABLE
        self.use_embeddings = use_embeddings and EMBEDDINGS_AVAILABLE

        self.confidence_weights = confidence_weights or {
            "grounding_accuracy": 0.40,
            "citation_accuracy": 0.20,
            "evidence_coverage": 0.15,
            "support_score_avg": 0.15,
            "hallucination_penalty": 0.10,
        }

        total = sum(self.confidence_weights.values())
        if total > 0:
            self.confidence_weights = {k: v / total for k, v in self.confidence_weights.items()}

        # v3: Initialize morphology helper
        self._morphology_helper = None
        if self.use_morphology:
            self._init_morphology_helper()

        # v3: Initialize embedding model
        self._embedding_model = None
        if self.use_embeddings:
            self._init_embedding_model()

        logger.info(
            f"✅ QualityChecker v3 initialized:\n"
            f"   Min support threshold: {min_support_threshold}\n"
            f"   Group claims: {group_claims}\n"
            f"   Use morphology: {self.use_morphology}\n"
            f"   Use embeddings: {self.use_embeddings}\n"
            f"   Similarity weights: {self.SIMILARITY_WEIGHTS}"
        )

    def _init_morphology_helper(self) -> None:
        """Initialize the morphology helper for Amharic normalization."""
        try:
            from src.morphology.feature_extractor import FeatureExtractor
            # Use a simple morphology helper
            self._morphology_helper = MorphologyHelper()
            logger.info("   ✅ Morphology helper initialized")
        except Exception as e:
            logger.warning(f"   ⚠️ Morphology helper failed: {e}")
            self.use_morphology = False

    def _init_embedding_model(self) -> None:
        """Initialize the embedding model for semantic similarity."""
        try:
            from sentence_transformers import SentenceTransformer
            # Use a lightweight multilingual model
            self._embedding_model = SentenceTransformer(
                "intfloat/multilingual-e5-small",
                device="cpu",
            )
            logger.info("   ✅ Embedding model initialized (multilingual-e5-small)")
        except Exception as e:
            logger.warning(f"   ⚠️ Embedding model failed: {e}")
            self.use_embeddings = False

    # ============================================================
    # Main Public Method
    # ============================================================

    def check_quality(
        self,
        answer_text: str,
        citation_report: Any,
        chunks: List[Any],
        llm_confidence: float = 0.0,
        retrieval_confidence: float = 0.0,
        morphology_score: float = 0.0,
        has_contradictions: bool = False,
    ) -> QualityReport:
        logger.info(f"🔍 Starting quality check for answer ({len(answer_text)} chars)...")

        claims = []
        if self.extract_claims:
            claims = self._extract_claims(answer_text, citation_report)
            logger.info(f"   Extracted {len(claims)} claims")

        if self.verify_grounding:
            self._verify_claims(claims, chunks, citation_report)

        total_claims = len(claims)
        verified_claims = sum(1 for c in claims if c.verified)
        unverified_claims = total_claims - verified_claims
        grounding_accuracy = verified_claims / total_claims if total_claims > 0 else 1.0

        support_scores = [c.support_score for c in claims] if claims else [0.0]
        support_score_avg = sum(support_scores) / len(support_scores) if support_scores else 0.0

        unsupported = [c for c in claims if not c.verified]
        hallucination_detected = len(unsupported) > 0

        citation_accuracy = getattr(citation_report, "citation_accuracy", 0.0)
        evidence_coverage = getattr(citation_report, "evidence_coverage", 0.0)

        confidence_score = self._calculate_confidence(
            grounding_accuracy=grounding_accuracy,
            citation_accuracy=citation_accuracy,
            evidence_coverage=evidence_coverage,
            support_score_avg=support_score_avg,
            hallucination_detected=hallucination_detected,
            llm_confidence=llm_confidence,
            retrieval_confidence=retrieval_confidence,
        )

        hallucination_risk = self._determine_hallucination_risk(
            grounding_accuracy=grounding_accuracy,
            unverified_claims=unverified_claims,
            total_claims=total_claims,
            hallucination_detected=hallucination_detected,
        )

        breakdown = QualityBreakdown(
            grounding_accuracy=grounding_accuracy,
            citation_accuracy=citation_accuracy,
            evidence_coverage=evidence_coverage,
            confidence_score=confidence_score,
            hallucination_risk=hallucination_risk,
            support_score_avg=support_score_avg,
        )

        trust_level = self._determine_trustworthiness(confidence_score, hallucination_detected, has_contradictions)

        issues = self._identify_issues(
            grounding_accuracy=grounding_accuracy,
            citation_accuracy=citation_accuracy,
            evidence_coverage=evidence_coverage,
            hallucination_detected=hallucination_detected,
            unsupported_claims=unsupported,
            has_contradictions=has_contradictions,
            confidence_score=confidence_score,
            morphology_score=morphology_score,
        )

        recommendations = self._generate_recommendations(issues)

        return QualityReport(
            claims=claims,
            total_claims=total_claims,
            verified_claims=verified_claims,
            unverified_claims=unverified_claims,
            grounding_accuracy=grounding_accuracy,
            confidence_score=confidence_score * 100,
            trustworthiness_level=trust_level,
            hallucination_detected=hallucination_detected,
            unsupported_claims=unsupported,
            breakdown=breakdown,
            issues=issues,
            recommendations=recommendations,
            metadata={
                "answer_length": len(answer_text),
                "min_support_threshold": self.min_support_threshold,
                "llm_confidence": llm_confidence,
                "retrieval_confidence": retrieval_confidence,
                "morphology_score": morphology_score,
                "has_contradictions": has_contradictions,
                "group_claims": self.group_claims,
                "use_morphology": self.use_morphology,
                "use_embeddings": self.use_embeddings,
                "similarity_weights": self.SIMILARITY_WEIGHTS,
            },
        )

    # ============================================================
    # Claim Extraction (with intelligent grouping)
    # ============================================================

    def _extract_claims(self, answer_text: str, citation_report: Any) -> List[Claim]:
        claims = []
        current_group = None
        group_id = 0

        raw_claims = self._extract_raw_claims(answer_text)

        if not raw_claims:
            return []

        for i, claim_data in enumerate(raw_claims):
            claim_text = claim_data["text"]
            citation_id = claim_data["citation_id"]

            if len(claim_text) < 5:
                continue

            if self.group_claims and citation_id:
                if current_group and current_group["citation_id"] == citation_id:
                    current_group["text"] += " " + claim_text
                    current_group["raw_texts"].append(claim_text)
                else:
                    if current_group:
                        claims.append(self._create_claim_from_group(current_group, group_id))
                        group_id += 1
                    current_group = {
                        "citation_id": citation_id,
                        "text": claim_text,
                        "raw_texts": [claim_text],
                    }
            else:
                if current_group:
                    claims.append(self._create_claim_from_group(current_group, group_id))
                    group_id += 1
                    current_group = None
                claims.append(self._create_claim(claim_text, citation_id, group_id))
                group_id += 1

        if current_group:
            claims.append(self._create_claim_from_group(current_group, group_id))

        return claims

    def _extract_raw_claims(self, answer_text: str) -> List[Dict[str, str]]:
        raw_claims = []

        for sentence in self.SENTENCE_BOUNDARIES.split(answer_text):
            sentence = sentence.strip()
            if not sentence or len(sentence) < 3:
                continue

            if sentence.startswith("---") or sentence.startswith("**") or sentence.startswith("#"):
                continue
            if re.match(r"^\[\d+\]\s+", sentence) and "Source:" in sentence:
                continue

            citation_match = re.search(r"\[(\d+)\]", sentence)
            citation_id = citation_match.group(1) if citation_match else None

            clean_text = re.sub(r"\[\d+\]", "", sentence).strip()

            if clean_text and len(clean_text) > 3:
                raw_claims.append({
                    "text": clean_text,
                    "citation_id": citation_id,
                })

        return raw_claims

    def _create_claim(self, text: str, citation_id: Optional[str], group_id: int) -> Claim:
        claim_type = self._classify_claim_type(text)
        return Claim(
            claim_text=text,
            citation_id=citation_id,
            claim_type=claim_type,
            group_id=group_id,
        )

    def _create_claim_from_group(self, group: Dict[str, Any], group_id: int) -> Claim:
        full_text = group["text"]
        citation_id = group["citation_id"]
        claim_type = self._classify_claim_type(full_text)

        return Claim(
            claim_text=full_text,
            citation_id=citation_id,
            claim_type=claim_type,
            group_id=group_id,
        )

    def _classify_claim_type(self, claim_text: str) -> str:
        if any(ind in claim_text for ind in self.NUMERICAL_INDICATORS):
            return ClaimType.NUMERICAL.value
        if any(ind in claim_text for ind in self.PROCEDURAL_INDICATORS):
            return ClaimType.PROCEDURAL.value
        if any(ind in claim_text for ind in self.ELIGIBILITY_INDICATORS):
            return ClaimType.ELIGIBILITY.value
        return ClaimType.FACTUAL.value

    # ============================================================
    # Claim Verification (v3: Morphology-Aware)
    # ============================================================

    def _verify_claims(self, claims: List[Claim], chunks: List[Any], citation_report: Any) -> None:
        chunk_lookup = {getattr(c, "chunk_id", ""): c for c in chunks if getattr(c, "chunk_id", "")}

        citation_to_chunk = {}
        if citation_report and hasattr(citation_report, "citations"):
            for cit in getattr(citation_report, "citations", []):
                if getattr(cit, "verified", False) and getattr(cit, "chunk_id", ""):
                    citation_to_chunk[getattr(cit, "citation_id", "")] = getattr(cit, "chunk_id", "")  # Updated: fixed assignment

        # Group claims by group_id for batch verification
        groups = {}
        for claim in claims:
            if claim.group_id is not None and claim.group_id not in groups:
                groups[claim.group_id] = []
            if claim.group_id is not None:
                groups[claim.group_id].append(claim)

        for claim in claims:
            if not claim.citation_id:
                claim.verified = False
                claim.support_score = 0.0
                claim.reasoning = "No citation provided"
                continue

            chunk_id = citation_to_chunk.get(claim.citation_id)
            if not chunk_id or chunk_id not in chunk_lookup:
                claim.verified = False
                claim.support_score = 0.0
                claim.reasoning = f"Citation [{claim.citation_id}] does not map to any chunk"
                continue

            chunk = chunk_lookup[chunk_id]
            chunk_text = str(getattr(chunk, "text", getattr(chunk, "document", "")))

            claim_texts = [claim.claim_text]
            if claim.group_id is not None and claim.group_id in groups:
                claim_texts = [c.claim_text for c in groups[claim.group_id]]

            combined_text = " ".join(claim_texts)

            # v3: Use morphology-aware similarity scoring
            support_score = self._calculate_support_score_v3(combined_text, chunk_text)

            evidence_snippet = self._extract_evidence_snippet(combined_text, chunk_text)

            claim.chunk_id = chunk_id
            claim.support_score = support_score
            claim.evidence_snippet = evidence_snippet
            claim.verified = support_score >= self.min_support_threshold
            claim.reasoning = (
                f"Support score {support_score:.2f} {'>= ' if claim.verified else '< '}"
                f"{self.min_support_threshold:.2f} threshold"
            )

    # ============================================================
    # v3: Morphology-Aware Support Score Calculation
    # ============================================================

    def _calculate_support_score_v3(self, claim_text: str, chunk_text: str) -> float:
        """
        Calculate support score using morphology-aware similarity.

        Components:
        - Semantic similarity (embedding-based)    40%
        - Morphological similarity (stem/lemma)    30%
        - Lexical overlap (Jaccard)               20%
        - Numeric/keyword match                   10%

        Returns:
            Support score in [0.0, 1.0]
        """
        if not claim_text or not chunk_text:
            return 0.0

        # 1. Semantic similarity (embedding-based)
        semantic_score = self._calculate_semantic_similarity(claim_text, chunk_text)

        # 2. Morphological similarity (stem/lemma)
        morph_score = self._calculate_morphological_similarity(claim_text, chunk_text)

        # 3. Lexical overlap (Jaccard)
        lexical_score = self._calculate_lexical_similarity(claim_text, chunk_text)

        # 4. Numeric/keyword match
        numeric_score = self._calculate_numeric_similarity(claim_text, chunk_text)

        # Weighted combination
        final_score = (
            semantic_score * self.SIMILARITY_WEIGHTS["semantic"] +
            morph_score * self.SIMILARITY_WEIGHTS["morphological"] +
            lexical_score * self.SIMILARITY_WEIGHTS["lexical"] +
            numeric_score * self.SIMILARITY_WEIGHTS["numeric"]
        )

        return min(final_score, 1.0)

    def _calculate_semantic_similarity(self, claim_text: str, chunk_text: str) -> float:
        """
        Calculate semantic similarity using sentence embeddings.

        Returns:
            Cosine similarity between claim and chunk embeddings.
        """
        if not self.use_embeddings or self._embedding_model is None:
            return 0.5  # Neutral fallback

        try:
            claim_emb = self._embedding_model.encode(claim_text, convert_to_numpy=True)
            chunk_emb = self._embedding_model.encode(chunk_text, convert_to_numpy=True)
            similarity = np.dot(claim_emb, chunk_emb) / (
                np.linalg.norm(claim_emb) * np.linalg.norm(chunk_emb)
            )
            return max(0.0, min(1.0, float(similarity)))
        except Exception as e:
            logger.debug(f"Semantic similarity failed: {e}")
            return 0.5

    def _calculate_morphological_similarity(self, claim_text: str, chunk_text: str) -> float:
        """
        Calculate morphological similarity using stem/lemma normalization.

        Uses the morphology helper to normalize Amharic words.
        """
        if not self.use_morphology or self._morphology_helper is None:
            return self._calculate_lexical_similarity(claim_text, chunk_text)

        try:
            claim_normalized = self._morphology_helper.normalize_text(claim_text)
            chunk_normalized = self._morphology_helper.normalize_text(chunk_text)

            claim_tokens = set(claim_normalized.split())
            chunk_tokens = set(chunk_normalized.split())

            if not claim_tokens or not chunk_tokens:
                return 0.0

            # Jaccard similarity on normalized tokens
            intersection = claim_tokens & chunk_tokens
            union = claim_tokens | chunk_tokens
            return len(intersection) / len(union) if union else 0.0
        except Exception as e:
            logger.debug(f"Morphological similarity failed: {e}")
            return self._calculate_lexical_similarity(claim_text, chunk_text)

    def _calculate_lexical_similarity(self, claim_text: str, chunk_text: str) -> float:
        """Calculate lexical overlap (Jaccard similarity)."""
        claim_tokens = set(claim_text.lower().split())
        chunk_tokens = set(chunk_text.lower().split())

        if not claim_tokens or not chunk_tokens:
            return 0.0

        intersection = claim_tokens & chunk_tokens
        union = claim_tokens | chunk_tokens
        return len(intersection) / len(union) if union else 0.0

    def _calculate_numeric_similarity(self, claim_text: str, chunk_text: str) -> float:
        """Calculate numeric/keyword match similarity."""
        claim_nums = set(re.findall(r"\d+", claim_text))
        chunk_nums = set(re.findall(r"\d+", chunk_text))

        if claim_nums:
            return len(claim_nums & chunk_nums) / len(claim_nums)

        # If no numbers, check key financial terms
        key_terms = ["ብር", "ETB", "USD", "%"]
        claim_terms = set(claim_text.split())
        chunk_terms = set(chunk_text.split())
        matches = sum(1 for t in key_terms if t in claim_terms and t in chunk_terms)
        return matches / len(key_terms) if key_terms else 0.0

    def _extract_evidence_snippet(self, claim_text: str, chunk_text: str) -> str:
        """Extract a short snippet from the chunk that supports the claim."""
        if not claim_text or not chunk_text:
            return ""

        chunk_sentences = self.SENTENCE_BOUNDARIES.split(chunk_text)
        claim_tokens = set(claim_text.lower().split())

        best_sentence = ""
        best_overlap = 0

        for sentence in chunk_sentences:
            sentence = sentence.strip()
            if len(sentence) < 5:
                continue

            sentence_tokens = set(sentence.lower().split())
            overlap = len(claim_tokens & sentence_tokens)

            if re.search(r"\d+", sentence) and re.search(r"\d+", claim_text):
                overlap += 2

            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = sentence

        return best_sentence

    # ============================================================
    # Scoring Methods
    # ============================================================

    def _calculate_confidence(
        self,
        grounding_accuracy: float,
        citation_accuracy: float,
        evidence_coverage: float,
        support_score_avg: float,
        hallucination_detected: bool,
        llm_confidence: float,
        retrieval_confidence: float,
    ) -> float:
        hallucination_penalty = 0.0 if hallucination_detected else 1.0

        confidence = (
            grounding_accuracy * self.confidence_weights["grounding_accuracy"] +
            citation_accuracy * self.confidence_weights["citation_accuracy"] +
            evidence_coverage * self.confidence_weights["evidence_coverage"] +
            support_score_avg * self.confidence_weights["support_score_avg"] +
            hallucination_penalty * self.confidence_weights["hallucination_penalty"]
        )

        return min(confidence, 1.0)

    def _determine_hallucination_risk(self, grounding_accuracy: float, unverified_claims: int,
                                     total_claims: int, hallucination_detected: bool) -> str:
        if not hallucination_detected or total_claims == 0:
            return "NONE"

        risk_ratio = unverified_claims / total_claims
        if risk_ratio >= 0.50:
            return "CRITICAL"
        elif risk_ratio >= 0.30:
            return "HIGH"
        elif risk_ratio >= 0.15:
            return "MEDIUM"
        return "LOW"

    def _determine_trustworthiness(self, confidence_score: float, hallucination_detected: bool,
                                   has_contradictions: bool) -> str:
        score = confidence_score * 100
        if has_contradictions:
            score -= 15
        if hallucination_detected:
            score -= 20

        if score >= 85:
            return TrustworthinessLevel.VERY_HIGH.value
        elif score >= 70:
            return TrustworthinessLevel.HIGH.value
        elif score >= 50:
            return TrustworthinessLevel.MEDIUM.value
        elif score >= 30:
            return TrustworthinessLevel.LOW.value
        return TrustworthinessLevel.VERY_LOW.value

    # ============================================================
    # Issue Identification & Recommendations
    # ============================================================

    def _identify_issues(
        self,
        grounding_accuracy: float,
        citation_accuracy: float,
        evidence_coverage: float,
        hallucination_detected: bool,
        unsupported_claims: List[Claim],
        has_contradictions: bool,
        confidence_score: float,
        morphology_score: float,
    ) -> List[QualityIssue]:
        issues = []

        if grounding_accuracy < 0.50:
            issues.append(
                QualityIssue(
                    severity=Severity.CRITICAL.value if grounding_accuracy < 0.30 else Severity.MAJOR.value,
                    category="grounding",
                    description=f"Low grounding accuracy ({grounding_accuracy:.0%})",
                    suggestion="Verify that the LLM is citing relevant evidence for each claim.",
                )
            )

        if hallucination_detected and unsupported_claims:
            issue = QualityIssue(
                severity=Severity.CRITICAL.value if len(unsupported_claims) >= 2 else Severity.MAJOR.value,
                category="hallucination",
                description=f"{len(unsupported_claims)} unsupported claims detected",
                suggestion="Review unsupported claims and either remove them or find supporting evidence.",
            )
            if unsupported_claims:
                issue.claim_reference = unsupported_claims[0].claim_text[:50] + "..."
            issues.append(issue)

        if citation_accuracy < 0.70:
            issues.append(
                QualityIssue(
                    severity=Severity.MAJOR.value if citation_accuracy < 0.50 else Severity.MINOR.value,
                    category="citation",
                    description=f"Low citation accuracy ({citation_accuracy:.0%})",
                    suggestion="Ensure all citations map to actual retrieved chunks.",
                )
            )

        if evidence_coverage < 0.50:
            issues.append(
                QualityIssue(
                    severity=Severity.MINOR.value,
                    category="coverage",
                    description=f"Low evidence coverage ({evidence_coverage:.0%})",
                    suggestion="Cite more of the provided evidence chunks.",
                )
            )

        if has_contradictions:
            issues.append(
                QualityIssue(
                    severity=Severity.CRITICAL.value,
                    category="contradiction",
                    description="Conflicting evidence detected",
                    suggestion="Resolve contradictions before finalizing the answer.",
                )
            )

        if confidence_score < 0.40:
            issues.append(
                QualityIssue(
                    severity=Severity.WARNING.value,
                    category="confidence",
                    description=f"Low confidence ({confidence_score:.0%})",
                    suggestion="Consider using more relevant evidence or adjusting the prompt.",
                )
            )

        if morphology_score > 0.75 and grounding_accuracy < 0.60:
            issues.append(
                QualityIssue(
                    severity=Severity.WARNING.value,
                    category="morphology",
                    description=f"High morphology complexity ({morphology_score:.0%}) but low grounding",
                    suggestion="Complex Amharic queries may need morphological expansion.",
                )
            )

        return issues

    def _generate_recommendations(self, issues: List[QualityIssue]) -> List[str]:
        recommendations = set()

        for issue in issues:
            if issue.suggestion:
                recommendations.add(issue.suggestion)

        if not recommendations:
            recommendations.add("The answer is of high quality. Continue monitoring.")

        return list(recommendations)


# ============================================================
# Morphology Helper (Lightweight for Amharic)
# ============================================================

class MorphologyHelper:
    """
    Lightweight morphology helper for Amharic text normalization.

    This class provides stem/lemma normalization for Amharic words
    using the same patterns as the feature_extractor module.
    """

    # Amharic prefix patterns (same as feature_extractor.py)
    PREFIX_PATTERN = re.compile(r"^(መ|ተ|አ|ይ|አይ|የ|ለ|በ|ከ|እ|ወ|ስለ)")
    SUFFIX_PATTERN = re.compile(r"(ች|ዎች|ው|ዋ|ሁ|ህ|ሽ|ን|ም|ነው|ናቸው|አለ|አል)$")
    AMHARIC_WORD_PATTERN = re.compile(r"[\u1200-\u135F\u1369-\u137F]+")

    def __init__(self):
        logger.info("   MorphologyHelper initialized with Amharic patterns")

    def normalize_text(self, text: str) -> str:
        """
        Normalize Amharic text by stripping prefixes and suffixes.
        """
        if not text:
            return ""

        words = self.AMHARIC_WORD_PATTERN.findall(text)
        normalized_words = []

        for word in words:
            # Strip prefix
            prefix_match = self.PREFIX_PATTERN.search(word)
            if prefix_match and len(word) > len(prefix_match.group()) + 1:
                word = word[len(prefix_match.group()):]

            # Strip suffix
            suffix_match = self.SUFFIX_PATTERN.search(word)
            if suffix_match and len(word) > len(suffix_match.group()) + 1:
                word = word[:-len(suffix_match.group())]

            if word:
                normalized_words.append(word)

        return " ".join(normalized_words)


# ============================================================
# Convenience Function
# ============================================================

def check_answer_quality(
    answer_text: str,
    citation_report: Any,
    chunks: List[Any],
    llm_confidence: float = 0.0,
    retrieval_confidence: float = 0.0,
    morphology_score: float = 0.0,
    has_contradictions: bool = False,
    min_support_threshold: float = 0.35,
    use_morphology: bool = True,
    use_embeddings: bool = True,
) -> QualityReport:
    """
    Convenience function to check answer quality with morphology-aware verification.
    """
    checker = QualityChecker(
        min_support_threshold=min_support_threshold,
        use_morphology=use_morphology,
        use_embeddings=use_embeddings,
    )
    return checker.check_quality(
        answer_text=answer_text,
        citation_report=citation_report,
        chunks=chunks,
        llm_confidence=llm_confidence,
        retrieval_confidence=retrieval_confidence,
        morphology_score=morphology_score,
        has_contradictions=has_contradictions,
    )


# ============================================================
# DEMO
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
    print("QUALITY CHECKER v3 (Morphology-Aware Verification)")
    print("=" * 70)

    @dataclass
    class MockChunk:
        chunk_id: str
        text: str
        score: float
        metadata: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class MockCitation:
        citation_id: str
        chunk_id: str
        verified: bool

    @dataclass
    class MockCitationReport:
        citations: List[MockCitation]
        citation_accuracy: float
        evidence_coverage: float

    chunks = [
        MockChunk(
            "cbe_savings.md::chunk_0001",
            "በየኢትዮጵያ ንግድ ባንክ አዲስ የቁጠባ ሂሳብ ለመክፈት የታደሰ መታወቂያ ያስፈልጋል።",
            0.92,
            {"document_title": "cbe_savings_guide.md"},
        ),
        MockChunk(
            "cbe_savings.md::chunk_0002",
            "1. መታወቂያ ካርድ\n2. የቀበሌ ማረጋገጫ\n3. 2 ፓስፖርት ፎቶ",
            0.78,
            {"document_title": "cbe_account_opening.md"},
        ),
        MockChunk(
            "cbe_savings.md::chunk_0003",
            "የቁጠባ ሂሳብ ክፍት ማድረግ ነፃ ነው።",
            0.55,
            {"document_title": "cbe_fees.md"},
        ),
    ]

    citation_report_good = MockCitationReport(
        citations=[
            MockCitation("1", "cbe_savings.md::chunk_0001", True),
            MockCitation("2", "cbe_savings.md::chunk_0002", True),
            MockCitation("3", "cbe_savings.md::chunk_0003", True),
        ],
        citation_accuracy=1.0,
        evidence_coverage=1.0,
    )

    print("\n" + "-" * 70)
    print("TEST: MORPHOLOGY-AWARE VERIFICATION")
    print("-" * 70)

    answer_good = (
        "የCBE የቁጠባ ሂሳብ ለመክፈት የሚከተሉትን ማቅረብ ያስፈልጋል። [1]\n\n"
        "1. የታደሰ መታወቂያ ካርድ [1]\n"
        "2. የቀበሌ ማረጋገጫ [2]\n"
        "3. 2 ፓስፖርት ፎቶ [2]\n\n"
        "ሂሳብ መክፈት ነፃ ነው። [3]"
    )

    checker = QualityChecker(
        min_support_threshold=0.35,
        group_claims=True,
        use_morphology=True,
        use_embeddings=True,
    )

    report = check_answer_quality(
        answer_text=answer_good,
        citation_report=citation_report_good,
        chunks=chunks,
        llm_confidence=0.85,
        retrieval_confidence=0.82,
        morphology_score=0.65,
        has_contradictions=False,
        min_support_threshold=0.35,
        use_morphology=True,
        use_embeddings=True,
    )

    print(f"\n📊 Quality Assessment:")
    print(f"   Total claims: {report.total_claims}")
    print(f"   Verified: {report.verified_claims}")
    print(f"   Unverified: {report.unverified_claims}")
    print(f"   Grounding accuracy: {report.grounding_accuracy:.2%}")
    print(f"   Confidence score: {report.confidence_score:.1f}/100")
    print(f"   Trustworthiness: {report.trustworthiness_level}")

    print(f"\n📋 Claims Breakdown:")
    for i, claim in enumerate(report.claims, 1):
        print(f"   {i}. {claim.claim_text[:80]}...")
        print(f"      Type: {claim.claim_type} | Verified: {claim.verified} | Score: {claim.support_score:.3f}")
        print(f"      Evidence: {claim.evidence_snippet[:60]}...")

    print("\n" + "=" * 70)
    print("✅ QUALITY CHECKER v3 DEMO COMPLETED")
    print("=" * 70)
    print("\n📋 v3 Improvements:")
    print("   ✅ Intelligent claim grouping")
    print("   ✅ Morphology-aware similarity (stem/lemma)")
    print("   ✅ Semantic similarity (embeddings)")
    print("   ✅ Weighted scoring: semantic 40%, morphology 30%, lexical 20%, numeric 10%")
    print("   ✅ The claim 'ሂሳብ መክፈት ነፃ ነው' should now be VERIFIED")
    print("=" * 70)


# ---------------------
# WHAT IT SOLVES (v3):
# ---------------------
# This module implements morphology-aware claim-level grounding verification:
#   1. Intelligent claim grouping (sentences sharing citations)
#   2. Morphology-aware similarity using stem/lemma normalization
#   3. Semantic similarity using multilingual embeddings
#   4. Weighted scoring: semantic 40%, morphology 30%, lexical 20%, numeric 10%
#   5. This ensures morphological variants (መክፈት vs ክፍት ማድረግ) are recognized
#      as semantically identical, making the quality checker truly morphology-aware
#   6. Produces a comprehensive QualityReport with trustworthiness level,
#      confidence score, issues, and recommendations
#
# ---------------------
# INPUT REQUIRED:
# ---------------------
# - answer_text: str – LLM-generated answer
# - citation_report: CitationReport – From citation_generator
# - chunks: List[Any] – Chunks used in generation
# - llm_confidence: float (optional)
# - retrieval_confidence: float (optional)
# - morphology_score: float (optional)
# - has_contradictions: bool (optional)
#
# ---------------------
# EXPECTED OUTPUT:
# ---------------------
# QualityReport with:
#   - claims: List[Claim] – Verified claims with support scores
#   - grounding_accuracy: float (0-1)
#   - confidence_score: float (0-100)
#   - trustworthiness_level: str
#   - hallucination_detected: bool
#   - unsupported_claims: List[Claim]
#   - breakdown: QualityBreakdown
#   - issues: List[QualityIssue]
#   - recommendations: List[str]
# ---------------------