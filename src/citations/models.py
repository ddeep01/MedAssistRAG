from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class Evidence:
    """Represents a retrieved evidence chunk with assigned temporary evidence_id (E1, E2, ...)."""
    evidence_id: str
    document_id: str
    chunk_id: str
    text: str
    source: str
    title: str
    url: Optional[str] = None
    hybrid_score: float = 0.0
    reranker_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "title": self.title,
            "url": self.url,
            "hybrid_score": self.hybrid_score,
            "reranker_score": self.reranker_score,
        }


@dataclass
class Citation:
    """Represents a validated citation mapped to human-readable numeric citation number ([1], [2], ...)."""
    citation_number: int
    evidence_ids: List[str]
    source: str
    title: str
    url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "citation_number": self.citation_number,
            "evidence_ids": self.evidence_ids,
            "source": self.source,
            "title": self.title,
            "url": self.url,
        }


@dataclass
class ValidationResult:
    """Result container for citation extraction, validation, marker replacement, and source list generation."""
    cleaned_answer: str
    valid_citations: List[Citation] = field(default_factory=list)
    invalid_citation_ids: List[str] = field(default_factory=list)
    sources_list: List[Dict[str, Any]] = field(default_factory=list)
    formatted_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cleaned_answer": self.cleaned_answer,
            "valid_citations": [c.to_dict() for c in self.valid_citations],
            "invalid_citation_ids": self.invalid_citation_ids,
            "sources_list": self.sources_list,
            "formatted_text": self.formatted_text,
        }
