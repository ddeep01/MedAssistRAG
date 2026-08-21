import re
import logging
from typing import List, Dict, Any, Optional, Set, Tuple

from src.citations.models import Evidence, Citation, ValidationResult
from src.config import load_citations_config

logger = logging.getLogger("MedAssistRAG.CitationManager")


class CitationManager:
    """
    Manages evidence object creation, prompt evidence formatting, citation extraction,
    validation of LLM citation markers, consolidation of document sources, and source list generation.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = load_citations_config(config_path)
        c_cfg = self.config.get("citations", {})
        self.enabled = c_cfg.get("enabled", True)
        self.max_evidence_chunks = int(c_cfg.get("max_evidence_chunks", 5))
        self.strip_invalid_citations = c_cfg.get("strip_invalid_citations", True)

    def create_evidence_objects(self, candidate_results: List[Dict[str, Any]]) -> List[Evidence]:
        """
        Assigns temporary unique evidence IDs (E1, E2, E3, ...) to retrieved candidates
        and preserves metadata required for source attribution.
        """
        evidence_list = []
        if not candidate_results:
            return []

        for i, item in enumerate(candidate_results[:self.max_evidence_chunks], start=1):
            e_id = f"E{i}"
            doc_id = str(item.get("document_id", item.get("doc_id", f"doc_{i}")))
            chunk_id = str(item.get("chunk_id", f"{doc_id}_chunk_{i}"))
            text = str(item.get("text", "")).strip()

            # Metadata fallback handling
            source = str(item.get("source", item.get("publisher", ""))).strip()
            title = str(item.get("title", item.get("doc_title", ""))).strip()
            url = item.get("url", item.get("link", None))
            if url:
                url = str(url).strip()

            hybrid_score = float(item.get("hybrid_score", item.get("dense_score", 0.0)))
            reranker_score = float(item.get("reranker_score", 0.0))

            ev = Evidence(
                evidence_id=e_id,
                document_id=doc_id,
                chunk_id=chunk_id,
                text=text,
                source=source,
                title=title,
                url=url,
                hybrid_score=hybrid_score,
                reranker_score=reranker_score
            )
            evidence_list.append(ev)

        return evidence_list

    def build_evidence_context(self, evidence_list: List[Evidence]) -> str:
        """
        Formats structured medical evidence context block for LLM prompt.
        Each evidence item is tagged with its temporary ID [E1], [E2], etc.
        """
        if not evidence_list:
            return "No medical evidence available."

        blocks = ["MEDICAL EVIDENCE:\n"]
        for ev in evidence_list:
            source_str = ev.source if ev.source else "Medical Source"
            title_str = f" ({ev.title})" if ev.title else ""

            blocks.append(
                f"[{ev.evidence_id}]\n"
                f"Source: {source_str}{title_str}\n"
                f"Content:\n{ev.text}\n"
            )

        return "\n".join(blocks)

    def extract_citations(self, text: str) -> List[str]:
        """Extracts raw citation markers (e.g. ['E1', 'E2', 'E99']) from generated answer text."""
        if not text or not isinstance(text, str):
            return []
        matches = re.findall(r"\[E(\d+)\]", text)
        return [f"E{m}" for m in matches]

    def format_source_label(self, source: str, title: str) -> str:
        """Generates clean human-readable source label with safe metadata fallbacks."""
        if source and title:
            return f"{source} — {title}"
        elif title:
            return title
        elif source:
            return source
        else:
            return "Retrieved medical evidence"

    def validate_and_format_citations(
        self,
        raw_answer: str,
        evidence_list: List[Evidence]
    ) -> ValidationResult:
        """
        Extracts citation markers, validates them against evidence_list, strips invalid IDs,
        maps valid [E1] markers to human-readable numeric citations [1], consolidates duplicate
        sources, and appends formatted Source List.
        """
        if not raw_answer or not isinstance(raw_answer, str):
            return ValidationResult(cleaned_answer="", formatted_text="")

        if not self.enabled or not evidence_list:
            # Strip any hallucinated [E#] markers if no evidence or citations disabled
            cleaned = re.sub(r"\[E\d+\]", "", raw_answer).strip()
            return ValidationResult(
                cleaned_answer=cleaned,
                formatted_text=cleaned
            )

        # Map evidence_id -> Evidence object
        evidence_map = {ev.evidence_id: ev for ev in evidence_list}
        valid_ids = set(evidence_map.keys())

        # Extract all referenced IDs in order
        all_referenced_ids = self.extract_citations(raw_answer)

        valid_referenced_ids = []
        invalid_ids = []

        for e_id in all_referenced_ids:
            if e_id in valid_ids:
                if e_id not in valid_referenced_ids:
                    valid_referenced_ids.append(e_id)
            else:
                if e_id not in invalid_ids:
                    invalid_ids.append(e_id)

        # Consolidate evidence items from the same document
        # Document key: (document_id, title, source)
        doc_key_to_num: Dict[Tuple[str, str, str], int] = {}
        e_id_to_num: Dict[str, int] = {}
        citations_list: List[Citation] = []
        sources_meta: List[Dict[str, Any]] = []

        num_counter = 1
        for e_id in valid_referenced_ids:
            ev = evidence_map[e_id]
            doc_key = (ev.document_id, ev.title, ev.source)

            if doc_key not in doc_key_to_num:
                doc_key_to_num[doc_key] = num_counter
                e_id_to_num[e_id] = num_counter

                source_label = self.format_source_label(ev.source, ev.title)
                c = Citation(
                    citation_number=num_counter,
                    evidence_ids=[e_id],
                    source=ev.source,
                    title=ev.title,
                    url=ev.url
                )
                citations_list.append(c)
                sources_meta.append({
                    "citation_number": num_counter,
                    "label": source_label,
                    "source": ev.source,
                    "title": ev.title,
                    "url": ev.url
                })
                num_counter += 1
            else:
                existing_num = doc_key_to_num[doc_key]
                e_id_to_num[e_id] = existing_num
                # Find existing citation object and append evidence_id
                for c in citations_list:
                    if c.citation_number == existing_num and e_id not in c.evidence_ids:
                        c.evidence_ids.append(e_id)

        # Function to replace [E#] markers in text
        def replace_marker(match: re.Match) -> str:
            raw_marker = match.group(0)  # e.g. "[E1]"
            e_num = match.group(1)
            e_id = f"E{e_num}"

            if e_id in e_id_to_num:
                return f"[{e_id_to_num[e_id]}]"
            elif self.strip_invalid_citations:
                return ""
            return raw_marker

        # Perform replacement across raw answer
        cleaned_text = re.sub(r"\[E(\d+)\]", replace_marker, raw_answer)
        # Clean up any leftover whitespace or empty brackets
        cleaned_text = re.sub(r" +", " ", cleaned_text).strip()

        # Format Source List block
        if sources_meta:
            source_lines = ["\n\nSources:"]
            for s in sources_meta:
                url_part = f" ({s['url']})" if s.get('url') else ""
                source_lines.append(f"[{s['citation_number']}] {s['label']}{url_part}")
            sources_block = "\n".join(source_lines)
            formatted_text = cleaned_text + sources_block
        else:
            formatted_text = cleaned_text

        logger.debug(
            f"[CitationManager] Extracted={len(all_referenced_ids)} Valid={len(valid_referenced_ids)} "
            f"Invalid={len(invalid_ids)} Consolidated Sources={len(sources_meta)}"
        )

        return ValidationResult(
            cleaned_answer=cleaned_text,
            valid_citations=citations_list,
            invalid_citation_ids=invalid_ids,
            sources_list=sources_meta,
            formatted_text=formatted_text
        )
