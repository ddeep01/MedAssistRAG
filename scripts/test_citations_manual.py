import os
import sys

# Ensure workspace root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retriever.hybrid_retriever import HybridRetriever
from src.confidence.scorer import ConfidenceScorer
from src.citations.citation_manager import CitationManager
from src.query.query_rewriter import QueryRewriter
import faiss
from sentence_transformers import SentenceTransformer


def main():
    print("=" * 75)
    print("      CITATIONS & SOURCE ATTRIBUTION MANUAL INTEGRATION TEST     ")
    print("=" * 75)

    # Synthetic structured medical corpus for citation test
    corpus_items = [
        {
            "document_id": "medline_001",
            "chunk_id": "medline_001_chunk_1",
            "text": "Diabetes mellitus symptoms include frequent urination, excessive thirst, extreme hunger, unexplained weight loss, and fatigue.",
            "source": "MedlinePlus",
            "title": "Diabetes Overview",
            "url": "https://medlineplus.gov/diabetes.html"
        },
        {
            "document_id": "pubmed_102",
            "chunk_id": "pubmed_102_chunk_1",
            "text": "Type 2 diabetes complications may develop over time, including cardiovascular disease, neuropathy, nephropathy, and retinopathy.",
            "source": "PubMed",
            "title": "Complications of Diabetes Mellitus",
            "url": "https://pubmed.ncbi.nlm.nih.gov/67890"
        },
        {
            "document_id": "medline_001",
            "chunk_id": "medline_001_chunk_2",
            "text": "Blurred vision and slow-healing sores are additional early warning signs of elevated blood sugar levels.",
            "source": "MedlinePlus",
            "title": "Diabetes Overview",
            "url": "https://medlineplus.gov/diabetes.html"
        }
    ]

    corpus_texts = [item["text"] for item in corpus_items]

    # Initialize vector index and search components
    emb_model = SentenceTransformer("BAAI/bge-small-en")
    embeddings = emb_model.encode(corpus_texts, normalize_embeddings=True)
    dim = embeddings.shape[1]
    faiss_idx = faiss.IndexFlatL2(dim)
    faiss_idx.add(embeddings)

    # Wrap corpus items in retriever
    retriever = HybridRetriever(texts=corpus_texts, faiss_index=faiss_idx)
    scorer = ConfidenceScorer()
    citation_manager = CitationManager()

    query = "What are the symptoms of diabetes?"
    print(f"\nUser Query: \"{query}\"")
    print("-" * 75)

    # 1. Search structured candidates
    raw_candidates = retriever.search(query, mode="hybrid", final_top_k=3)

    # Attach metadata to candidates
    structured_candidates = []
    for idx, c in enumerate(raw_candidates):
        matched_item = corpus_items[c["index"]] if c["index"] < len(corpus_items) else corpus_items[0]
        merged = dict(c)
        merged.update({
            "document_id": matched_item["document_id"],
            "chunk_id": matched_item["chunk_id"],
            "source": matched_item["source"],
            "title": matched_item["title"],
            "url": matched_item["url"]
        })
        structured_candidates.append(merged)

    # 2. Evaluate retrieval confidence
    eval_res = scorer.evaluate_retrieval(query, structured_candidates)
    print(f"Confidence Evaluation: Level = {eval_res['level']} ({eval_res['confidence']:.4f})")

    # 3. Create evidence objects (E1, E2, ...)
    evidence_objects = citation_manager.create_evidence_objects(structured_candidates)
    print(f"\nAssigned Evidence Objects:")
    for ev in evidence_objects:
        print(f"  [{ev.evidence_id}] Doc: {ev.document_id} | Source: {ev.source} — {ev.title}")

    # 4. Format evidence context for LLM prompt
    evidence_ctx_str = citation_manager.build_evidence_context(evidence_objects)
    print(f"\nFormatted Evidence Context Prompt:")
    print("-" * 40)
    print(evidence_ctx_str.strip())
    print("-" * 40)

    # 5. Simulate raw LLM response with E# markers (including one invalid marker E99 to verify stripping)
    raw_llm_draft = (
        "Common symptoms of diabetes include frequent urination, excessive thirst, and fatigue. [E1] "
        "Blurred vision and slow-healing sores are also early warning signs. [E3] "
        "Elevated blood sugar can also cause ungrounded claims. [E99]"
    )
    print(f"\nRaw LLM Draft Output (with E# tags):\n\"{raw_llm_draft}\"")

    # 6. Validate citations, convert E1 -> [1], consolidate duplicate sources, and format Source List
    val_result = citation_manager.validate_and_format_citations(raw_llm_draft, evidence_objects)

    print(f"\nValidation Summary:")
    print(f"  Valid Citations Count:   {len(val_result.valid_citations)}")
    print(f"  Invalid IDs Stripped:    {val_result.invalid_citation_ids}")
    print(f"  Consolidated Sources:    {len(val_result.sources_list)}")

    print(f"\nFinal Validated Answer & Sources Output:\n")
    print("=" * 75)
    print(val_result.formatted_text)
    print("=" * 75)


if __name__ == "__main__":
    main()
