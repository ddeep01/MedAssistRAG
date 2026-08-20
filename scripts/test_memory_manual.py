import os
import sys

# Ensure workspace root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.memory.memory_manager import MemoryManager
from src.retriever.hybrid_retriever import HybridRetriever
from src.confidence.scorer import ConfidenceScorer
from src.query.query_rewriter import QueryRewriter
import faiss
from sentence_transformers import SentenceTransformer


def main():
    print("=" * 70)
    print("      CONVERSATION MEMORY MANUAL MULTI-TURN INTEGRATION TEST     ")
    print("=" * 70)

    # Synthetic medical corpus for conversational RAG test
    corpus_texts = [
        "Hypertension (high blood pressure) is a chronic medical condition in which the blood pressure in the arteries is persistently elevated.",
        "Symptoms of hypertension include headache, shortness of breath, dizziness, and nosebleeds, though it is often asymptomatic.",
        "Hypertension complications include coronary artery disease, stroke, heart failure, peripheral vascular disease, vision loss, and chronic kidney disease.",
        "Treatment of hypertension includes lifestyle changes, salt reduction, exercise, and antihypertensive medications like lisinopril, amlodipine, or hydrochlorothiazide.",
        "Common side effects of antihypertensive medications like lisinopril include dry cough, dizziness, elevated potassium levels, and headache."
    ]

    # Initialize vector index and search components
    emb_model = SentenceTransformer("BAAI/bge-small-en")
    embeddings = emb_model.encode(corpus_texts, normalize_embeddings=True)
    dim = embeddings.shape[1]
    faiss_idx = faiss.IndexFlatL2(dim)
    faiss_idx.add(embeddings)

    retriever = HybridRetriever(texts=corpus_texts, faiss_index=faiss_idx)
    scorer = ConfidenceScorer()
    rewriter = QueryRewriter()
    memory_manager = MemoryManager()

    cid = memory_manager.create_conversation("test-001")

    turns = [
        "What is hypertension?",
        "What are its symptoms?",
        "What complications can it cause?",
        "How is it treated?",
        "What about the medication side effects?"
    ]

    for turn_num, orig_query in enumerate(turns, start=1):
        print(f"\n======================================================================")
        print(f"TURN {turn_num} | Conversation ID: {cid}")
        print(f"Original Query: \"{orig_query}\"")
        print("======================================================================")

        # 1. Get prior memory context
        prior_context = memory_manager.get_context(cid)

        # 2. Record user query in memory
        memory_manager.add_message(cid, "user", orig_query)

        # 3. Extract and get updated entities
        current_entities = memory_manager.entity_memory.get_entities(cid)

        # 4. Generate standalone query if prior memory exists
        if prior_context.recent_messages or prior_context.entities:
            standalone_query = rewriter.rewrite(orig_query, prior_context)
        else:
            standalone_query = orig_query

        # 5. Perform Hybrid Retrieval + Confidence Evaluation
        candidates = retriever.search(standalone_query, mode="hybrid", final_top_k=3)
        eval_res = scorer.evaluate_retrieval(standalone_query, candidates)

        # 6. Generate answer string from retrieved evidence
        if eval_res["level"] != "LOW" and candidates:
            top_evidence = candidates[0]["text"]
            answer = f"Based on retrieved evidence: {top_evidence}"
        else:
            answer = "Insufficient evidence found to answer the query with confidence."

        # 7. Record assistant response in memory
        memory_manager.add_message(cid, "assistant", answer)

        # Display Turn Output Trace
        print(f"  [Prior Messages Count]:     {len(prior_context.recent_messages)}")
        print(f"  [Tracked Entities]:         {current_entities}")
        print(f"  [Rewritten Standalone Query]: \"{standalone_query}\"")
        print(f"  [Confidence Level]:         {eval_res['level']} ({eval_res['confidence']})")
        if candidates:
            print(f"  [Top Evidence Snippet]:     {candidates[0]['text'][:100]}...")
        print(f"  [Final Answer Output]:      {answer[:120]}...")
        print("-" * 70)


if __name__ == "__main__":
    main()
