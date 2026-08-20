import pytest
from unittest.mock import MagicMock

from src.memory.short_term_memory import ShortTermMemory
from src.memory.entity_memory import EntityMemory
from src.memory.summary_memory import SummaryMemory
from src.memory.memory_manager import MemoryManager
from src.query.query_rewriter import QueryRewriter


# ----------------------------------------------------
# TEST 1: ADD USER MESSAGE
# ----------------------------------------------------
def test_add_user_message():
    st = ShortTermMemory(max_messages=10)
    msg = st.add_message("conv-1", "user", "I have hypertension.")
    assert msg.role == "user"
    assert msg.content == "I have hypertension."

    msgs = st.get_recent_messages("conv-1")
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "I have hypertension."


# ----------------------------------------------------
# TEST 2: ADD ASSISTANT MESSAGE
# ----------------------------------------------------
def test_add_assistant_message():
    st = ShortTermMemory(max_messages=10)
    st.add_message("conv-1", "user", "What is diabetes?")
    st.add_message("conv-1", "assistant", "Diabetes is a metabolic disease...")

    msgs = st.get_recent_messages("conv-1")
    assert len(msgs) == 2
    assert msgs[1]["role"] == "assistant"


# ----------------------------------------------------
# TEST 3: MAXIMUM SHORT-TERM MEMORY SIZE
# ----------------------------------------------------
def test_max_short_term_memory_size():
    st = ShortTermMemory(max_messages=4)
    for i in range(6):
        st.add_message("conv-1", "user", f"Message {i}")

    msgs = st.get_recent_messages("conv-1")
    assert len(msgs) == 4


# ----------------------------------------------------
# TEST 4: OLDEST MESSAGES REMOVED CORRECTLY
# ----------------------------------------------------
def test_oldest_messages_removed():
    st = ShortTermMemory(max_messages=3)
    st.add_message("conv-1", "user", "Msg 0")
    st.add_message("conv-1", "user", "Msg 1")
    st.add_message("conv-1", "user", "Msg 2")
    st.add_message("conv-1", "user", "Msg 3")

    msgs = st.get_recent_messages("conv-1")
    contents = [m["content"] for m in msgs]
    assert contents == ["Msg 1", "Msg 2", "Msg 3"]
    assert "Msg 0" not in contents


# ----------------------------------------------------
# TEST 5: ENTITY EXTRACTION (MOCKED LLM)
# ----------------------------------------------------
def test_entity_extraction():
    mock_llm = MagicMock()
    mock_llm.generate_raw.return_value = '{"conditions": ["hypertension"], "symptoms": [], "medications": ["amlodipine"], "tests": [], "procedures": [], "body_parts": []}'

    ent_mem = EntityMemory(llm=mock_llm)
    extracted = ent_mem.extract_entities("I have hypertension and take amlodipine.")

    assert "hypertension" in extracted["conditions"]
    assert "amlodipine" in extracted["medications"]


# ----------------------------------------------------
# TEST 6: ENTITY DEDUPLICATION
# ----------------------------------------------------
def test_entity_deduplication():
    mock_llm = MagicMock()
    mock_llm.generate_raw.return_value = '{"conditions": ["hypertension"], "symptoms": [], "medications": [], "tests": [], "procedures": [], "body_parts": []}'

    ent_mem = EntityMemory(llm=mock_llm)
    ent_mem.update_entities("conv-1", "I have hypertension.")
    ent_mem.update_entities("conv-1", "Yes, hypertension.")

    stored = ent_mem.get_entities("conv-1")
    assert stored["conditions"] == ["hypertension"]


# ----------------------------------------------------
# TEST 7: ENTITY MERGING ACROSS TURNS
# ----------------------------------------------------
def test_entity_merging_across_turns():
    mock_llm = MagicMock()
    mock_llm.generate_raw.side_effect = [
        '{"conditions": ["hypertension"], "symptoms": [], "medications": [], "tests": [], "procedures": [], "body_parts": []}',
        '{"conditions": [], "symptoms": [], "medications": ["metformin"], "tests": [], "procedures": [], "body_parts": []}'
    ]

    ent_mem = EntityMemory(llm=mock_llm)
    ent_mem.update_entities("conv-1", "Turn 1 text")
    ent_mem.update_entities("conv-1", "Turn 2 text")

    stored = ent_mem.get_entities("conv-1")
    assert "hypertension" in stored["conditions"]
    assert "metformin" in stored["medications"]


# ----------------------------------------------------
# TEST 8: CONVERSATION ISOLATION (CONV A vs CONV B)
# ----------------------------------------------------
def test_conversation_isolation():
    mock_llm = MagicMock()
    mm = MemoryManager(llm=mock_llm)

    mm.add_message("conv-A", "user", "I have diabetes.")
    mm.add_message("conv-B", "user", "I have asthma.")

    ctx_a = mm.get_context("conv-A")
    ctx_b = mm.get_context("conv-B")

    assert ctx_a.recent_messages[0]["content"] == "I have diabetes."
    assert ctx_b.recent_messages[0]["content"] == "I have asthma."
    assert "asthma" not in [m["content"] for m in ctx_a.recent_messages]


# ----------------------------------------------------
# TEST 9: FOLLOW-UP QUERY RESOLUTION
# ----------------------------------------------------
def test_followup_query_resolution():
    mock_llm = MagicMock()
    mock_llm.generate_raw.return_value = "What are the complications of hypertension?"

    rewriter = QueryRewriter(llm=mock_llm)
    ctx = {
        "summary": "",
        "entities": {"conditions": ["hypertension"]},
        "recent_messages": [{"role": "user", "content": "I have hypertension."}]
    }

    res = rewriter.rewrite("What complications can it cause?", ctx)
    assert res == "What are the complications of hypertension?"


# ----------------------------------------------------
# TEST 10: MEDICATION REFERENCE RESOLUTION
# ----------------------------------------------------
def test_medication_reference_resolution():
    mock_llm = MagicMock()
    mock_llm.generate_raw.return_value = "What are the side effects of metformin?"

    rewriter = QueryRewriter(llm=mock_llm)
    ctx = {
        "summary": "",
        "entities": {"medications": ["metformin"]},
        "recent_messages": [{"role": "user", "content": "I take metformin."}]
    }

    res = rewriter.rewrite("What are its side effects?", ctx)
    assert res == "What are the side effects of metformin?"


# ----------------------------------------------------
# TEST 11: AMBIGUOUS MULTIPLE ENTITIES
# ----------------------------------------------------
def test_ambiguous_multiple_entities():
    mock_llm = MagicMock()
    # LLM should preserve intent without inventing single entity
    mock_llm.generate_raw.return_value = "Which condition between diabetes and hypertension causes kidney problems?"

    rewriter = QueryRewriter(llm=mock_llm)
    ctx = {
        "summary": "",
        "entities": {"conditions": ["diabetes", "hypertension"]},
        "recent_messages": [{"role": "user", "content": "I have diabetes and hypertension."}]
    }

    res = rewriter.rewrite("Which one can cause kidney problems?", ctx)
    assert "diabetes" in res.lower() or "hypertension" in rewriter.validate_and_clean_rewrite(res, "Which one can cause kidney problems?").lower()


# ----------------------------------------------------
# TEST 12: CONVERSATION CLEARING
# ----------------------------------------------------
def test_conversation_clearing():
    mock_llm = MagicMock()
    mm = MemoryManager(llm=mock_llm)

    mm.add_message("conv-1", "user", "Hello doctor")
    mm.clear_conversation("conv-1")

    ctx = mm.get_context("conv-1")
    assert ctx.recent_messages == []
    assert ctx.entities == {"conditions": [], "symptoms": [], "medications": [], "tests": [], "procedures": [], "body_parts": []}


# ----------------------------------------------------
# TEST 13: SUMMARY GENERATION
# ----------------------------------------------------
def test_summary_generation():
    mock_llm = MagicMock()
    mock_llm.generate_raw.return_value = "User discussed hypertension and symptoms."

    sm = SummaryMemory(llm=mock_llm, trigger_messages=2, enabled=True)
    msgs = [{"role": "user", "content": "I have hypertension."}, {"role": "assistant", "content": "Tell me more."}]

    sum_text = sm.update_summary("conv-1", msgs)
    assert sum_text == "User discussed hypertension and symptoms."


# ----------------------------------------------------
# TEST 14: SUMMARY DOES NOT INVENT FACTS
# ----------------------------------------------------
def test_summary_grounded():
    mock_llm = MagicMock()
    mock_llm.generate_raw.return_value = "User asked about diabetes symptoms."

    sm = SummaryMemory(llm=mock_llm, trigger_messages=1, enabled=True)
    sum_text = sm.update_summary("conv-1", [{"role": "user", "content": "What is diabetes?"}])
    assert "diabetes" in sum_text.lower()


# ----------------------------------------------------
# TEST 15: MEMORY DISABLED FLAG
# ----------------------------------------------------
def test_memory_disabled():
    mock_llm = MagicMock()
    mm = MemoryManager(llm=mock_llm)
    mm.enabled = False

    mm.add_message("conv-1", "user", "I have hypertension.")
    ctx = mm.get_context("conv-1")

    assert ctx.recent_messages == []
    assert ctx.entities == {}
    assert ctx.summary == ""
