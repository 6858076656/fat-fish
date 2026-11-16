"""Mock-based unit tests; no Ollama server is needed to run these."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))
import rag_engine as rag
import numpy as np
import pytest


DATA_DIR = Path(__file__).parents[1] / "knowledge"


def test_sample_files_and_inventory():
    docs = rag.sample_documents(DATA_DIR)
    assert len(docs) == 3
    text = rag.document_inventory_answer(docs)
    assert "01_sales_performance.txt" in text
    assert "02_customer_feedback.txt" in text
    assert "03_operations_decision_rules.txt" in text
    assert "FICTIONAL" in docs[0].text
    assert "bundled fictional sample" in text


def test_chunking_keeps_context_and_sources():
    document = rag.KnowledgeDocument("test.txt", " ".join(str(x) for x in range(15)))
    chunks = rag.chunk_document(document, chunk_words=10, overlap_words=3)
    assert len(chunks) == 2
    assert chunks[0].source == "test.txt"
    assert chunks[0].text.split()[-3:] == chunks[1].text.split()[:3]


def test_retrieval_uses_cosine_similarity(monkeypatch):
    docs = [rag.KnowledgeDocument("sales.txt", "sales revenue growth"), rag.KnowledgeDocument("staff.txt", "staff queue waiting")]
    def fake_embedding(texts, model="embeddinggemma", server=rag.DEFAULT_OLLAMA_URL):
        def v(text):
            return [1, 0] if "revenue" in text or "sales" in text else [0, 1]
        return np.asarray([v(t) for t in texts], dtype=np.float32)
    monkeypatch.setattr(rag, "embed_texts", fake_embedding)
    index = rag.build_index(docs)
    results = rag.retrieve(index, "Which revenue source?", top_k=1)
    assert results[0][0].source == "sales.txt"
    assert results[0][1] == pytest.approx(1)


def test_inventory_question_does_not_call_model(monkeypatch):
    docs = rag.sample_documents(DATA_DIR)
    index = rag.RAGIndex(docs, [], np.empty((0, 2)), "embeddinggemma")
    monkeypatch.setattr(rag, "ollama_request", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("unexpected model call")))
    answer, sources = rag.ask(index, "What are the texts you uploaded in your platform?")
    assert "03_operations_decision_rules.txt" in answer
    assert sources == []


def test_ask_uses_retrieved_evidence_and_discloses_fiction(monkeypatch):
    docs = [rag.KnowledgeDocument("operations.txt", "Spend up to THB 8,000 on a pilot.")]
    def fake_embedding(texts, model="embeddinggemma", server=rag.DEFAULT_OLLAMA_URL):
        return np.ones((len(texts), 3), dtype=np.float32)
    monkeypatch.setattr(rag, "embed_texts", fake_embedding)
    saved = []
    def fake_request(endpoint, payload, server=rag.DEFAULT_OLLAMA_URL):
        saved.append((endpoint, payload))
        return {"message": {"content": "Test a pilot within THB 8,000 [1]."}}
    monkeypatch.setattr(rag, "ollama_request", fake_request)
    result, sources = rag.ask(rag.build_index(docs), "What is the trial budget?")
    assert "THB 8,000" in result
    assert len(sources) == 1
    assert "operations.txt" in saved[0][1]["messages"][1]["content"]
    assert "fictional" in saved[0][1]["messages"][0]["content"].lower()


def test_reject_empty_or_invalid_text():
    with pytest.raises(ValueError):
        rag.read_document("empty.txt", b" \n ")
    with pytest.raises(ValueError):
        rag.read_document("bad.docx", b"hi")
    with pytest.raises(ValueError):
        rag.chunk_document(rag.KnowledgeDocument("x", "words"), chunk_words=5, overlap_words=5)
