"""Run `streamlit run app.py` from this folder."""

from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st

from rag_engine import (
    KnowledgeDocument,
    ask,
    build_index,
    document_inventory_answer,
    is_document_inventory_question,
    read_document,
    sample_documents,
)

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Bangkok Business RAG", page_icon="📊", layout="wide")
st.title("📊 Bangkok Business Knowledge Assistant")
st.caption("Business Analytics & Decision-Making | Retrieval-Augmented Generation (RAG) | Ollama + Codex-ready project")
st.info("The included Bangkok cafe documents are fictional teaching examples, not real sales or survey data.")

with st.sidebar:
    st.header("Knowledge sources")
    include_sample = st.checkbox("Include 3 sample cafe texts", value=True)
    uploads = st.file_uploader("Add your own files", type=["txt", "md", "pdf"], accept_multiple_files=True)
    st.caption("Supported: text-based PDF, UTF-8 TXT, Markdown. Up to 10 MB per file.")
    st.header("Local model settings")
    chat_model = st.text_input("Ollama chat model", value="gemma3:4b")
    embedding_model = st.text_input("Ollama embedding model", value="embeddinggemma")
    top_k = st.slider("Retrieved excerpts", 1, 8, 4)

try:
    documents: list[KnowledgeDocument] = sample_documents(ROOT / "knowledge") if include_sample else []
    for item in uploads or []:
        if item.size > 10 * 1024 * 1024:
            st.warning(f"Skipped {item.name}: larger than 10 MB.")
            continue
        try:
            documents.append(read_document(item.name, item.getvalue()))
        except Exception as exc:
            # A malformed PDF may raise a parser-specific exception.
            st.warning(f"Skipped {item.name}: {exc}")
except Exception as exc:
    st.error(f"Could not load the demo files: {exc}")
    st.stop()

file_signature = hashlib.sha256(
    (embedding_model + "\0" + "\0".join(f"{d.origin}|{d.name}|{d.text}" for d in documents)).encode("utf-8")
).hexdigest()

left, right = st.columns([3, 2])
with left:
    st.subheader("1. Your available texts")
    if documents:
        for doc in documents:
            with st.expander(f"{doc.name} · {doc.origin}"):
                st.text(doc.text[:3500] + ("\n... (preview truncated)" if len(doc.text) > 3500 else ""))
    else:
        st.warning("No documents loaded. Include the sample files or upload a document.")
    st.markdown("**Answer to the assignment question:**")
    st.text(document_inventory_answer(documents))

    st.subheader("2. Build your local RAG index")
    if st.button("Build / refresh index", type="primary", disabled=not documents):
        try:
            with st.spinner("Ollama is converting documents into vector embeddings..."):
                index = build_index(documents, model=embedding_model)
                st.session_state["rag_index"] = index
                st.session_state["rag_signature"] = file_signature
            st.success(f"Indexed {len(index.chunks)} searchable excerpts from {len(index.documents)} documents.")
        except Exception as exc:
            st.error(str(exc))
            st.info("Check that Ollama is running and that you have pulled the embedding model.")

with right:
    st.subheader("3. Ask a business question")
    st.caption("Try a real RAG question to retrieve evidence before the model answers.")
    with st.form("question_form"):
        question = st.text_area("Your question", value="What is the most important problem for the cafe, and which low-cost pilot should management test?", height=110)
        submit = st.form_submit_button("Ask using local knowledge")

    current_index = st.session_state.get("rag_index")
    fresh_index = current_index is not None and st.session_state.get("rag_signature") == file_signature
    if current_index is not None and not fresh_index:
        st.warning("Source documents or embedding model changed. Rebuild the index before asking a RAG question.")

    if submit:
        if is_document_inventory_question(question):
            st.markdown("**Answer**")
            st.text(document_inventory_answer(documents))
        elif not fresh_index:
            st.warning("Click **Build / refresh index** first.")
        else:
            try:
                with st.spinner("Retrieving passages and asking Ollama..."):
                    answer, retrieved = ask(current_index, question, chat_model=chat_model, top_k=top_k)
                st.markdown("**Answer grounded in local texts**")
                st.markdown(answer)
                with st.expander("View retrieved evidence and similarity scores"):
                    for i, (chunk, score) in enumerate(retrieved, start=1):
                        st.markdown(f"**[{i}] {chunk.source} — chunk {chunk.part}** (cosine similarity: {score:.3f})")
                        st.write(chunk.text)
            except Exception as exc:
                st.error(str(exc))

st.divider()
st.caption("This prototype creates embeddings locally through Ollama. Codex CLI can be used to inspect or modify this codebase; it is not required to run the app.")
