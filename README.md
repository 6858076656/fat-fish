# Bangkok Business Analytics — Local-Knowledge RAG

A beginner-friendly RAG prototype for **Business Analytics and Decision-Making**. The prototype uses **Ollama** for local embeddings and local text generation, and the project can be edited or reviewed using **Codex CLI**. It bundles **three fictional Bangkok cafe texts** and can ingest more documents through a Streamlit upload interface.

## 1. Install the prerequisites

Install Python 3.10+ and [Ollama](https://ollama.com/download). Make sure the Ollama app or local server is running (`ollama serve` if it is not already running).

Download the two models in your terminal:

```bash
ollama pull embeddinggemma
ollama pull gemma3:4b
```

`embeddinggemma` is an **embedding model** for finding relevant source passages. `gemma3:4b` is the local **chat model** that writes the final answer. Neither is a Codex model.

## 2. Run the project

Open a terminal inside the unzipped project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows, activate with `.venv\Scripts\activate` instead. Streamlit opens a local page in your browser, typically `http://localhost:8501`.

1. Leave **Include 3 sample cafe texts** checked.
2. Review the three document names and text previews.
3. Click **Build / refresh index**. Ollama creates text embeddings.
4. Ask: `What are the texts you uploaded in your platform?` This always lists the actual currently loaded files, without hallucinating file names.
5. Ask: `What is the most important operational problem, and what is a low-cost experiment?` This triggers semantic retrieval, shows relevant source excerpts, and asks Ollama for an evidence-grounded answer.
6. Optionally upload `.txt`, `.md`, or text-based `.pdf` files, then **rebuild the index** before asking about them.

## 3. Command-line alternative

```bash
python cli.py --list
python cli.py --ask "What should the cafe test to reduce lunchtime waits?"
```

`--list` needs Python dependencies but **does not require Ollama**. `--ask` requires the Ollama server and both models. Add a local file with `--file my_notes.txt`.

## 4. Using Codex with Ollama

OpenAI **Codex CLI** is a coding assistant, not the same thing as the Ollama embedding model. You can use it to build, review, and improve this Python project. Ollama documents direct Codex CLI integration:

```bash
npm install -g @openai/codex
ollama launch codex
```

Run those commands from the project folder. You need a **separately installed, tool-capable Ollama model** suitable for Codex (and enough local memory). The RAG app itself does **not** require Codex to remain open. See `CODEX_PROMPT.md` for a ready-to-use coding prompt. If your coursework asks you to show proof of Codex usage, actually run it and include your own screenshot or session output; these files do not prove that Codex was run.

Official docs: https://docs.ollama.com/integrations/codex and https://docs.ollama.com/capabilities/embeddings

## 5. Project files

| File | Purpose |
|---|---|
| `app.py` | Streamlit upload and question-answer interface |
| `rag_engine.py` | Text extraction, chunking, Ollama embeddings, retrieval, grounded answering |
| `cli.py` | Terminal listing and RAG query commands |
| `knowledge/01_sales_performance.txt` | Fictional cafe sales and analytics |
| `knowledge/02_customer_feedback.txt` | Fictional customer feedback |
| `knowledge/03_operations_decision_rules.txt` | Fictional staffing, budget and metrics |
| `assignment_answer.md` | Ready-to-edit explanation answering the assignment question |
| `CODEX_PROMPT.md` | Prompt to try in Codex CLI |
| `tests/test_rag.py` | Unit tests using mock Ollama responses |

## 6. How it works

```text
TXT / Markdown / text PDF
        ↓
Text extraction and overlapping chunks
        ↓
Ollama /api/embed (embeddinggemma)
        ↓
Vectors + cosine-similarity search ← Question embedding
        ↓
Top relevant excerpts → Grounded prompt
        ↓
Ollama /api/chat (gemma3:4b)
        ↓
Answer with excerpt citations [1], [2], ...
```

A special **document inventory** question is answered directly from file metadata rather than sending it to the language model. This prevents the model from making up uploaded filenames.

## 7. Notes / limitations

- All bundled cafe data are **fictional** and intended for classroom demonstration only.
- Local Ollama mode does not send text to an OpenAI endpoint. Uploaded files are held in the Streamlit session and their text is processed by Ollama locally.
- The code is a learning prototype, not production-ready access-controlled infrastructure. Be careful with private data in a shared app.
- Image-only scanned PDFs need OCR beforehand.
- There is no external internet research, actual customer records, verified market analysis, or statistical inference.
- API requests will fail unless Ollama is running and the named models are installed; the supplied automated tests simulate the Ollama calls and do not prove an end-to-end live model session.

## 8. Run the automated tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q tests
```

The tests check document inventory, chunking, cosine similarity retrieval, evidence grounding, and input validation without a running Ollama service.
