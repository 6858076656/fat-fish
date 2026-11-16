"""Optional command-line interface for testing the same RAG system without a web UI."""

import argparse
from pathlib import Path
from rag_engine import ask, build_index, document_inventory_answer, read_document, sample_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Bangkok Business local-knowledge RAG")
    parser.add_argument("--list", action="store_true", help="Print exact document inventory (no model required)")
    parser.add_argument("--ask", help="Question to answer using RAG")
    parser.add_argument("--file", action="append", type=Path, default=[], help="Additional TXT/MD/PDF (repeatable)")
    parser.add_argument("--chat-model", default="gemma3:4b")
    parser.add_argument("--embedding-model", default="embeddinggemma")
    args = parser.parse_args()

    docs = sample_documents(Path(__file__).parent / "knowledge")
    for path in args.file:
        docs.append(read_document(path.name, path.read_bytes()))
    if args.list or not args.ask:
        print(document_inventory_answer(docs))
    if args.ask:
        index = build_index(docs, model=args.embedding_model)
        answer, evidence = ask(index, args.ask, chat_model=args.chat_model)
        print("\nAnswer:\n" + answer)
        if evidence:
            print("\nRetrieved sources:")
            for i, (chunk, score) in enumerate(evidence, 1):
                print(f"[{i}] {chunk.source} (chunk {chunk.part}, similarity {score:.3f})")


if __name__ == "__main__":
    main()
