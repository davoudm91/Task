#!/usr/bin/env python3
"""CLI for the improved offline RAG assistant."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CORPUS_PATH
from src.pipeline import RagPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Document-grounded QA over corpus.jsonl")
    parser.add_argument("query", nargs="?", help="Question to answer")
    parser.add_argument("--corpus", default=str(CORPUS_PATH), help="Path to corpus.jsonl")
    parser.add_argument("--llm", action="store_true", help="Use Qwen3-1.7B-Base for generation")
    parser.add_argument("--no-llm", action="store_true", help="Force extractive answers (default)")
    parser.add_argument("--show-hits", action="store_true", help="Print retrieval hits")
    args = parser.parse_args()

    use_llm = bool(args.llm) and not args.no_llm
    pipe = RagPipeline.create(corpus_path=args.corpus, use_llm=use_llm, load_llm=use_llm)

    if not args.query:
        examples = [
            "What is the rated output of the C-100 compressor?",
            "How often should temperature sensors be calibrated?",
            "What is the motor power of pump P-200?",
            "What is the maximum operating pressure of the P-200?",
        ]
        for q in examples:
            result = pipe.answer(q)
            print("Q:", q)
            print("A:", result.answer)
            if args.show_hits:
                for h in result.hits:
                    print(f"  hit {h.doc.id} score={h.score:.3f}")
            print("-" * 60)
        return

    result = pipe.answer(args.query)
    print("Q:", result.query)
    print("A:", result.answer)
    if args.show_hits:
        print("abstain:", result.abstained, result.gate.reason)
        for h in result.hits:
            print(f"  hit {h.doc.id} score={h.score:.3f} title={h.doc.title}")


if __name__ == "__main__":
    main()
