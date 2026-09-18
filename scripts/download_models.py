"""Download embedding + generation models into ./models for offline use."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    EMBED_MODEL_DIR,
    EMBED_MODEL_ID,
    LLM_MODEL_DIR,
    LLM_MODEL_ID,
    MODELS_DIR,
)


def download_embed(force: bool = False) -> None:
    from sentence_transformers import SentenceTransformer

    if EMBED_MODEL_DIR.exists() and any(EMBED_MODEL_DIR.iterdir()) and not force:
        print(f"Embed model already present at {EMBED_MODEL_DIR}")
        return
    print(f"Downloading {EMBED_MODEL_ID} -> {EMBED_MODEL_DIR}")
    EMBED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(EMBED_MODEL_ID)
    model.save(str(EMBED_MODEL_DIR))
    print("Embed model saved.")


def download_llm(force: bool = False) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if LLM_MODEL_DIR.exists() and any(LLM_MODEL_DIR.iterdir()) and not force:
        print(f"LLM already present at {LLM_MODEL_DIR}")
        return
    print(f"Downloading {LLM_MODEL_ID} -> {LLM_MODEL_DIR}")
    LLM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    tokenizer.save_pretrained(str(LLM_MODEL_DIR))
    model.save_pretrained(str(LLM_MODEL_DIR))
    print("LLM saved.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download local models for offline RAG.")
    parser.add_argument("--embed-only", action="store_true", help="Download MiniLM only")
    parser.add_argument("--llm-only", action="store_true", help="Download Qwen3-1.7B-Base only")
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    do_embed = not args.llm_only
    do_llm = not args.embed_only
    if do_embed:
        download_embed(force=args.force)
    if do_llm:
        download_llm(force=args.force)
    print("Done.")


if __name__ == "__main__":
    main()
