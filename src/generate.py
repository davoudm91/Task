"""Grounded answer generation with Qwen3-1.7B-Base and extractive fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from src.config import LLM_MODEL_DIR, LLM_MODEL_ID, MAX_NEW_TOKENS
from src.corpus_policy import Document, detect_pressure_conflict, expand_near_dups


@dataclass
class Generator:
    tokenizer: object | None = None
    model: object | None = None
    device: str = "cpu"

    @property
    def available(self) -> bool:
        return self.model is not None and self.tokenizer is not None


def load_generator(prefer_llm: bool = True) -> Generator:
    if not prefer_llm:
        return Generator()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        return Generator()

    model_path = str(LLM_MODEL_DIR) if LLM_MODEL_DIR.exists() and any(LLM_MODEL_DIR.iterdir()) else LLM_MODEL_ID
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        return Generator(tokenizer=tokenizer, model=model, device=device)
    except Exception as exc:  # noqa: BLE001 — fallback is intentional for offline/limited hosts
        print(f"[generate] LLM unavailable ({exc}); using extractive fallback.")
        return Generator()


def _citations(docs: Sequence[Document]) -> str:
    ids = expand_near_dups([d.id for d in docs])
    return ", ".join(f"[{i}]" for i in ids)


def extractive_answer(query: str, docs: Sequence[Document]) -> str:
    """Deterministic answer from retrieved passages (no LLM)."""
    docs_list = list(docs)
    conflict = detect_pressure_conflict(docs_list)
    # Prefer conflict disclosure when the query is about pressure / max operating pressure
    q_lower = query.lower()
    if conflict and ("pressure" in q_lower or "bar" in q_lower):
        return f"{conflict} Citations: {_citations(docs_list)}"

    # Use first 1–2 passages as grounded extract
    parts = []
    for d in docs_list[:2]:
        parts.append(f"[{d.id}] {d.text}")
    cite = _citations(docs_list[:2])
    body = " ".join(parts)
    return f"{body} Citations: {cite}"


def _build_prompt(query: str, docs: Sequence[Document]) -> str:
    context_blocks = []
    for d in docs[:3]:
        context_blocks.append(f"[{d.id}] {d.title}\n{d.text}")
    context = "\n\n".join(context_blocks)
    # Completion-style prompt for a base (non-instruct) model
    return (
        "Documents:\n"
        f"{context}\n\n"
        "Answer the question using only the documents above. "
        "If the documents conflict, state the conflict and cite both. "
        "Do not invent facts.\n"
        f"Question: {query}\n"
        "Answer:"
    )


def generate_answer(
    query: str,
    docs: Sequence[Document],
    generator: Generator | None = None,
    use_llm: bool = True,
) -> str:
    docs_list = list(docs)
    if not docs_list:
        from src.config import ABSTAIN_MESSAGE

        return ABSTAIN_MESSAGE

    conflict = detect_pressure_conflict(docs_list)
    if conflict and ("pressure" in query.lower() or "bar" in query.lower()):
        return f"{conflict} Citations: {_citations(docs_list)}"

    if not use_llm or generator is None or not generator.available:
        return extractive_answer(query, docs_list)

    prompt = _build_prompt(query, docs_list)
    tokenizer = generator.tokenizer
    model = generator.model
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(generator.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    # Take first paragraph / line to limit base-model rambling
    for sep in ("\n\n", "\n"):
        if sep in text:
            text = text.split(sep)[0].strip()
    if not text:
        return extractive_answer(query, docs_list)
    cite = _citations(docs_list)
    if "DOC-" not in text:
        text = f"{text} Citations: {cite}"
    return text
