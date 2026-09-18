"""Light lexical re-ranking and entity boosts."""

from __future__ import annotations

import re
from typing import Sequence

from src.bm25 import tokenize
from src.corpus_policy import Document

# Equipment / error / part codes common in industrial queries
_ENTITY_RE = re.compile(
    r"\b(?:[A-Z]-?\d{2,4}|E-\d{3}|BRG-\d+|DOC-\d+|P-\d+|C-\d+|M-\d+|F-\d+)\b",
    re.I,
)


def extract_entities(text: str) -> set[str]:
    return {m.group(0).upper().replace(" ", "") for m in _ENTITY_RE.finditer(text or "")}


def lexical_overlap_score(query: str, doc: Document) -> float:
    q_toks = set(tokenize(query))
    d_toks = set(tokenize(doc.search_text))
    if not q_toks:
        return 0.0
    return len(q_toks & d_toks) / len(q_toks)


def entity_boost(query: str, doc: Document) -> float:
    q_ents = extract_entities(query)
    if not q_ents:
        return 0.0
    d_ents = extract_entities(doc.search_text)
    hits = q_ents & d_ents
    return 0.15 * len(hits)


def rerank(
    query: str,
    hits: Sequence[tuple[Document, float]],
) -> list[tuple[Document, float]]:
    """Blend RRF/fused score with lexical overlap and exact entity boosts."""
    rescored: list[tuple[Document, float]] = []
    for doc, score in hits:
        combined = float(score) + 0.35 * lexical_overlap_score(query, doc) + entity_boost(query, doc)
        rescored.append((doc, combined))
    rescored.sort(key=lambda x: x[1], reverse=True)
    return rescored
