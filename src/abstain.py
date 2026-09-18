"""Deterministic abstention gate for hallucination control."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.bm25 import tokenize
from src.config import ABSTAIN_MESSAGE, ABSTAIN_SCORE_THRESHOLD, SUPPORT_TOKEN_COVERAGE
from src.corpus_policy import Document
from src.rerank import extract_entities
from src.retrieve import RetrievalHit

# Stopwords / interrogatives that should not drive support coverage
_STOP = {
    "what",
    "which",
    "when",
    "where",
    "who",
    "whom",
    "whose",
    "why",
    "how",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "for",
    "and",
    "or",
    "with",
    "by",
    "from",
    "at",
    "as",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "do",
    "does",
    "did",
    "should",
    "would",
    "could",
    "can",
    "may",
    "might",
    "must",
    "about",
    "into",
    "than",
    "then",
    "also",
    "any",
    "all",
    "please",
    "tell",
    "me",
    "give",
}


@dataclass
class GateResult:
    abstain: bool
    reason: str
    message: str
    score: float
    support_coverage: float


def _content_tokens(query: str) -> list[str]:
    return [t for t in tokenize(query) if t not in _STOP and len(t) > 1]


def support_coverage(query: str, docs: list[Document]) -> float:
    """
    Fraction of *focus* query tokens found in retrieved texts.

    Equipment codes (P-200, C-100, …) identify which asset is meant but do not
    prove the asked attribute is present. Coverage is therefore computed primarily
    over non-entity content tokens (e.g. motor, power, oil, warranty).
    """
    ents = extract_entities(query)
    ent_toks: set[str] = set()
    for e in ents:
        ent_toks |= set(tokenize(e))

    q_toks = _content_tokens(query)
    focus = [t for t in q_toks if t not in ent_toks]
    if not focus:
        focus = q_toks
    if not focus:
        return 1.0

    blob = " ".join(d.search_text for d in docs).lower()
    d_ents: set[str] = set()
    for d in docs:
        d_ents |= extract_entities(d.search_text)

    # Entity must match when the query names one
    if ents and not (ents & d_ents):
        return 0.0

    hit = sum(1 for t in focus if t in blob)
    return hit / len(focus)


def entity_supported(query: str, docs: list[Document]) -> bool:
    ents = extract_entities(query)
    if not ents:
        return True
    d_ents: set[str] = set()
    for d in docs:
        d_ents |= extract_entities(d.search_text)
    return bool(ents & d_ents)


def decide_abstain(query: str, hits: list[RetrievalHit]) -> GateResult:
    if not hits:
        return GateResult(
            abstain=True,
            reason="empty_retrieval",
            message=ABSTAIN_MESSAGE,
            score=0.0,
            support_coverage=0.0,
        )

    best = hits[0].score
    docs = [h.doc for h in hits]
    cov = support_coverage(query, docs)

    if best < ABSTAIN_SCORE_THRESHOLD:
        return GateResult(
            abstain=True,
            reason=f"score_below_threshold({best:.3f}<{ABSTAIN_SCORE_THRESHOLD})",
            message=ABSTAIN_MESSAGE,
            score=best,
            support_coverage=cov,
        )

    if not entity_supported(query, docs):
        return GateResult(
            abstain=True,
            reason="query_entities_missing_from_hits",
            message=ABSTAIN_MESSAGE,
            score=best,
            support_coverage=cov,
        )

    if cov < SUPPORT_TOKEN_COVERAGE:
        return GateResult(
            abstain=True,
            reason=f"low_support_coverage({cov:.2f}<{SUPPORT_TOKEN_COVERAGE})",
            message=ABSTAIN_MESSAGE,
            score=best,
            support_coverage=cov,
        )

    return GateResult(
        abstain=False,
        reason="ok",
        message="",
        score=best,
        support_coverage=cov,
    )


def looks_like_abstain(text: str) -> bool:
    t = (text or "").strip().lower()
    return t.startswith("not found in the document") or "not found in the documents" in t
