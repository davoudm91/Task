"""BM25 keyword retrieval over title+text."""

from __future__ import annotations

import re
from typing import Sequence

from rank_bm25 import BM25Okapi

from src.corpus_policy import Document

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?", re.I)

_BM25_STOP = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "by",
    "from",
    "at",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
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
    "how",
    "what",
    "which",
    "when",
    "where",
    "who",
    "why",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "into",
    "than",
    "then",
    "also",
    "any",
    "all",
    "often",
    "about",
}


def _light_stem(token: str) -> str:
    t = token.lower()
    for suffix in ("ation", "ing", "ed", "es", "s", "ly"):
        if t.endswith(suffix) and len(t) > len(suffix) + 2:
            stem = t[: -len(suffix)]
            if suffix == "ation":
                return stem + "ate" if not stem.endswith("ate") else stem
            return stem + "e" if suffix == "ed" and not stem.endswith("e") else stem
    return t


def tokenize(text: str) -> list[str]:
    raw = [t.lower() for t in _TOKEN_RE.findall(text or "")]
    out: list[str] = []
    for t in raw:
        if t in _BM25_STOP or len(t) <= 1:
            continue
        out.append(_light_stem(t))
    return out


class BM25Index:
    def __init__(self, docs: Sequence[Document]):
        self.docs = list(docs)
        self._corpus_tokens = [tokenize(d.search_text) for d in self.docs]
        self._bm25 = BM25Okapi(self._corpus_tokens)

    def search(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: list[tuple[Document, float]] = []
        for i in order[:k]:
            if scores[i] <= 0:
                break
            out.append((self.docs[i], float(scores[i])))
        return out
