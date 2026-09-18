"""BM25 keyword retrieval over title+text."""

from __future__ import annotations

import re
from typing import Sequence

from rank_bm25 import BM25Okapi

from src.corpus_policy import Document

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?", re.I)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


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
