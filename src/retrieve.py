"""Hybrid dense + BM25 retrieval with RRF fusion and light re-rank."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from src.bm25 import BM25Index
from src.config import K_BM25, K_DENSE, RRF_K, TOP_N
from src.corpus_policy import Document, load_docs
from src.embed import encode_texts, load_embed_model
from src.rerank import rerank


@dataclass
class RetrievalHit:
    doc: Document
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None


@dataclass
class RetrievalIndex:
    docs: list[Document]
    vectors: np.ndarray
    embed_model: SentenceTransformer
    bm25: BM25Index


def build_index(corpus_path: str, embed_model: SentenceTransformer | None = None) -> RetrievalIndex:
    docs = load_docs(corpus_path)
    model = embed_model or load_embed_model()
    vectors = encode_texts(model, [d.search_text for d in docs])
    return RetrievalIndex(docs=docs, vectors=vectors, embed_model=model, bm25=BM25Index(docs))


def _rrf_fuse(
    dense_hits: list[tuple[Document, float]],
    bm25_hits: list[tuple[Document, float]],
    rrf_k: int = RRF_K,
) -> list[tuple[Document, float, int | None, int | None]]:
    scores: dict[str, float] = {}
    docs_by_id: dict[str, Document] = {}
    dense_rank: dict[str, int] = {}
    bm25_rank: dict[str, int] = {}

    for rank, (doc, _) in enumerate(dense_hits, start=1):
        docs_by_id[doc.id] = doc
        dense_rank[doc.id] = rank
        scores[doc.id] = scores.get(doc.id, 0.0) + 1.0 / (rrf_k + rank)

    for rank, (doc, _) in enumerate(bm25_hits, start=1):
        docs_by_id[doc.id] = doc
        bm25_rank[doc.id] = rank
        scores[doc.id] = scores.get(doc.id, 0.0) + 1.0 / (rrf_k + rank)

    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        (docs_by_id[did], score, dense_rank.get(did), bm25_rank.get(did))
        for did, score in ordered
    ]


def dense_search(
    query: str,
    index: RetrievalIndex,
    k: int = K_DENSE,
) -> list[tuple[Document, float]]:
    q = encode_texts(index.embed_model, [query])[0]
    sims = index.vectors @ q
    order = np.argsort(-sims)[:k]
    return [(index.docs[int(i)], float(sims[int(i)])) for i in order]


def retrieve(
    query: str,
    index: RetrievalIndex,
    k_dense: int = K_DENSE,
    k_bm25: int = K_BM25,
    top_n: int = TOP_N,
) -> list[RetrievalHit]:
    dense_hits = dense_search(query, index, k=k_dense)
    bm25_hits = index.bm25.search(query, k=k_bm25)
    fused = _rrf_fuse(dense_hits, bm25_hits)
    # Map fused RRF score into re-rank input
    pre = [(doc, score) for doc, score, _, _ in fused]
    ranked = rerank(query, pre)[:top_n]

    meta = {doc.id: (d_r, b_r) for doc, _, d_r, b_r in fused}
    hits: list[RetrievalHit] = []
    for doc, score in ranked:
        d_r, b_r = meta.get(doc.id, (None, None))
        hits.append(RetrievalHit(doc=doc, score=float(score), dense_rank=d_r, bm25_rank=b_r))
    return hits


def retrieve_docs(query: str, index: RetrievalIndex, top_n: int = TOP_N) -> list[Document]:
    return [h.doc for h in retrieve(query, index, top_n=top_n)]
