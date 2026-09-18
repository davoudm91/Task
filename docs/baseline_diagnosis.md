# Baseline diagnosis notes

Observed weaknesses in `baseline_rag.py` (kept unchanged for comparison):

1. **Top-1 only** — `np.argmax` returns a single chunk; no multi-hit or re-ranking.
2. **Dense-only** — no keyword/BM25 path; brittle on exact codes (`E-207`, `BRG-4410`).
3. **Chunking no-op** — `CHUNK_SIZE=400` on short passages ≈ one chunk per doc; title never indexed.
4. **No abstention** — always returns the best chunk, so unanswerable questions are hallucinated as grounded.
5. **No conflict / near-dup policy** — DOC-01 vs DOC-02 disagree on P-200 max pressure (16 vs 12 bar); DOC-05 ≈ DOC-06.
6. **Dirty titles** — mojibake around em dashes in several titles.
7. **Answer = raw chunk** — no grounding check beyond cosine similarity.

These notes feed the README decision log.
