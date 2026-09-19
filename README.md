# i4Twins offline RAG — improved baseline

Document-grounded QA over a small industrial corpus (`corpus.jsonl`), designed for on-premises / offline use with limited compute. Focus: diagnosis, hybrid retrieval, abstention, evaluation, and data-quality policies.

`baseline_rag.py` is kept unchanged for comparison.

## Quick start

```powershell
cd E:\Task
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# Download models into .\models (once; then runtime can be offline)
python scripts\download_models.py --embed-only          # MiniLM (required)
python scripts\download_models.py --llm-only            # Qwen/Qwen3-1.7B-Base (optional)

# Ask a question (extractive by default)
python run_rag.py "What is the rated output of the C-100 compressor?"
python run_rag.py "What is the motor power of pump P-200?" --show-hits

# Optional local generation
python run_rag.py "What does error code E-207 mean?" --llm

# Reproduce reported metrics
python -m eval.run_eval
```

Metrics are written to `eval/results.json`.

## Diagnosis — why the baseline is weak

| Issue | Effect |
| --- | --- |
| Top-1 dense only (`argmax`) | No multi-hit evidence; brittle ranking |
| No keyword path | Exact codes (`E-207`, `BRG-4410`) under-served |
| Title not indexed; chunk size 400 on short docs | Chunking is a no-op; titles unused |
| No score / support gate | Always returns a chunk for unanswerable questions |
| No conflict / near-dup policy | DOC-01 vs DOC-02 pressure conflict; DOC-05 ≈ DOC-06 |
| Dirty titles | Em-dash / encoding noise in titles |

On this corpus, dense top-1 often finds the right doc for answerable queries, but abstention recall is 0% — the main industrial failure mode.

## What changed

1. **Hybrid retrieval** — MiniLM dense top-k + BM25 (stopword-filtered, light stemming), RRF fusion, then light lexical / entity re-rank.
2. **Abstention gate** — return `Not found in the documents.` when score is below threshold, equipment codes are missing from hits, or non-entity support coverage is too low (scoped to entity-matching docs).
3. **Data-quality policies**
   - Encoding: normalize title separators; do not rewrite codes like `P-200`.
   - Near-duplicates (DOC-05 / DOC-06): keep both; cite as one cluster.
   - Conflicts (DOC-01 = 16 bar vs DOC-02 = 12 bar): disclose both; do not pick one silently.
   - No factual rewrite of corpus body text.
4. **Generation** — optional `Qwen/Qwen3-1.7B-Base`; default answers are extractive. Abstention runs before generation.

## Models

| Role | Model | Why |
| --- | --- | --- |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Small, local, enough for short English passages |
| Generation | `Qwen/Qwen3-1.7B-Base` | On-prem small LM; base model, so refusal is not trusted to it |

Weights live under `models/` after download (gitignored).

## Evaluation

`eval/eval_set.jsonl`: 10 answerable (including near-dup + conflict) and 6 unanswerable.

```text
python -m eval.run_eval
```

### Results (extractive mode)

| Metric | Baseline | Improved |
| --- | ---: | ---: |
| Retrieval Hit@1 | 100% | 100% |
| Retrieval Hit@3 | 100% | 100% |
| Abstain recall (unanswerable) | 0% | 100% |
| Abstention precision | n/a | 100% |
| Abstention accuracy | n/a | 100% |
| Citation groundedness | n/a | 100% |
| Conflict disclosure | n/a | 100% |

Interpretation: baseline retrieval is already strong on 16 short docs; the gain is trust (abstention + conflict disclosure). Thresholds in `src/config.py` (`ABSTAIN_SCORE_THRESHOLD=0.18`, `SUPPORT_TOKEN_COVERAGE=0.40`) were set using this eval set.

## Trade-offs

- No cross-encoder re-ranker (RAM/latency); BM25 + entity boost cover industrial codes.
- LLM generation is optional; extractive mode is the default for reproducibility.
- Conflict / near-dup lists are explicit for this corpus; a larger corpus would need clustering / NLI.
- Offline after the first model download.

## Layout

```text
baseline_rag.py
corpus.jsonl
run_rag.py
scripts/download_models.py
src/
eval/eval_set.jsonl
eval/run_eval.py
```

## AI Usage

AI coding assistance was used. (Cursor)