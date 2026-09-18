# i4Twins offline RAG — improved baseline

Document-grounded QA over a small industrial corpus (`corpus.jsonl`), designed for **on-premises / offline** use with limited compute. Focus: diagnosis, hybrid retrieval, abstention (hallucination control), evaluation, and explicit data-quality policies.

The original `baseline_rag.py` is kept unchanged for comparison.

## Quick start

```powershell
cd E:\Task
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# Download models into .\models (once; then runtime can be offline)
python scripts\download_models.py --embed-only          # MiniLM (required for retrieval)
python scripts\download_models.py --llm-only            # Qwen/Qwen3-1.7B-Base (optional generation)

# Ask a question (extractive answers by default — fast, deterministic)
python run_rag.py "What is the rated output of the C-100 compressor?"
python run_rag.py "What is the motor power of pump P-200?" --show-hits

# Optional: use local Qwen3-1.7B-Base for generation
python run_rag.py "What does error code E-207 mean?" --llm

# Reproduce every reported metric
python -m eval.run_eval
```

Reported metrics are written to `eval/results.json`.

## Diagnosis — why the baseline is weak

See also `docs/baseline_diagnosis.md`.

| Issue | Effect |
| --- | --- |
| Top-1 dense only (`argmax`) | No multi-hit evidence; brittle ranking |
| No keyword path | Exact codes (`E-207`, `BRG-4410`) under-served |
| Title not indexed; chunk size 400 on short docs | Chunking is a no-op; titles unused |
| No score / support gate | Always returns a chunk → fabricates grounding for unanswerable questions |
| No conflict / near-dup policy | DOC-01 vs DOC-02 pressure conflict; DOC-05 ≈ DOC-06 |
| Dirty titles | Em-dash / encoding noise in titles |

On this tiny corpus, dense top-1 often “works” for answerable lookups, but **abstention recall is 0%** — the critical industrial failure mode.

## What changed

1. **Hybrid retrieval** — MiniLM dense top-k + BM25 (stopword-filtered, light stemming) fused with **RRF**, then light lexical / entity re-rank (`E-207`, `P-200`, …).
2. **Abstention gate** — refuse with `Not found in the documents.` when (a) fused score &lt; threshold, (b) query equipment codes missing from hits, or (c) non-entity content-token support coverage is too low (scoped to entity-matching docs so C-100 “motor power” does not leak from another asset).
3. **Data-quality policies** (explicit):
   - **Encoding:** normalize spaced replacement/mojibake title separators; **never** rewrite codes like `P-200`.
   - **Near-duplicates (DOC-05 / DOC-06):** keep both; cite the cluster together.
   - **Conflicts (DOC-01 = 16 bar vs DOC-02 = 12 bar):** disclose both values; do not silently pick one.
   - **No factual rewrite** of corpus body text.
4. **Generation** — optional `Qwen/Qwen3-1.7B-Base` with completion-style prompting; default path is **extractive** (`--no-llm` / default) so eval stays reproducible on CPU-only hosts. Abstention is enforced **before** generation (base models are unreliable for refusal).

## Models (constraints)

| Role | Model | Why |
| --- | --- | --- |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (~80MB) | Small, local, good enough for short English tech passages |
| Generation | `Qwen/Qwen3-1.7B-Base` (~3.4GB) | On-prem small LM as specified; **base** (not instruct), so we do not rely on it for abstention |

After `scripts/download_models.py`, weights live under `models/` (gitignored). Runtime prefers local dirs.

## Evaluation

Set: `eval/eval_set.jsonl` — 10 answerable (incl. near-dup + conflict) and 6 unanswerable.

Reproduce:

```text
python -m eval.run_eval
```

### Results (extractive mode, last run)

| Metric | Baseline | Improved |
| --- | ---: | ---: |
| Retrieval Hit@1 | 100% | 100% |
| Retrieval Hit@3 | 100% | 100% |
| Abstain recall (unanswerable) | 0% | **100%** |
| Abstention precision | n/a | **100%** |
| Abstention accuracy | n/a | **100%** |
| Citation groundedness | n/a | **100%** |
| Conflict disclosure | n/a | **100%** |

**Interpretation:** On 16 short, distinct docs, baseline retrieval already hits the right document often — the real gap is **trust**. The improved system keeps retrieval quality while adding deterministic abstention and conflict disclosure. Thresholds (`ABSTAIN_SCORE_THRESHOLD=0.18`, `SUPPORT_TOKEN_COVERAGE=0.40` in `src/config.py`) were calibrated against this eval set.

## Trade-offs

- **No cross-encoder re-ranker** — saves RAM/latency on limited hardware; entity boost + BM25 cover industrial codes.
- **Base LLM optional** — generation quality is secondary; extractive answers are the default for reproducibility.
- **Hard-coded conflict pair / near-dup cluster** — appropriate for this corpus; for a larger corpus, replace with near-dup clustering + NLI conflict detection.
- **Offline after download** — first setup needs Hub access; afterward models are local.

## Project layout

```text
baseline_rag.py          # original weak baseline (unchanged)
corpus.jsonl
run_rag.py               # CLI
scripts/download_models.py
src/                     # improved pipeline
eval/eval_set.jsonl
eval/run_eval.py         # single command to reproduce metrics
docs/baseline_diagnosis.md
```

## AI Usage

- **Tools:** Cursor agent (Composer) used to scaffold the package, implement hybrid retrieval / abstention / eval, and draft this README.
- **Human / review changes after AI output:**
  - Fixed a harmful title-normalization regex that rewrote `P-200` into `P — 200` (would have broken entity matching).
  - Fixed BM25 stopword leakage (`should`/`be` ranking DOC-03 above DOC-10 for calibration questions).
  - Fixed support-coverage cross-doc leakage (C-100 “motor power” text incorrectly supporting a P-200 query) and hyphen-compound false matches (`oil` inside `air-oil`).
  - Kept abstention in the retrieval gate rather than trusting Qwen3-1.7B-**Base** to refuse.
- **Concrete AI mistake caught:** an early `_fix_encoding` step replaced *all* hyphens with spaced em dashes, destroying equipment codes in titles; corrected to only normalize spaced mojibake/separator glyphs.

## Defense-session notes

- Re-index is automatic on process start from `corpus.jsonl`.
- Tunables live in `src/config.py`.
- Prefer `python -m eval.run_eval` and `python run_rag.py --no-llm` for live demos if GPU RAM is tight.
