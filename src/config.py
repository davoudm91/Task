"""Tunable constants for the offline RAG pipeline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "corpus.jsonl"
EVAL_SET_PATH = ROOT / "eval" / "eval_set.jsonl"
RESULTS_PATH = ROOT / "eval" / "results.json"

# Local model dirs (populated by scripts/download_models.py)
MODELS_DIR = ROOT / "models"
EMBED_MODEL_DIR = MODELS_DIR / "all-MiniLM-L6-v2"
LLM_MODEL_DIR = MODELS_DIR / "Qwen3-1.7B-Base"

# Hugging Face ids used for first-time download
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_ID = "Qwen/Qwen3-1.7B-Base"

# Retrieval
K_DENSE = 5
K_BM25 = 5
TOP_N = 3
RRF_K = 60

# Abstention: calibrated on eval (see README). Cosine similarity after hybrid re-score.
ABSTAIN_SCORE_THRESHOLD = 0.18
# Minimum fraction of non-entity content tokens that must appear in retrieved text.
SUPPORT_TOKEN_COVERAGE = 0.40

# Generation
MAX_NEW_TOKENS = 96
ABSTAIN_MESSAGE = "Not found in the documents."

# Known near-duplicate clusters (same factual claim)
NEAR_DUP_CLUSTERS = (
    frozenset({"DOC-05", "DOC-06"}),
)

# Known conflicting document pairs on overlapping topics
CONFLICT_PAIRS = (
    frozenset({"DOC-01", "DOC-02"}),  # P-200 max operating pressure 16 vs 12 bar
)
