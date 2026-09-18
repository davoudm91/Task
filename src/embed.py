"""Dense embedding index helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBED_MODEL_DIR, EMBED_MODEL_ID


def load_embed_model() -> SentenceTransformer:
    if EMBED_MODEL_DIR.exists() and any(EMBED_MODEL_DIR.iterdir()):
        return SentenceTransformer(str(EMBED_MODEL_DIR))
    return SentenceTransformer(EMBED_MODEL_ID)


def encode_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    vectors = model.encode(texts, show_progress_bar=False)
    vectors = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vectors / norms
