"""Shared fastembed singleton used by both guardrail evaluators and TDA."""

from __future__ import annotations

import numpy as np

_model = None


def get_embedding_model():
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        _model = TextEmbedding("BAAI/bge-small-en-v1.5")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_embedding_model()
    return np.array(list(model.embed(texts)))
