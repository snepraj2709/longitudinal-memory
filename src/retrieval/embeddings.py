"""Deterministic development embeddings for retrieval-index mechanics."""

from __future__ import annotations

import hashlib
import math
from typing import Protocol
import unicodedata

from .contracts import EMBEDDING_DIMENSION, EMBEDDING_VERSION, RetrievalIndexError


class Embedder(Protocol):
    """Fixed-dimensional embedding boundary for retrieval index builders."""

    version: str
    dimension: int

    def embed(self, text: str) -> tuple[float, ...]: ...


class DeterministicTokenHashEmbedder:
    """Create a reproducible signed token-count vector without a model."""

    version = EMBEDDING_VERSION
    dimension = EMBEDDING_DIMENSION

    def embed(self, text: str) -> tuple[float, ...]:
        if not isinstance(text, str):
            raise RetrievalIndexError("embedding input must be text")
        counts = [0] * self.dimension
        for token in normalized_tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:8], "big") % self.dimension
            counts[bucket] += 1 if digest[8] & 1 == 0 else -1
        norm = math.sqrt(sum(value * value for value in counts))
        if norm == 0.0:
            return tuple(0.0 for _ in range(self.dimension))
        return tuple(float(value / norm) for value in counts)


def normalized_tokens(text: str) -> tuple[str, ...]:
    """Return NFKC, case-folded Unicode alphanumeric runs."""

    if not isinstance(text, str):
        raise RetrievalIndexError("token input must be text")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens: list[str] = []
    current: list[str] = []
    for character in normalized:
        if character.isalnum():
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tuple(tokens)
