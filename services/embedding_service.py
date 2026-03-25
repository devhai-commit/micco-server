import logging
from typing import Protocol

import config

logger = logging.getLogger(__name__)

# ── Provider protocol ─────────────────────────────────────────────────────────

class _EmbedProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


# ── BGE (local SentenceTransformer) ──────────────────────────────────────────

class _BgeProvider:
    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading local embedding model: %s", config.EMBED_MODEL)
        self._model = SentenceTransformer(config.EMBED_MODEL)
        logger.info("Embedding model loaded.")

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            vectors = self._model.encode(texts, batch_size=8, normalize_embeddings=True)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                logger.warning("OOM on embedding batch — retrying with batch_size=1")
                try:
                    import torch
                    torch.cuda.empty_cache()
                except ImportError:
                    pass
                vectors = self._model.encode(texts, batch_size=1, normalize_embeddings=True)
            else:
                raise
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]


# ── OpenAI ────────────────────────────────────────────────────────────────────

class _OpenAIProvider:
    def __init__(self) -> None:
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set — required for EMBEDDING_PROVIDER=openai")
        from openai import OpenAI
        self._client = OpenAI(api_key=config.OPENAI_API_KEY)
        self._model = config.OPENAI_EMBED_MODEL
        logger.info("OpenAI embedding provider ready: model=%s dimensions=%d", self._model, config.EMBED_DIMENSIONS)

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            model=self._model,
            input=texts,
            dimensions=config.EMBED_DIMENSIONS,  # text-embedding-3-* supports dimension truncation
        )
        # Sort by index to preserve order
        items = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in items]


# ── Lazy singleton ────────────────────────────────────────────────────────────

_provider: _EmbedProvider | None = None


def _get_provider() -> _EmbedProvider:
    global _provider
    if _provider is None:
        provider_name = config.EMBEDDING_PROVIDER.lower()
        if provider_name == "openai":
            _provider = _OpenAIProvider()
        else:
            if provider_name != "bge":
                logger.warning("Unknown EMBEDDING_PROVIDER=%r, falling back to bge", provider_name)
            _provider = _BgeProvider()
    return _provider


# ── Public API ────────────────────────────────────────────────────────────────

def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings.

    Provider is selected via EMBEDDING_PROVIDER env var:
      - "bge"    → local BAAI/bge-m3 via SentenceTransformer (default)
      - "openai" → OpenAI API (requires OPENAI_API_KEY, uses OPENAI_EMBED_MODEL)

    Both providers output EMBED_DIMENSIONS-dim vectors (512) to match
    the vector(512) column in document_chunks.

    Returns:
        List of 512-dim float vectors.
    """
    if not texts:
        return []
    return _get_provider().embed(texts)
