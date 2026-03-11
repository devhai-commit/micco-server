import logging
import os
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        model_name = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
        logger.info("Loading embedding model: %s", model_name)
        _model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded.")
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings using bge-m3 (1024-dim).

    Retries with batch_size=1 on OOM errors.

    Returns:
        List of 1024-dim float vectors.
    """
    if not texts:
        return []

    model = _get_model()
    try:
        vectors = model.encode(texts, batch_size=8, normalize_embeddings=True)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            logger.warning("OOM on embedding batch — retrying with batch_size=1")
            vectors = model.encode(texts, batch_size=1, normalize_embeddings=True)
        else:
            raise

    # SentenceTransformer returns numpy arrays; convert to plain Python lists
    return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]
