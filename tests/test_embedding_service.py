import pytest
from unittest.mock import MagicMock, patch


def _make_mock_model(dim=1024):
    model = MagicMock()
    import numpy as np
    model.encode = MagicMock(
        side_effect=lambda texts, **kwargs: np.random.rand(len(texts), dim).tolist()
    )
    return model


def test_embed_returns_list_of_vectors():
    with patch("services.embedding_service.SentenceTransformer", return_value=_make_mock_model()):
        from services import embedding_service
        embedding_service._model = None  # reset singleton
        result = embedding_service.embed(["hello", "world"])
        assert len(result) == 2
        assert len(result[0]) == 1024


def test_embed_empty_input_returns_empty():
    with patch("services.embedding_service.SentenceTransformer", return_value=_make_mock_model()):
        from services import embedding_service
        embedding_service._model = None
        result = embedding_service.embed([])
        assert result == []


def test_embed_model_is_cached():
    mock_model = _make_mock_model()
    with patch("services.embedding_service.SentenceTransformer", return_value=mock_model) as mock_cls:
        from services import embedding_service
        embedding_service._model = None
        embedding_service.embed(["a"])
        embedding_service.embed(["b"])
        # SentenceTransformer constructor called only once
        assert mock_cls.call_count == 1


def test_embed_oom_retries_with_batch_1():
    import numpy as np
    call_count = {"n": 0}
    def fake_encode(texts, batch_size=8, **kwargs):
        call_count["n"] += 1
        if batch_size > 1:
            raise RuntimeError("CUDA out of memory")
        return np.random.rand(len(texts), 1024).tolist()

    mock_model = MagicMock()
    mock_model.encode = fake_encode

    with patch("services.embedding_service.SentenceTransformer", return_value=mock_model):
        from services import embedding_service
        embedding_service._model = None
        result = embedding_service.embed(["text1", "text2"])
        assert len(result) == 2
        assert call_count["n"] == 2  # first attempt (OOM) + retry with batch_size=1
