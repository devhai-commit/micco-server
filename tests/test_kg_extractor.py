# backend/tests/test_kg_extractor.py
import json
import pytest
from unittest.mock import MagicMock, patch


def _make_doc(name="test.pdf", category="HopDong"):
    doc = MagicMock()
    doc.id = 1
    doc.name = name
    doc.category = category
    return doc


def _make_openai_response(content: str):
    """Build a minimal mock OpenAI chat completion response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


def test_extract_kg_returns_entities_and_relationships():
    payload = json.dumps({
        "entities": [
            {"name": "HĐ-001", "label": "HopDong"},
            {"name": "Công ty ABC", "label": "NhaCungCap"},
        ],
        "relationships": [
            {
                "source": "HĐ-001", "source_label": "HopDong",
                "relation": "TU_NHA_CUNG_CAP",
                "target": "Công ty ABC", "target_label": "NhaCungCap",
            }
        ],
    })
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response(payload)

    with patch("openai.OpenAI", return_value=mock_client):
        from services.kg_extractor import extract_kg
        result = extract_kg(["chunk text"], _make_doc())

    assert len(result["entities"]) == 2
    assert len(result["relationships"]) == 1
    assert result["entities"][0]["name"] == "HĐ-001"


def test_extract_kg_filters_invalid_labels():
    payload = json.dumps({
        "entities": [
            {"name": "Valid", "label": "HopDong"},
            {"name": "Bad", "label": "NotALabel"},
        ],
        "relationships": [],
    })
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response(payload)

    with patch("openai.OpenAI", return_value=mock_client):
        from services.kg_extractor import extract_kg
        result = extract_kg(["text"], _make_doc())

    assert len(result["entities"]) == 1
    assert result["entities"][0]["name"] == "Valid"


def test_extract_kg_filters_invalid_relation_types():
    payload = json.dumps({
        "entities": [
            {"name": "A", "label": "VatTu"},
            {"name": "B", "label": "NhaCungCap"},
        ],
        "relationships": [
            {
                "source": "A", "source_label": "VatTu",
                "relation": "FAKE_REL",
                "target": "B", "target_label": "NhaCungCap",
            }
        ],
    })
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response(payload)

    with patch("openai.OpenAI", return_value=mock_client):
        from services.kg_extractor import extract_kg
        result = extract_kg(["text"], _make_doc())

    assert result["relationships"] == []


def test_extract_kg_returns_empty_dict_on_openai_exception():
    with patch("openai.OpenAI", side_effect=Exception("API error")):
        from services.kg_extractor import extract_kg
        result = extract_kg(["text"], _make_doc())
    assert result == {}


def test_extract_kg_returns_empty_dict_on_json_parse_error():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response("not json {{")

    with patch("openai.OpenAI", return_value=mock_client):
        from services.kg_extractor import extract_kg
        result = extract_kg(["text"], _make_doc())

    assert result == {}


def test_extract_kg_returns_empty_dict_for_empty_chunks():
    from services.kg_extractor import extract_kg
    result = extract_kg([], _make_doc())
    assert result == {}


def test_extract_kg_uses_only_first_5_chunks():
    """Verify only the first 5 chunks are sent to GPT-4o."""
    chunks = [f"chunk {i}" for i in range(10)]
    payload = json.dumps({"entities": [], "relationships": []})
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response(payload)

    with patch("openai.OpenAI", return_value=mock_client):
        from services.kg_extractor import extract_kg
        extract_kg(chunks, _make_doc())

    call_kwargs = mock_client.chat.completions.create.call_args
    user_content = call_kwargs[1]["messages"][1]["content"]
    # chunks 5-9 must NOT appear in the prompt
    assert "chunk 5" not in user_content
    assert "chunk 4" in user_content


def test_extract_kg_returns_empty_dict_when_llm_returns_empty_lists():
    """When GPT-4o returns no valid entities or relationships, return {}."""
    payload = json.dumps({"entities": [], "relationships": []})
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response(payload)

    with patch("openai.OpenAI", return_value=mock_client):
        from services.kg_extractor import extract_kg
        result = extract_kg(["text"], _make_doc())

    assert result == {}
