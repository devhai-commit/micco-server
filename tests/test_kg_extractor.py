# backend/tests/test_kg_extractor.py
import json
import sys
import importlib
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def _reload_kg_extractor():
    """Reload kg_extractor module before each test to clear stubs."""
    # Remove module from cache if it's a stub (types.ModuleType without __file__)
    if "services.kg_extractor" in sys.modules:
        mod = sys.modules["services.kg_extractor"]
        if not hasattr(mod, "__file__"):
            del sys.modules["services.kg_extractor"]
    if "services" in sys.modules:
        # Also reload services to clear cached reference
        try:
            importlib.reload(sys.modules["services"])
        except (ImportError, AttributeError):
            pass


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
            {"name": "HĐ-001", "label": "HopDong", "attributes": {"so_van_ban": "001", "ngay": "15/05/2025"}},
            {"name": "Công ty ABC", "label": "NhaCungCap", "attributes": {"ma_so_thue": "0100101072"}},
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
    assert result["entities"][0]["attributes"]["so_van_ban"] == "001"
    assert result["entities"][1]["attributes"]["ma_so_thue"] == "0100101072"


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


def test_extract_kg_batches_chunks():
    """Verify chunks are batched (batch size = 20)."""
    chunks = [f"chunk {i}" for i in range(25)]
    payload = json.dumps({"entities": [], "relationships": []})
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response(payload)

    with patch("openai.OpenAI", return_value=mock_client):
        from services.kg_extractor import extract_kg
        extract_kg(chunks, _make_doc())

    # 25 chunks → 2 entity batches
    assert mock_client.chat.completions.create.call_count == 2
    # First batch should contain chunk 0-19
    first_call = mock_client.chat.completions.create.call_args_list[0]
    user_content = first_call[1]["messages"][1]["content"]
    assert "chunk 0" in user_content
    assert "chunk 19" in user_content
    assert "chunk 20" not in user_content


def test_extract_kg_returns_empty_dict_when_llm_returns_empty_lists():
    """When GPT-4o returns no valid entities or relationships, return {}."""
    payload = json.dumps({"entities": [], "relationships": []})
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response(payload)

    with patch("openai.OpenAI", return_value=mock_client):
        from services.kg_extractor import extract_kg
        result = extract_kg(["text"], _make_doc())

    assert result == {}


def test_extract_kg_entity_without_attributes_gets_empty_dict():
    """Entities returned without 'attributes' key get an empty dict."""
    payload = json.dumps({
        "entities": [{"name": "Thép CT3", "label": "VatTu"}],
        "relationships": [],
    })
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response(payload)

    with patch("openai.OpenAI", return_value=mock_client):
        from services.kg_extractor import extract_kg
        result = extract_kg(["text"], _make_doc())

    assert result["entities"][0]["attributes"] == {}


def test_extract_kg_merges_attributes_across_batches():
    """When the same entity appears in multiple batches, attributes are merged."""
    batch1 = json.dumps({
        "entities": [{"name": "Cty A", "label": "NhaCungCap", "attributes": {"dia_chi": "Hà Nội"}}],
    })
    batch2 = json.dumps({
        "entities": [{"name": "Cty A", "label": "NhaCungCap", "attributes": {"ma_so_thue": "012345"}}],
    })
    rel_payload = json.dumps({"relationships": []})

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        _make_openai_response(batch1),
        _make_openai_response(batch2),
        _make_openai_response(rel_payload),
    ]

    # Use 25 chunks to trigger 2 entity batches (batch size = 20)
    chunks = [f"chunk {i}" for i in range(25)]

    with patch("openai.OpenAI", return_value=mock_client):
        from services.kg_extractor import extract_kg
        result = extract_kg(chunks, _make_doc())

    assert len(result["entities"]) == 1
    attrs = result["entities"][0]["attributes"]
    assert attrs["dia_chi"] == "Hà Nội"
    assert attrs["ma_so_thue"] == "012345"
