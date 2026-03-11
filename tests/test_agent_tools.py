import json
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.tools import ToolException


def _get_tools(mock_db=None):
    """Helper: create tools with a mock DB session."""
    if mock_db is None:
        mock_db = MagicMock()
    from services.agent.tools import make_tools
    return make_tools(mock_db)


def _get_tool(tools, name):
    return next(t for t in tools if t.name == name)


# ── make_tools ──────────────────────────────────────────────

def test_make_tools_returns_three_tools():
    tools = _get_tools()
    assert len(tools) == 3
    names = {t.name for t in tools}
    assert names == {"query_knowledge_graph", "search_document_chunks", "get_document_details"}


# ── query_knowledge_graph ────────────────────────────────────

def test_query_knowledge_graph_valid_match():
    with patch("services.agent.tools.neo4j_service") as mock_neo4j:
        mock_neo4j.available = True
        mock_neo4j.run_cypher.return_value = [{"name": "VatTuA"}]
        tool = _get_tool(_get_tools(), "query_knowledge_graph")
        result = tool.invoke({"cypher": "MATCH (n:VatTu) RETURN n.name AS name"})
        data = json.loads(result)
        assert data[0]["name"] == "VatTuA"
        mock_neo4j.run_cypher.assert_called_once()


def test_query_knowledge_graph_blocks_create():
    with patch("services.agent.tools.neo4j_service") as mock_neo4j:
        mock_neo4j.available = True
        tool = _get_tool(_get_tools(), "query_knowledge_graph")
        with pytest.raises(ToolException, match="Write operations"):
            tool.invoke({"cypher": "CREATE (n:VatTu {name: 'x'}) RETURN n"})


def test_query_knowledge_graph_blocks_merge():
    with patch("services.agent.tools.neo4j_service") as mock_neo4j:
        mock_neo4j.available = True
        tool = _get_tool(_get_tools(), "query_knowledge_graph")
        with pytest.raises(ToolException, match="Write operations"):
            tool.invoke({"cypher": "MERGE (n:VatTu {id: 1}) RETURN n"})


def test_query_knowledge_graph_raises_when_neo4j_unavailable():
    with patch("services.agent.tools.neo4j_service") as mock_neo4j:
        mock_neo4j.available = False
        tool = _get_tool(_get_tools(), "query_knowledge_graph")
        with pytest.raises(ToolException, match="Graph DB unavailable"):
            tool.invoke({"cypher": "MATCH (n) RETURN n"})


def test_query_knowledge_graph_caps_at_50_rows():
    with patch("services.agent.tools.neo4j_service") as mock_neo4j:
        mock_neo4j.available = True
        mock_neo4j.run_cypher.return_value = [{"i": i} for i in range(100)]
        tool = _get_tool(_get_tools(), "query_knowledge_graph")
        result = tool.invoke({"cypher": "MATCH (n) RETURN n"})
        data = json.loads(result)
        assert len(data) == 50


# ── search_document_chunks ───────────────────────────────────

def test_search_document_chunks_returns_formatted_result():
    row = MagicMock()
    row.document_id = 3
    row.similarity = 0.87
    row.content = "Vật tư A là loại thép."

    db = MagicMock()
    db.execute.return_value.fetchall.return_value = [row]

    with patch("services.agent.tools.embed", return_value=[[0.1] * 1024]):
        tool = _get_tool(_get_tools(db), "search_document_chunks")
        result = tool.invoke({"query": "thép"})

    assert "DOCUMENT_IDS: 3" in result
    assert "Vật tư A" in result


def test_search_document_chunks_includes_document_ids_header():
    row1 = MagicMock(document_id=1, similarity=0.9, content="A")
    row2 = MagicMock(document_id=2, similarity=0.8, content="B")

    db = MagicMock()
    db.execute.return_value.fetchall.return_value = [row1, row2]

    with patch("services.agent.tools.embed", return_value=[[0.0] * 1024]):
        tool = _get_tool(_get_tools(db), "search_document_chunks")
        result = tool.invoke({"query": "anything"})

    assert result.startswith("DOCUMENT_IDS:")
    first_line = result.splitlines()[0]
    ids = [int(x) for x in first_line.replace("DOCUMENT_IDS:", "").strip().split(",")]
    assert ids == [1, 2]


def test_search_document_chunks_returns_empty_string_on_db_error():
    db = MagicMock()
    db.execute.side_effect = Exception("Connection refused")

    with patch("services.agent.tools.embed", return_value=[[0.0] * 1024]):
        tool = _get_tool(_get_tools(db), "search_document_chunks")
        result = tool.invoke({"query": "test"})

    assert result == ""


def test_search_document_chunks_returns_no_results_message_when_empty():
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = []

    with patch("services.agent.tools.embed", return_value=[[0.0] * 1024]):
        tool = _get_tool(_get_tools(db), "search_document_chunks")
        result = tool.invoke({"query": "nothing matches"})

    assert "No matching" in result


# ── get_document_details ─────────────────────────────────────

def test_get_document_details_found():
    doc = MagicMock()
    doc.id = 7
    doc.name = "HopDong_2026.pdf"
    doc.category = "HopDong"
    doc.owner_name = "admin"
    doc.date = "2026-01-15"
    doc.ingest_status = "completed"

    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = doc

    tool = _get_tool(_get_tools(db), "get_document_details")
    result = tool.invoke({"document_id": 7})

    assert "HopDong_2026.pdf" in result
    assert "HopDong" in result
    assert "completed" in result


def test_get_document_details_not_found():
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None

    tool = _get_tool(_get_tools(db), "get_document_details")
    result = tool.invoke({"document_id": 999})

    assert result == "Document not found."
