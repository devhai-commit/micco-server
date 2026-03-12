# backend/tests/test_agent_init.py
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


def _make_app(messages):
    """Helper: mock build_graph to return an app that yields given messages."""
    mock_app = MagicMock()
    mock_app.invoke.return_value = {
        "messages": messages,
        "intent": "structural",
        "document_ids": [],
    }
    return mock_app


def test_run_agent_returns_agent_result_type():
    """run_agent must return an AgentResult, not a plain str."""
    from services.agent import run_agent, AgentResult
    mock_app = _make_app([HumanMessage(content="q"), AIMessage(content="Answer.")])
    with patch("services.agent.graph.build_graph", return_value=mock_app):
        result = run_agent("query", MagicMock())
    assert isinstance(result, AgentResult)
    assert result.answer == "Answer."


def test_run_agent_graph_data_populated_from_tool_message():
    """graph_data has 2 nodes and 1 edge when query_knowledge_graph ToolMessage present."""
    from services.agent import run_agent
    tool_msg = ToolMessage(
        content='[{"source":"A","source_type":"Supplier","relation":"PROVIDES","target":"B","target_type":"Contract"}]',
        tool_call_id="tc1",
        name="query_knowledge_graph",
    )
    final_msg = AIMessage(content="Found supplier A.")
    mock_app = _make_app([HumanMessage(content="q"), tool_msg, final_msg])
    with patch("services.agent.graph.build_graph", return_value=mock_app):
        result = run_agent("query", MagicMock())
    assert result.graph_data is not None
    assert len(result.graph_data["nodes"]) == 2
    assert len(result.graph_data["edges"]) == 1
    assert result.graph_data["edges"][0] == {"from": "A", "to": "B", "relation": "PROVIDES"}


def test_run_agent_graph_data_none_when_no_graph_tool_called():
    """graph_data is None when no query_knowledge_graph ToolMessage in state."""
    from services.agent import run_agent
    mock_app = _make_app([HumanMessage(content="q"), AIMessage(content="Answer.")])
    with patch("services.agent.graph.build_graph", return_value=mock_app):
        result = run_agent("query", MagicMock())
    assert result.graph_data is None


def test_run_agent_graph_data_none_when_malformed_json():
    """graph_data is None (not an exception) when ToolMessage content is not valid JSON."""
    from services.agent import run_agent
    tool_msg = ToolMessage(
        content="not valid json at all",
        tool_call_id="tc1",
        name="query_knowledge_graph",
    )
    mock_app = _make_app([HumanMessage(content="q"), tool_msg, AIMessage(content="Ans.")])
    with patch("services.agent.graph.build_graph", return_value=mock_app):
        result = run_agent("query", MagicMock())
    assert result.graph_data is None


def test_run_agent_graph_data_none_when_all_rows_missing_columns():
    """graph_data is None when all rows lack the required 5 columns."""
    from services.agent import run_agent
    tool_msg = ToolMessage(
        content='[{"a":"x","b":"y"},{"c":"z"}]',
        tool_call_id="tc1",
        name="query_knowledge_graph",
    )
    mock_app = _make_app([HumanMessage(content="q"), tool_msg, AIMessage(content="Ans.")])
    with patch("services.agent.graph.build_graph", return_value=mock_app):
        result = run_agent("query", MagicMock())
    assert result.graph_data is None


def test_run_agent_graph_data_partial_when_some_rows_valid():
    """graph_data is not None when some rows are valid; invalid rows are silently skipped."""
    from services.agent import run_agent
    tool_msg = ToolMessage(
        content=(
            '[{"a":"x"},'
            '{"source":"A","source_type":"Supplier","relation":"PROVIDES","target":"B","target_type":"Contract"}]'
        ),
        tool_call_id="tc1",
        name="query_knowledge_graph",
    )
    mock_app = _make_app([HumanMessage(content="q"), tool_msg, AIMessage(content="Ans.")])
    with patch("services.agent.graph.build_graph", return_value=mock_app):
        result = run_agent("query", MagicMock())
    assert result.graph_data is not None
    assert len(result.graph_data["nodes"]) == 2
    assert len(result.graph_data["edges"]) == 1
