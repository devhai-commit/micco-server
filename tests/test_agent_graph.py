import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from services.agent import AgentResult


# ── _parse_tool_output ───────────────────────────────────────

def test_parse_tool_output_extracts_document_ids():
    from services.agent.graph import _parse_tool_output
    state = {
        "messages": [
            ToolMessage(
                content="DOCUMENT_IDS: 1,2,3\n---\nChunk 1: some content",
                tool_call_id="tc1",
            )
        ],
        "intent": "semantic",
        "document_ids": [],
    }
    result = _parse_tool_output(state)
    assert result["document_ids"] == [1, 2, 3]


def test_parse_tool_output_no_header_leaves_unchanged():
    from services.agent.graph import _parse_tool_output
    state = {
        "messages": [
            ToolMessage(content="Graph DB unavailable.", tool_call_id="tc1")
        ],
        "intent": "structural",
        "document_ids": [5, 6],
    }
    result = _parse_tool_output(state)
    assert result["document_ids"] == [5, 6]


def test_parse_tool_output_ignores_non_tool_messages():
    from services.agent.graph import _parse_tool_output
    state = {
        "messages": [
            AIMessage(content="DOCUMENT_IDS: 99,100"),  # not a ToolMessage
        ],
        "intent": "hybrid",
        "document_ids": [],
    }
    result = _parse_tool_output(state)
    assert result["document_ids"] == []


def test_parse_tool_output_handles_malformed_ids_gracefully():
    from services.agent.graph import _parse_tool_output
    state = {
        "messages": [
            ToolMessage(content="DOCUMENT_IDS: abc,def\n---\nChunk", tool_call_id="tc1")
        ],
        "intent": "semantic",
        "document_ids": [],
    }
    result = _parse_tool_output(state)
    assert result["document_ids"] == []


# ── _should_continue ─────────────────────────────────────────

def test_should_continue_routes_to_tools_when_tool_calls_present():
    from services.agent.graph import _should_continue
    msg = AIMessage(
        content="",
        tool_calls=[{
            "name": "query_knowledge_graph",
            "args": {"cypher": "MATCH (n) RETURN n"},
            "id": "tc1",
            "type": "tool_call",
        }],
    )
    state = {"messages": [msg], "intent": "structural", "document_ids": []}
    assert _should_continue(state) == "tools"


def test_should_continue_routes_to_end_when_no_tool_calls():
    from services.agent.graph import _should_continue
    from langgraph.graph import END
    msg = AIMessage(content="The supplier is NhaCungCapA.")
    state = {"messages": [msg], "intent": "structural", "document_ids": []}
    assert _should_continue(state) == END


# ── run_agent via mocked build_graph ─────────────────────────

def test_run_agent_returns_final_ai_message_content():
    from services.agent import run_agent
    mock_app = MagicMock()
    final_msg = AIMessage(content="The supplier is NhaCungCapA.")
    mock_app.invoke.return_value = {
        "messages": [HumanMessage(content="query"), final_msg],
        "intent": "structural",
        "document_ids": [],
    }
    with patch("services.agent.graph.build_graph", return_value=mock_app):
        result = run_agent("Nhà cung cấp nào?", MagicMock())
    assert result.answer == "The supplier is NhaCungCapA."


def test_run_agent_recursion_returns_exact_fallback():
    from services.agent import run_agent
    from services.agent.graph import _FALLBACK_RECURSION
    from langgraph.errors import GraphRecursionError

    mock_app = MagicMock()
    mock_app.invoke.side_effect = GraphRecursionError("too many steps")

    with patch("services.agent.graph.build_graph", return_value=mock_app):
        result = run_agent("query", MagicMock())

    assert result.answer == _FALLBACK_RECURSION
    assert "unable to complete" in result.answer.lower()


def test_run_agent_api_error_returns_service_unavailable():
    from services.agent import run_agent, _FALLBACK_API_ERROR

    mock_app = MagicMock()
    mock_app.invoke.side_effect = Exception("OpenAI timeout")

    with patch("services.agent.graph.build_graph", return_value=mock_app):
        result = run_agent("query", MagicMock())

    assert result.answer == _FALLBACK_API_ERROR
    assert "unavailable" in result.answer.lower()


def test_run_agent_never_raises():
    from services.agent import run_agent

    mock_app = MagicMock()
    mock_app.invoke.side_effect = RuntimeError("unexpected crash")

    with patch("services.agent.graph.build_graph", return_value=mock_app):
        try:
            result = run_agent("anything", MagicMock())
            assert isinstance(result, AgentResult)
        except Exception:
            pytest.fail("run_agent raised an exception — it must never raise")


def test_run_agent_returns_fallback_when_no_ai_message_in_result():
    from services.agent import run_agent
    from services.agent.graph import _FALLBACK_RECURSION

    mock_app = MagicMock()
    # Simulate: invoke succeeded but messages contain only a HumanMessage (no AIMessage)
    mock_app.invoke.return_value = {
        "messages": [HumanMessage(content="original query")],
        "intent": "structural",
        "document_ids": [],
    }
    with patch("services.agent.graph.build_graph", return_value=mock_app):
        result = run_agent("query", MagicMock())

    assert result.answer == _FALLBACK_RECURSION
