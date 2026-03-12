# backend/services/agent/__init__.py
import json
import logging
from dataclasses import dataclass

from sqlalchemy.orm import Session
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError

logger = logging.getLogger(__name__)

_FALLBACK_API_ERROR = "Service temporarily unavailable. Please try again."


@dataclass
class AgentResult:
    answer: str
    graph_data: dict | None
    # graph_data shape when not None:
    # {
    #   "nodes": [{"id": str, "label": str, "type": str}],
    #   "edges": [{"from": str, "to": str, "relation": str}]
    # }


def _extract_graph_data(state: dict) -> dict | None:
    """Extract structured graph data from the last query_knowledge_graph ToolMessage.

    Returns None if the tool was not called, JSON was unparseable,
    or no rows had the required 5 columns (source, source_type, relation, target, target_type).
    """
    try:
        for msg in reversed(state["messages"]):
            if not isinstance(msg, ToolMessage):
                continue
            if getattr(msg, "name", None) != "query_knowledge_graph":
                continue

            rows = json.loads(msg.content)
            nodes: dict[str, dict] = {}  # keyed by id; first occurrence wins
            edges: list[dict] = []

            for row in rows:
                if not all(k in row for k in ("source", "source_type", "relation", "target", "target_type")):
                    continue  # skip rows missing any required column
                if row["source"] not in nodes:
                    nodes[row["source"]] = {
                        "id": row["source"],
                        "label": row["source"],
                        "type": row["source_type"],
                    }
                if row["target"] not in nodes:
                    nodes[row["target"]] = {
                        "id": row["target"],
                        "label": row["target"],
                        "type": row["target_type"],
                    }
                edges.append({"from": row["source"], "to": row["target"], "relation": row["relation"]})

            if not nodes:
                return None
            return {"nodes": list(nodes.values()), "edges": edges}

    except Exception:
        return None

    return None  # for loop exhausted without finding a matching query_knowledge_graph ToolMessage


def run_agent(query: str, db: Session) -> AgentResult:
    """Run the LangGraph ReAct agent and return an AgentResult.

    Never raises — all exceptions are caught and returned as AgentResult with a fallback answer.
    """
    from langchain_core.messages import HumanMessage

    try:
        import services.agent.graph as _graph  # deferred import — allows patching in tests

        app = _graph.build_graph(db)
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "intent": "",
            "document_ids": [],
        }
        result = app.invoke(initial_state, config={"recursion_limit": 6})

        graph_data = _extract_graph_data(result)

        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                return AgentResult(answer=msg.content, graph_data=graph_data)

        return AgentResult(answer=_graph._FALLBACK_RECURSION, graph_data=None)

    except GraphRecursionError:
        logger.warning("Agent hit recursion limit for query: %.80s", query)
        return AgentResult(answer=_graph._FALLBACK_RECURSION, graph_data=None)

    except Exception as exc:
        logger.error("Agent failed for query: %.80s — %s", query, exc, exc_info=True)
        return AgentResult(answer=_FALLBACK_API_ERROR, graph_data=None)
