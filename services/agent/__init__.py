import logging

from sqlalchemy.orm import Session
from langchain_core.messages import AIMessage
from langgraph.errors import GraphRecursionError

logger = logging.getLogger(__name__)

_FALLBACK_API_ERROR = "Service temporarily unavailable. Please try again."


def run_agent(query: str, db: Session) -> str:
    """Run the MICCO GraphRAG agent for a user query.

    Args:
        query: User's natural language question (Vietnamese or English).
        db:    SQLAlchemy Session (caller owns lifecycle; tools use it read-only).

    Returns:
        Agent's answer as a plain string. Never raises — returns a fallback
        string on any failure so callers need no error handling.
    """
    from langchain_core.messages import HumanMessage
    import services.agent.graph as _graph
    try:
        app = _graph.build_graph(db)
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "intent": "",
            "document_ids": [],
        }
        result = app.invoke(initial_state, config={"recursion_limit": 6})

        # Return the last non-tool AIMessage content
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                return msg.content

        return _graph._FALLBACK_RECURSION

    except GraphRecursionError:
        # GraphRecursionError is raised by app.invoke() when the iteration cap is hit.
        # It is caught here because run_agent() owns the invoke() call.
        logger.warning("Agent hit recursion limit for query: %.80s", query)
        return _graph._FALLBACK_RECURSION

    except Exception as exc:
        logger.error("Agent failed for query: %.80s — %s", query, exc, exc_info=True)
        return _FALLBACK_API_ERROR
