import logging

from sqlalchemy.orm import Session
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from services.agent.state import AgentState
from services.agent.intent_router import intent_router
from services.agent.tools import make_tools
from services.agent.prompts import build_system_message

logger = logging.getLogger(__name__)

_FALLBACK_RECURSION = (
    "I was unable to complete the answer within the allowed steps. "
    "Please rephrase your question or ask a more specific query."
)


def _parse_tool_output(state: AgentState) -> AgentState:
    """Extract DOCUMENT_IDS header from the last ToolMessage into state."""
    doc_ids = list(state.get("document_ids", []))
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            content = msg.content or ""
            for line in content.splitlines():
                if line.startswith("DOCUMENT_IDS:"):
                    raw = line.removeprefix("DOCUMENT_IDS:").strip()
                    try:
                        doc_ids = [int(x) for x in raw.split(",") if x.strip()]
                    except ValueError:
                        pass  # malformed header — leave document_ids unchanged
            break
    return {**state, "document_ids": doc_ids}


def _should_continue(state: AgentState) -> str:
    """Conditional edge: route to 'tools' if the agent requested a tool call, else END."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


def build_graph(db: Session):
    """Compile the LangGraph ReAct graph. Called once per run_agent() invocation."""
    tools = make_tools(db)
    llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)

    def agent_node(state: AgentState) -> AgentState:
        system_msg = build_system_message(state.get("intent", "hybrid"))
        messages = [system_msg] + list(state["messages"])
        response = llm.invoke(messages)
        return {**state, "messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("intent_router", intent_router)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("parse_tool_output", _parse_tool_output)

    graph.add_edge(START, "intent_router")
    graph.add_edge("intent_router", "agent")
    graph.add_conditional_edges(
        "agent", _should_continue, {"tools": "tools", END: END}
    )
    graph.add_edge("tools", "parse_tool_output")
    graph.add_edge("parse_tool_output", "agent")

    return graph.compile()
