from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    intent: str           # "structural" | "semantic" | "hybrid"
    document_ids: list[int]  # populated by parse_tool_output node
