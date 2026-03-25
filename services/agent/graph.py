import logging

from sqlalchemy.orm import Session
from sqlalchemy import text as sa_text
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from services.agent.state import AgentState
from services.agent.intent_router import intent_router
from services.agent.tools import make_tools
from services.agent.prompts import build_system_message, build_context_message

logger = logging.getLogger(__name__)

_FALLBACK_RECURSION = (
    "I was unable to complete the answer within the allowed steps. "
    "Please rephrase your question or ask a more specific query."
)

# ── Retrieval helpers ────────────────────────────────────────────────────────

def _fetch_graphrag_local(
    query: str,
    vector: list[float],
    db: Session,
    dept_id: int | None,
    top_k: int = 5,
) -> str:
    """GraphRAG Local: entity embedding → graph traversal → community → chunks."""
    from services.neo4j_service import neo4j_service

    parts: list[str] = []

    # 1. Entity vector similarity (pgvector)
    try:
        entity_rows = db.execute(
            sa_text("SELECT * FROM search_entities_by_embedding(CAST(:emb AS vector), :limit)"),
            {"emb": str(vector), "limit": top_k},
        ).fetchall()
    except Exception as exc:
        db.rollback()
        logger.warning("Entity embedding search failed: %s", exc)
        entity_rows = []

    entity_names = [r.entity_name for r in entity_rows]

    if entity_rows:
        parts.append("### Entities tìm được")
        for r in entity_rows:
            line = f"- [{r.entity_label}] **{r.entity_name}** (sim={r.similarity:.3f})"
            if r.description:
                line += f"\n  {r.description}"
            parts.append(line)

    # 2. Graph traversal (Neo4j)
    if neo4j_service.available and entity_names:
        try:
            placeholders = ", ".join(f"'{n}'" for n in entity_names)
            rels = neo4j_service.run_cypher(
                f"""
                MATCH (a)-[r]-(b)
                WHERE a.name IN [{placeholders}] OR b.name IN [{placeholders}]
                RETURN a.name AS source, labels(a)[0] AS src_type,
                       type(r) AS relation,
                       b.name AS target, labels(b)[0] AS tgt_type
                LIMIT 40
                """,
                {},
            )
            if rels:
                parts.append("\n### Quan hệ trong Knowledge Graph")
                seen: set[tuple] = set()
                for rel in rels:
                    key = (rel.get("source"), rel.get("relation"), rel.get("target"))
                    if key not in seen:
                        seen.add(key)
                        parts.append(
                            f"- {rel['source']} ({rel.get('src_type', '')}) "
                            f"--[{rel['relation']}]--> "
                            f"{rel['target']} ({rel.get('tgt_type', '')})"
                        )
        except Exception as exc:
            logger.warning("Graph traversal failed: %s", exc)

    # 3. Community context
    if entity_names:
        try:
            community_rows = db.execute(
                sa_text("""
                    SELECT DISTINCT c.title, c.summary
                    FROM communities c
                    JOIN community_entities ce ON c.id = ce.community_id
                    WHERE ce.entity_name = ANY(:names)
                    ORDER BY c.rank DESC
                    LIMIT 3
                """),
                {"names": entity_names},
            ).fetchall()
            if community_rows:
                parts.append("\n### Community Context")
                for cr in community_rows:
                    parts.append(f"- **{cr.title}**: {cr.summary}")
        except Exception as exc:
            db.rollback()
            logger.warning("Community context fetch failed: %s", exc)

    # 4. Source chunks (entity-related)
    try:
        chunk_rows = db.execute(
            sa_text(
                "SELECT * FROM search_chunks_by_embedding"
                "(CAST(:emb AS vector), :dept_id, :limit)"
            ),
            {"emb": str(vector), "dept_id": dept_id, "limit": top_k},
        ).fetchall()
        if chunk_rows:
            parts.append("\n### Nội dung tài liệu liên quan (GraphRAG)")
            for i, cr in enumerate(chunk_rows, 1):
                parts.append(
                    f"**[{i}]** {cr.source_type} '{cr.source_name}'"
                    f" (sim={cr.similarity:.3f}):\n{cr.chunk_content}"
                )
    except Exception as exc:
        db.rollback()
        logger.warning("Chunk retrieval in local search failed: %s", exc)

    return "\n".join(parts)


def _fetch_vector_chunks(
    vector: list[float],
    db: Session,
    dept_id: int | None,
    limit: int = 8,
    seen_names: set[str] | None = None,
) -> str:
    """Pure pgvector semantic search — returns chunks not already in GraphRAG results."""
    parts: list[str] = []
    try:
        rows = db.execute(
            sa_text(
                "SELECT * FROM search_chunks_by_embedding"
                "(CAST(:emb AS vector), :dept_id, :limit)"
            ),
            {"emb": str(vector), "dept_id": dept_id, "limit": limit},
        ).fetchall()

        # Deduplicate against what GraphRAG already returned
        novel = [
            r for r in rows
            if (seen_names is None or r.source_name not in seen_names)
        ]
        if novel:
            parts.append("### Nội dung tài liệu bổ sung (Vector Search)")
            for i, r in enumerate(novel, 1):
                parts.append(
                    f"**[{i}]** {r.source_type} '{r.source_name}'"
                    f" (sim={r.similarity:.3f}):\n{r.chunk_content}"
                )
    except Exception as exc:
        db.rollback()
        logger.warning("Vector chunk search failed: %s", exc)

    return "\n".join(parts)


# ── Graph nodes ──────────────────────────────────────────────────────────────

def _make_retrieval_node(db: Session, department_id: int | None):
    """Factory: returns a retrieval_node closure bound to the current db + dept."""

    def retrieval_node(state: AgentState) -> AgentState:
        """Always-on parallel retrieval: GraphRAG Local + pgvector chunk search.

        Runs both searches with the same query embedding, combines results
        and stores them in state['retrieval_context'] for the agent.
        """
        from services.embedding_service import embed

        # Extract user query
        query = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                query = msg.content
                break

        if not query:
            return {**state, "retrieval_context": ""}

        try:
            # Compute embedding once, reuse for both searches
            vector = embed([query])[0]

            # ── GraphRAG Local Search ────────────────────────────────
            local_ctx = _fetch_graphrag_local(query, vector, db, department_id)

            # Collect source names already returned by local search to avoid dups
            seen_names: set[str] = set()
            for line in local_ctx.splitlines():
                if line.startswith("**[") and "'" in line:
                    try:
                        seen_names.add(line.split("'")[1])
                    except IndexError:
                        pass

            # ── Pure Vector Chunk Search ─────────────────────────────
            chunks_ctx = _fetch_vector_chunks(vector, db, department_id, seen_names=seen_names)

            # ── Combine ──────────────────────────────────────────────
            sections = []
            if local_ctx:
                sections.append("## GraphRAG Local Search\n" + local_ctx)
            if chunks_ctx:
                sections.append("## Document Vector Search\n" + chunks_ctx)

            retrieval_context = "\n\n".join(sections)
            logger.debug(
                "Retrieval node: %d chars of context for query: %.60s",
                len(retrieval_context), query,
            )
            return {**state, "retrieval_context": retrieval_context}

        except Exception as exc:
            logger.error("Retrieval node failed: %s", exc, exc_info=True)
            try:
                db.rollback()
            except Exception:
                pass
            return {**state, "retrieval_context": ""}

    return retrieval_node


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
                        pass
            break
    return {**state, "document_ids": doc_ids}


def _should_continue(state: AgentState) -> str:
    """Conditional edge: route to 'tools' if the agent requested a tool call, else END."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


# ── Graph builder ────────────────────────────────────────────────────────────

def build_graph(db: Session, department_id: int | None = None):
    """Compile the LangGraph hybrid graph. Called once per run_agent() invocation.

    Flow:
        START
          → intent_router          (classify query intent)
          → retrieval_node         (always: GraphRAG local + pgvector chunks)
          → agent                  (synthesize from context; may call extra tools)
          → tools (optional loop)  (follow-up tool calls if context insufficient)
          → END
    """
    tools = make_tools(db, department_id=department_id)
    llm   = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)

    def agent_node(state: AgentState) -> AgentState:
        system_msg = build_system_message(state.get("intent", "hybrid"))
        messages   = [system_msg]

        # Inject pre-fetched retrieval context before the conversation
        retrieval_context = state.get("retrieval_context", "")
        if retrieval_context:
            messages.append(build_context_message(retrieval_context))

        messages += list(state["messages"])
        response = llm.invoke(messages)
        return {**state, "messages": [response]}

    retrieval_node = _make_retrieval_node(db, department_id)

    graph = StateGraph(AgentState)
    graph.add_node("intent_router",    intent_router)
    graph.add_node("retrieval",        retrieval_node)
    graph.add_node("agent",            agent_node)
    graph.add_node("tools",            ToolNode(tools))
    graph.add_node("parse_tool_output", _parse_tool_output)

    graph.add_edge(START,               "intent_router")
    graph.add_edge("intent_router",     "retrieval")
    graph.add_edge("retrieval",         "agent")
    graph.add_conditional_edges(
        "agent", _should_continue, {"tools": "tools", END: END}
    )
    graph.add_edge("tools",             "parse_tool_output")
    graph.add_edge("parse_tool_output", "agent")

    return graph.compile()
