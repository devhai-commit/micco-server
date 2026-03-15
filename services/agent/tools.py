import json
import logging
import re

from sqlalchemy.orm import Session
from sqlalchemy import text
from langchain_core.tools import tool, ToolException

from services.neo4j_service import neo4j_service
from services.embedding_service import embed
from models import Document

logger = logging.getLogger(__name__)

_WRITE_PATTERN = re.compile(
    r'\b(CREATE|MERGE|DELETE|SET|REMOVE|DROP|CALL|FOREACH)\b', re.IGNORECASE
)
_MAX_ROWS = 50


def make_tools(db: Session) -> list:
    """Create tool instances with `db` session captured in closure.

    Called once per run_agent() invocation. Each call returns fresh tool
    instances bound to the provided SQLAlchemy session.
    """

    @tool
    def query_knowledge_graph(cypher: str) -> str:
        """Run a read-only Cypher MATCH query on the Neo4j knowledge graph.

        Returns up to 50 rows as a JSON array. Only MATCH and RETURN
        clauses are allowed — write operations are rejected.
        """
        if _WRITE_PATTERN.search(cypher):
            raise ToolException(
                "Write operations are not allowed. Use only MATCH/RETURN."
            )
        if not neo4j_service.available:
            raise ToolException("Graph DB unavailable.")
        try:
            rows = neo4j_service.run_cypher(cypher, {})[:_MAX_ROWS]
            return json.dumps(rows, ensure_ascii=False, indent=2)
        except Exception as exc:
            raise ToolException(f"Graph query failed: {exc}") from exc

    @tool
    def search_kg_semantic(query: str, limit: int = 5) -> str:
        """Semantic search over document chunks using Neo4j vector similarity.

        Uses embeddings stored in Neo4j to find relevant document chunks.
        Returns chunk content, source document ID, and similarity score.
        Use this for content-based questions about document text.
        """
        try:
            vector = embed([query])[0]
            rows = neo4j_service.search_similar_chunks(vector, limit)
            if not rows:
                return "No matching chunks found in knowledge graph."
            lines = []
            doc_ids = []
            for i, row in enumerate(rows, 1):
                doc_ids.append(str(row.get("document_id", "")))
                lines.append(
                    f"Chunk {i} (doc_id={row.get('document_id')}, "
                    f"similarity={row.get('similarity', 0):.3f}):\n{row.get('content', '')}"
                )
            doc_id_str = ",".join(dict.fromkeys(doc_ids))
            return "DOCUMENT_IDS: " + doc_id_str + "\n---\n" + "\n\n".join(lines)
        except Exception as exc:
            logger.warning("Neo4j semantic search failed: %s", exc)
            return ""

    @tool
    def search_document_chunks(query: str, limit: int = 5) -> str:
        """Semantic search over document chunk embeddings.

        Returns top matching chunks with content and source document IDs.
        The result begins with a DOCUMENT_IDS header that the graph can use
        for follow-up Cypher queries. Returns an empty string on DB error.
        """
        try:
            vector = embed([query])[0]
            rows = db.execute(
                text("SELECT * FROM search_chunks_by_embedding(CAST(:embedding AS vector), :limit)"),
                {"embedding": str(vector), "limit": limit},
            ).fetchall()

            if not rows:
                return "No matching document chunks found."

            # Deduplicate document IDs while preserving order
            doc_ids = list(dict.fromkeys(row.document_id for row in rows))
            lines = [f"DOCUMENT_IDS: {','.join(str(i) for i in doc_ids)}", "---"]
            for i, row in enumerate(rows, 1):
                lines.append(
                    f"Chunk {i} (doc_id={row.document_id}, "
                    f"similarity={row.similarity:.3f}):\n{row.content}"
                )
            return "\n".join(lines)
        except Exception as exc:
            logger.warning("Vector search failed: %s", exc)
            return ""

    @tool
    def get_document_details(document_id: int) -> str:
        """Look up metadata for a specific document by its integer ID.

        Returns name, category, owner, date, and ingest status.
        """
        try:
            # Note: doc.owner_name lazy-loads the User relationship (N+1 if called in a loop).
            # Phase 3 should consider adding joinedload(Document.owner) if this becomes a hotspot.
            doc = db.query(Document).filter(Document.id == document_id).first()
            if doc is None:
                return "Document not found."
            return (
                f"Document ID: {doc.id}\n"
                f"Name: {doc.name}\n"
                f"Category: {doc.category}\n"
                f"Owner: {doc.owner_name}\n"
                f"Date: {doc.date}\n"
                f"Ingest status: {doc.ingest_status}"
            )
        except Exception as exc:
            raise ToolException(f"Document lookup failed: {exc}") from exc

    return [query_knowledge_graph, search_kg_semantic, search_document_chunks, get_document_details]
