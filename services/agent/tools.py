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


def make_tools(db: Session, department_id: int | None = None) -> list:
    """Create tool instances with `db` session and department scope captured in closure.

    department_id controls data isolation:
      - int  → filter all queries to that department only
      - None → Admin mode, no filtering (sees everything)

    Called once per run_agent() invocation. Each call returns fresh tool
    instances bound to the provided SQLAlchemy session.
    """

    # ── Helper: build Neo4j department filter clause ──────────
    def _neo4j_dept_filter(node_var: str = "doc") -> str:
        """Return a Cypher WHERE fragment that restricts to the user's department.

        Uses the document_id property on Document/TriThuc nodes to JOIN back
        to PostgreSQL documents/knowledge_entries (which carry department_id).
        For simplicity, filters on the document_id property stored on Neo4j nodes.
        """
        if department_id is None:
            return ""
        return f" AND {node_var}.department_id = {department_id}"

    @tool
    def query_knowledge_graph(cypher: str) -> str:
        """Run a read-only Cypher MATCH query on the Neo4j knowledge graph.

        Results are scoped to the current user's department.
        Returns up to 50 rows as a JSON array. Only MATCH and RETURN
        clauses are allowed — write operations are rejected.
        """
        if _WRITE_PATTERN.search(cypher):
            raise ToolException(
                "Write operations are not allowed. Use only MATCH/RETURN."
            )
        if not neo4j_service.available:
            raise ToolException("Graph DB unavailable.")

        # Auto-fix: If type(rel) is used but rel variable not defined
        if "type(rel)" in cypher and "-[:" not in cypher.replace("-[:", "-[rel:"):
            match = re.search(r'-\[(\w+):(\w+)\]->', cypher)
            if match:
                var_name, rel_type = match.groups()
                if var_name != "rel":
                    cypher = cypher.replace("type(rel)", f"'{rel_type}'")

        # Inject department filter: if query touches Document/TriThuc nodes
        if department_id is not None:
            # Append department filter for document-linked queries
            dept_filter = (
                f" AND (EXISTS {{ MATCH (doc)-[:MENTIONS]-(n) "
                f"WHERE doc.department_id = {department_id} }}"
                f" OR n.department_id = {department_id})"
            )
            # Only inject if query has a WHERE clause (safe heuristic)
            if "WHERE" in cypher.upper():
                cypher = cypher.replace("RETURN", f"{dept_filter}\nRETURN", 1)

        try:
            rows = neo4j_service.run_cypher(cypher, {})[:_MAX_ROWS]
            return json.dumps(rows, ensure_ascii=False, indent=2)
        except Exception as exc:
            raise ToolException(f"Graph query failed: {exc}") from exc

    @tool
    def search_document_chunks(query: str, limit: int = 5) -> str:
        """Semantic search over all chunk embeddings (documents + knowledge).

        Results are scoped to the current user's department.
        """
        try:
            vector = embed([query])[0]
            rows = db.execute(
                text("SELECT * FROM search_chunks_by_embedding(CAST(:embedding AS vector), :dept_id, :limit)"),
                {"embedding": str(vector), "dept_id": department_id, "limit": limit},
            ).fetchall()

            if not rows:
                return "No matching chunks found."

            doc_ids = list(dict.fromkeys(row.source_id for row in rows if row.source_type == "document"))
            lines = [f"DOCUMENT_IDS: {','.join(str(i) for i in doc_ids)}", "---"]
            for i, row in enumerate(rows, 1):
                lines.append(
                    f"Chunk {i} (source={row.source_type}, id={row.source_id}, "
                    f"name={row.source_name}, similarity={row.similarity:.3f}):\n{row.chunk_content}"
                )
            return "\n".join(lines)
        except Exception as exc:
            logger.warning("Vector search failed: %s", exc)
            return ""

    @tool
    def get_document_details(document_id: int) -> str:
        """Look up metadata for a specific document by its integer ID.

        Returns only documents belonging to the current user's department.
        """
        try:
            query = db.query(Document).filter(Document.id == document_id)
            if department_id is not None:
                query = query.filter(Document.department_id == department_id)
            doc = query.first()
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

    @tool
    def search_kg_flexible(keywords: str, limit: int = 10) -> str:
        """Flexible search in knowledge graph without knowing exact labels/relationships.

        Results are scoped to the current user's department.
        """
        if not neo4j_service.available:
            raise ToolException("Graph DB unavailable.")
        try:
            kw = keywords.lower().strip()

            # Department filter for Neo4j: restrict to nodes linked to dept's documents
            dept_clause = ""
            if department_id is not None:
                dept_clause = f" AND (n.department_id = {department_id} OR m.department_id = {department_id})"

            # Search for nodes by name containing keywords
            cypher = f"""
            MATCH (n)-[r]-(m)
            WHERE (toLower(n.name) CONTAINS '{kw}' OR toLower(m.name) CONTAINS '{kw}'){dept_clause}
            RETURN n.name AS source, labels(n)[0] AS source_type,
                   type(r) AS relation, m.name AS target, labels(m)[0] AS target_type
            LIMIT {limit * 2}
            """
            rows = neo4j_service.run_cypher(cypher, {})

            if not rows:
                label_map = {
                    "công ty": "NhaCungCap", "nhà cung cấp": "NhaCungCap", "ncc": "NhaCungCap",
                    "nhà máy": "NhaSanXuat", "sản xuất": "NhaSanXuat",
                    "vật tư": "VatTu", "vật liệu": "VatTu",
                    "hợp đồng": "HopDong",
                    "đơn hàng": "DonHang",
                    "báo giá": "ChaoGia", "chào giá": "ChaoGia",
                    "kho": "Kho",
                    "chứng chỉ": "ChungChi", "chứng nhận": "ChungChi",
                    "quy định": "QuyDinh",
                    "sự cố": "SuCo",
                    "kế hoạch": "KeHoachMuaSam",
                }

                found_label = None
                for key, label in label_map.items():
                    if key in kw:
                        found_label = label
                        break

                if found_label:
                    cypher = f"""
                    MATCH (n:{found_label})-[r]-(m)
                    WHERE true{dept_clause}
                    RETURN n.name AS source, labels(n)[0] AS source_type,
                           type(r) AS relation, m.name AS target, labels(m)[0] AS target_type
                    LIMIT {limit * 2}
                    """
                    rows = neo4j_service.run_cypher(cypher, {})

            if not rows:
                rel_keywords = ["cung cap", "cấp", "cung cấp", "mua", "bán", "chào giá",
                               "yêu cầu", "cần", "theo", "liên quan", "sản xuất", "nhập", "xuất"]
                for rkw in rel_keywords:
                    if rkw in kw:
                        cypher = f"""
                        MATCH (n)-[r]-(m)
                        WHERE toLower(type(r)) CONTAINS '{rkw}'{dept_clause}
                        RETURN n.name AS source, labels(n)[0] AS source_type,
                               type(r) AS relation, m.name AS target, labels(m)[0] AS target_type
                        LIMIT {limit * 2}
                        """
                        rows = neo4j_service.run_cypher(cypher, {})
                        if rows:
                            break

            if not rows:
                vector = embed([keywords])[0]
                pg_rows = db.execute(
                    text("SELECT * FROM search_chunks_by_embedding(CAST(:embedding AS vector), :dept_id, :limit)"),
                    {"embedding": str(vector), "dept_id": department_id, "limit": limit},
                ).fetchall()
                if pg_rows:
                    lines = [f"Found {len(pg_rows)} relevant document chunks:", ""]
                    for i, r in enumerate(pg_rows, 1):
                        lines.append(f"{i}. {r.source_type}_id={r.source_id}: {r.chunk_content[:200]}...")
                    return "\n".join(lines)
                return f"Không tìm thấy kết quả nào cho từ khóa: {keywords}"

            unique_results = []
            seen = set()
            for row in rows:
                key = (row.get("source", ""), row.get("relation", ""), row.get("target", ""))
                if key not in seen and row.get("source") and row.get("target"):
                    seen.add(key)
                    unique_results.append(row)

            unique_results = unique_results[:limit]

            lines = [f"Tìm thấy {len(unique_results)} kết quả cho '{keywords}':", ""]
            for i, row in enumerate(unique_results, 1):
                lines.append(
                    f"{i}. {row.get('source', '')} --[{row.get('relation', '')}]--> {row.get('target', '')}"
                )
            return "\n".join(lines)
        except Exception as exc:
            logger.warning("Flexible search failed: %s", exc)
            return f"Tìm kiếm thất bại: {exc}"

    @tool
    def list_kg_schema() -> str:
        """List all available node labels and relationship types in the knowledge graph.

        Returns schema information scoped to the current user's department.
        """
        if not neo4j_service.available:
            raise ToolException("Graph DB unavailable.")
        try:
            label_cypher = "CALL db.labels() YIELD label RETURN label"
            labels = neo4j_service.run_cypher(label_cypher, {})

            rel_cypher = "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
            rels = neo4j_service.run_cypher(rel_cypher, {})

            # Sample nodes: scope by department
            dept_where = f"AND n.department_id = {department_id}" if department_id is not None else ""
            sample_cypher = f"""
            MATCH (n)
            WHERE n.name IS NOT NULL {dept_where}
            RETURN labels(n)[0] AS label, n.name AS name
            LIMIT 20
            """
            samples = neo4j_service.run_cypher(sample_cypher, {})

            lines = ["=== NODE LABELS ===", ", ".join([l.get("label", "") for l in labels])]
            lines.append("")
            lines.append("=== RELATIONSHIP TYPES ===")
            lines.append(", ".join([r.get("relationshipType", "") for r in rels]))
            lines.append("")
            lines.append("=== SAMPLE NODES ===")
            for s in samples:
                lines.append(f"- [{s.get('label', '')}] {s.get('name', '')}")

            return "\n".join(lines)
        except Exception as exc:
            return f"Failed to list schema: {exc}"

    @tool
    def llm_reasoning(question: str) -> str:
        """Analyze and reason about a user question to determine the best search strategy."""
        try:
            from openai import OpenAI
            client = OpenAI()

            prompt = f"""Bạn là một trợ lý AI phân tích câu hỏi để tìm kiếm thông tin trong knowledge graph.

Hãy phân tích câu hỏi sau và trả lời theo format JSON:

{{
    "cau_hoi_goc": "{question}",
    "thuc_the_chinh": "Tên các thực thể chính (công ty, vật tư, hợp đồng...)",
    "loai_thong_tin": "loại thông tin cần tìm (quan hệ, nội dung, thống kê...)",
    "tu_khoa_tim_kiem": "các từ khóa để tìm kiếm trong graph",
    "cau_hoi_chuan_hoa": "câu hỏi được chuẩn hóa",
    "giai_phap_de_xuat": "nên dùng tool nào (query_knowledge_graph / search_kg_flexible / search_document_chunks)"
}}

CHỈ trả JSON, không giải thích gì thêm."""

            response = client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "Bạn là chuyên gia phân tích câu hỏi để tìm kiếm trong knowledge graph"},
                    {"role": "user", "content": prompt}
                ],
            )

            result = response.choices[0].message.content
            return result
        except Exception as exc:
            logger.warning("LLM reasoning failed: %s", exc)
            return f"Phân tích thất bại: {exc}"

    # ── GraphRAG Local Search ────────────────────────────────────

    @tool
    def search_local(query: str, top_k: int = 5, chunk_limit: int = 5) -> str:
        """GraphRAG Local Search: find entities by semantic similarity, then
        traverse the knowledge graph to collect relationships, community context,
        and source document chunks.

        Best for specific questions about entities, suppliers, contracts, materials.
        """
        try:
            # Step 1: Vector similarity on entity embeddings (PostgreSQL)
            vector = embed([query])[0]
            try:
                entity_rows = db.execute(
                    text("SELECT * FROM search_entities_by_embedding(CAST(:emb AS vector), :limit)"),
                    {"emb": str(vector), "limit": top_k},
                ).fetchall()
            except Exception as entity_exc:
                db.rollback()
                logger.warning("Entity embedding search unavailable, falling back to chunk search: %s", entity_exc)
                # Fallback: skip entity step, go straight to chunk search
                entity_rows = []

            entity_names = [r.entity_name for r in entity_rows]

            # Step 2: Graph traversal — get relationships for matched entities
            graph_context_parts = ["=== MATCHED ENTITIES ==="]
            for r in entity_rows:
                graph_context_parts.append(
                    f"- {r.entity_label}: {r.entity_name} (similarity={r.similarity:.3f})"
                )
                if r.description:
                    graph_context_parts.append(f"  {r.description}")

            # Fetch relationships from Neo4j for matched entities
            if neo4j_service.available:
                placeholders = ", ".join(f"'{n}'" for n in entity_names)
                rel_cypher = f"""
                MATCH (a)-[r]-(b)
                WHERE a.name IN [{placeholders}] OR b.name IN [{placeholders}]
                RETURN a.name AS source, labels(a)[0] AS source_type,
                       type(r) AS relation,
                       b.name AS target, labels(b)[0] AS target_type
                LIMIT 30
                """
                rels = neo4j_service.run_cypher(rel_cypher, {})
                if rels:
                    graph_context_parts.append("\n=== RELATIONSHIPS ===")
                    seen = set()
                    for rel in rels:
                        key = (rel.get("source"), rel.get("relation"), rel.get("target"))
                        if key not in seen:
                            seen.add(key)
                            graph_context_parts.append(
                                f"- {rel['source']} ({rel.get('source_type','')}) "
                                f"--[{rel['relation']}]--> "
                                f"{rel['target']} ({rel.get('target_type','')})"
                            )

                # Step 3: Community context — find communities containing matched entities
                community_rows = db.execute(
                    text("""
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
                    graph_context_parts.append("\n=== COMMUNITY CONTEXT ===")
                    for cr in community_rows:
                        graph_context_parts.append(f"- {cr.title}: {cr.summary}")

            # Step 4: Source text chunks from PostgreSQL (via MENTIONS links)
            chunk_rows = db.execute(
                text("""
                    SELECT * FROM search_chunks_by_embedding(
                        CAST(:emb AS vector), :dept_id, :limit
                    )
                """),
                {"emb": str(vector), "dept_id": department_id, "limit": chunk_limit},
            ).fetchall()

            doc_ids = []
            if chunk_rows:
                graph_context_parts.append("\n=== SOURCE TEXT CHUNKS ===")
                for i, cr in enumerate(chunk_rows, 1):
                    if cr.source_type == "document":
                        doc_ids.append(str(cr.source_id))
                    graph_context_parts.append(
                        f"Chunk {i} ({cr.source_type} id={cr.source_id}, "
                        f"name={cr.source_name}, sim={cr.similarity:.3f}):\n{cr.chunk_content}"
                    )

            header = "DOCUMENT_IDS: " + ",".join(dict.fromkeys(doc_ids)) if doc_ids else ""
            return header + "\n" + "\n".join(graph_context_parts)

        except Exception as exc:
            logger.warning("Local search failed: %s", exc)
            return f"Local search failed: {exc}"

    # ── GraphRAG Global Search ────────────────────────────────────

    @tool
    def search_global(query: str, top_k: int = 5) -> str:
        """GraphRAG Global Search: search over community summaries for broad,
        thematic questions that span multiple entities or the whole corpus.

        Best for overview questions: trends, comparisons, summaries.
        """
        try:
            vector = embed([query])[0]

            # Vector similarity on community summary embeddings
            rows = db.execute(
                text("""
                    SELECT c.id, c.title, c.summary, c.full_content,
                           c.entity_count, c.relationship_count,
                           1 - (c.embedding <=> CAST(:emb AS vector))::FLOAT AS similarity
                    FROM communities c
                    WHERE c.embedding IS NOT NULL
                    ORDER BY c.embedding <=> CAST(:emb AS vector) ASC
                    LIMIT :limit
                """),
                {"emb": str(vector), "limit": top_k},
            ).fetchall()

            if not rows:
                return "No community data available. Run community detection first."

            lines = [f"Found {len(rows)} relevant communities:\n"]
            for i, r in enumerate(rows, 1):
                lines.append(
                    f"--- Community {i}: {r.title} "
                    f"(entities={r.entity_count}, rels={r.relationship_count}, "
                    f"similarity={r.similarity:.3f}) ---"
                )
                lines.append(r.full_content or r.summary or "")
                lines.append("")

            return "\n".join(lines)

        except Exception as exc:
            logger.warning("Global search failed: %s", exc)
            return f"Global search failed: {exc}"

    return [
        query_knowledge_graph, search_kg_flexible, list_kg_schema,
        llm_reasoning, search_document_chunks, get_document_details,
        search_local, search_global,
    ]
