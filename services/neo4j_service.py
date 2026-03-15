import logging
import os
from neo4j import GraphDatabase
from kg.ontology import NodeLabel, RelType

logger = logging.getLogger(__name__)

# Allowlist of valid Neo4j labels (prevents Cypher label injection)
_ALLOWED_LABELS = {
    label.value for label in NodeLabel
}

_DOMAIN_RELS = {rel.value for rel in RelType}

CATEGORY_LABEL_MAP: dict[str, str] = {
    "VatTu":       "VatTu",
    "Tài liệu":    "VatTu",
    "HopDong":     "HopDong",
    "Hợp đồng":   "HopDong",
    "QuyDinh":     "QuyDinh",
    "Quy trình":   "QuyDinh",
    "BaoCao":      "SuCo",
    "Báo cáo":     "SuCo",
    "Report":      "SuCo",
    "Spreadsheet": "KeHoachMuaSam",
    "Kế hoạch":    "KeHoachMuaSam",
    "Biên bản":    "PhieuNhapKho",
    "ChungChi":    "ChungChi",
    "Certificate": "ChungChi",
}


def category_to_label(category: str | None) -> str:
    return CATEGORY_LABEL_MAP.get(category or "", "VatTu")


class Neo4jService:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "")
        self._driver = None
        self.available = False

    def connect(self) -> None:
        try:
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self._driver.verify_connectivity()
            self.available = True
            logger.info("Neo4j connected: %s", self.uri)
        except Exception as exc:
            self.available = False
            logger.error("Neo4j connection failed: %s", exc)

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self.available = False

    def merge_document_node(self, doc: dict) -> None:
        if not self.available:
            return
        label = doc["label"]
        if label not in _ALLOWED_LABELS:
            raise ValueError(f"Unexpected Neo4j label: {label!r}")
        cypher = (
            f"MERGE (n:{label} {{document_id: $document_id}}) "
            "SET n.ten = $ten, n.owner = $owner, n.created_at = $created_at"
        )
        with self._driver.session() as session:
            session.run(
                cypher,
                document_id=doc["document_id"],
                ten=doc.get("ten", ""),
                owner=doc.get("owner", ""),
                created_at=doc.get("created_at", ""),
            )

    def create_entity_graph(
        self,
        document_id: int,
        entities: list[dict],
        relationships: list[dict],
    ) -> None:
        """MERGE domain entities + relationships into Neo4j.

        Links each entity back to its source Document node via MENTIONS.
        Silently skips entries with invalid labels or relation types.
        No-op if Neo4j is unavailable.
        """
        if not self.available:
            return
        with self._driver.session() as session:
            # 1. MERGE entity nodes
            for entity in entities:
                label = entity.get("label", "")
                name = entity.get("name", "")
                if label not in _ALLOWED_LABELS or not name:
                    continue
                session.run(
                    f"MERGE (e:{label} {{name: $name}}) SET e.last_seen = datetime()",
                    name=name,
                )

            # 2. MERGE relationships
            for rel in relationships:
                s_label = rel.get("source_label", "")
                t_label = rel.get("target_label", "")
                rel_type = rel.get("relation", "")
                source = rel.get("source", "")
                target = rel.get("target", "")
                if (
                    s_label not in _ALLOWED_LABELS
                    or t_label not in _ALLOWED_LABELS
                    or rel_type not in _DOMAIN_RELS
                    or not source
                    or not target
                ):
                    continue
                session.run(
                    f"MATCH (s:{s_label} {{name: $source}}) "
                    f"MATCH (t:{t_label} {{name: $target}}) "
                    f"MERGE (s)-[:{rel_type}]->(t)",
                    source=source,
                    target=target,
                )

            # 3. Link entities to document via MENTIONS (infrastructure edge)
            for entity in entities:
                label = entity.get("label", "")
                name = entity.get("name", "")
                if label not in _ALLOWED_LABELS or not name:
                    continue
                session.run(
                    f"MATCH (doc:Document {{document_id: $doc_id}}) "
                    f"MATCH (e:{label} {{name: $name}}) "
                    "MERGE (doc)-[:MENTIONS]->(e)",
                    doc_id=document_id,
                    name=name,
                )

    def merge_document_chunk(
        self,
        document_id: int,
        chunk_index: int,
        content: str,
        embedding: list[float],
    ) -> None:
        """Create a DocumentChunk node with embedding for semantic search."""
        if not self.available:
            return
        cypher = """
            MERGE (c:DocumentChunk {
                document_id: $document_id,
                chunk_index: $chunk_index
            })
            SET c.content = $content,
                c.embedding = $embedding
            WITH c
            MATCH (d:Document {document_id: $document_id})
            MERGE (d)-[:HAS_CHUNK]->(c)
        """
        with self._driver.session() as session:
            session.run(
                cypher,
                document_id=document_id,
                chunk_index=chunk_index,
                content=content,
                embedding=embedding,
            )

    def search_similar_chunks(
        self,
        query_embedding: list[float],
        limit: int = 5,
    ) -> list[dict]:
        """Semantic search over DocumentChunk embeddings in Neo4j."""
        if not self.available:
            return []
        cypher = """
            MATCH (c:DocumentChunk)
            WHERE c.embedding IS NOT NULL
            RETURN c.document_id AS document_id,
                   c.chunk_index AS chunk_index,
                   c.content AS content,
                   apoc.vectors.similarity(c.embedding, $embedding) AS similarity
            ORDER BY similarity DESC
            LIMIT $limit
        """
        with self._driver.session() as session:
            result = session.run(cypher, embedding=query_embedding, limit=limit)
            return [dict(record) for record in result]

    def run_cypher(self, query: str, params: dict | None = None) -> list[dict]:
        if not self.available:
            return []
        with self._driver.session() as session:
            result = session.run(query, **(params or {}))
            return [dict(record) for record in result]


neo4j_service = Neo4jService()
