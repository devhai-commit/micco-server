import logging
import os
from neo4j import GraphDatabase
from kg.ontology import NodeLabel, RelType

logger = logging.getLogger(__name__)

# Allowlist of valid Neo4j labels (prevents Cypher label injection)
_ALLOWED_LABELS = {
    label.value for label in NodeLabel
    if label not in (NodeLabel.DOCUMENT, NodeLabel.DOCUMENT_CHUNK)
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

    def create_chunk_node(self, document_id: int, chunk_idx: int) -> None:
        if not self.available:
            return
        rel = RelType.HAS_CHUNK.value
        cypher = f"""
            MERGE (doc:Document {{document_id: $document_id}})
            MERGE (chunk:DocumentChunk {{document_id: $document_id, chunk_index: $chunk_index}})
            MERGE (doc)-[:{rel}]->(chunk)
        """
        with self._driver.session() as session:
            session.run(cypher, document_id=document_id, chunk_index=chunk_idx)

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

    def run_cypher(self, query: str, params: dict | None = None) -> list[dict]:
        if not self.available:
            return []
        with self._driver.session() as session:
            result = session.run(query, **(params or {}))
            return [dict(record) for record in result]


neo4j_service = Neo4jService()
