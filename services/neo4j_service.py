import logging
import os
from neo4j import GraphDatabase
from kg.ontology import RelType

logger = logging.getLogger(__name__)

# Allowlist of valid Neo4j labels (prevents Cypher label injection)
_ALLOWED_LABELS = {
    "VatTu", "HopDong", "QuyDinh", "SuCo", "KeHoachMuaSam",
    "PhieuNhapKho", "ChungChi",
}

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

    def run_cypher(self, query: str, params: dict | None = None) -> list[dict]:
        if not self.available:
            return []
        with self._driver.session() as session:
            result = session.run(query, **(params or {}))
            return [dict(record) for record in result]


neo4j_service = Neo4jService()
