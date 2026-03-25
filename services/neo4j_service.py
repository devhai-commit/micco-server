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

# Allowlist of attribute keys per label (prevents arbitrary property injection)
_ALLOWED_ATTRS: dict[str, set[str]] = {
    "NhaCungCap":    {"dia_chi", "ma_so_thue", "dien_thoai"},
    "HopDong":       {"so_van_ban", "ngay", "gia_tri", "hinh_thuc", "thoi_han"},
    "VatTu":         {"ma_vat_tu", "quy_cach", "don_vi_tinh", "don_gia", "xuat_xu"},
    "NguoiKiemTra":  {"chuc_vu", "phong_ban"},
    "ChungChi":      {"so_van_ban", "ngay", "co_quan_ban_hanh"},
    "QuyDinh":       {"so_van_ban", "ngay", "co_quan_ban_hanh"},
    "Kho":           {"dia_chi"},
    "CongTruong":    {"dia_chi"},
    "ChaoGia":       {"ngay", "gia_tri", "hieu_luc"},
    "SuCo":          {"ngay", "gia_tri", "tai_san"},
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
    # Knowledge categories → TriThuc
    "Chung":        "TriThuc",
    "Hướng dẫn":    "TriThuc",
    "Tiêu chuẩn":   "TriThuc",
    "Kinh nghiệm":  "TriThuc",
    "Kỹ thuật":     "TriThuc",
    "An toàn":       "TriThuc",
    "Vật tư":        "VatTu",
    "Nhà cung cấp":  "NhaCungCap",
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
            "SET n.ten = $ten, n.owner = $owner, n.created_at = $created_at, "
            "n.department_id = $department_id"
        )
        with self._driver.session() as session:
            session.run(
                cypher,
                document_id=doc["document_id"],
                ten=doc.get("ten", ""),
                owner=doc.get("owner", ""),
                created_at=doc.get("created_at", ""),
                department_id=doc.get("department_id"),
            )

    def create_entity_graph(
        self,
        document_id: int,
        entities: list[dict],
        relationships: list[dict],
        source_label: str = "Document",
    ) -> None:
        """MERGE domain entities + relationships into Neo4j.

        Links each entity back to its source node via MENTIONS.
        source_label can be "Document" or "TriThuc" (for knowledge entries).
        Silently skips entries with invalid labels or relation types.
        No-op if Neo4j is unavailable.
        """
        if not self.available:
            return
        if source_label not in _ALLOWED_LABELS:
            logger.warning("Invalid source_label=%r for entity graph", source_label)
            return
        with self._driver.session() as session:
            # 1. MERGE entity nodes with attributes
            for entity in entities:
                label = entity.get("label", "")
                name = entity.get("name", "")
                if label not in _ALLOWED_LABELS or not name:
                    continue
                # Filter attributes through allowlist
                raw_attrs = entity.get("attributes", {})
                allowed = _ALLOWED_ATTRS.get(label, set())
                attrs = {
                    k: str(v) for k, v in raw_attrs.items()
                    if k in allowed and v
                }
                # Build SET clause dynamically
                set_parts = ["e.last_seen = datetime()"]
                params: dict = {"name": name}
                for k, v in attrs.items():
                    set_parts.append(f"e.{k} = ${k}")
                    params[k] = v
                set_clause = ", ".join(set_parts)
                session.run(
                    f"MERGE (e:{label} {{name: $name}}) SET {set_clause}",
                    **params,
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

            # 3. Link entities to source node via MENTIONS (infrastructure edge)
            for entity in entities:
                label = entity.get("label", "")
                name = entity.get("name", "")
                if label not in _ALLOWED_LABELS or not name:
                    continue
                session.run(
                    f"MATCH (doc:{source_label} {{document_id: $doc_id}}) "
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
