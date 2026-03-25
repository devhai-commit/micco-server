import json
import logging
from kg.ontology import NodeLabel, RelType

logger = logging.getLogger(__name__)

_DOMAIN_LABELS: set[str] = {
    label.value for label in NodeLabel
    if label is not NodeLabel.DOCUMENT
}

_DOMAIN_RELS: set[str] = {
    rel.value for rel in RelType
}

_SYSTEM_PROMPT = (
    "You are an expert entity and relationship extractor for a Vietnamese procurement and supply chain document management system.\n"
    "Extract entities and relationships from Vietnamese business documents such as:\n"
    "- Hồ sơ mua sắm (procurement documents)\n"
    "- Hợp đồng (contracts)\n"
    "- Đơn chào hàng (quotations)\n"
    "- Giấy chứng nhận đăng ký doanh nghiệp (business registration)\n"
    "- Hồ sơ sửa chữa (repair records)\n"
    "- Hóa đơn (invoices)\n\n"
    "VALID ENTITY LABELS: " + ", ".join(sorted(_DOMAIN_LABELS)) + "\n"
    "VALID RELATIONSHIP TYPES: " + ", ".join(sorted(_DOMAIN_RELS)) + "\n\n"
    "EXTRACT THESE ENTITIES (use exact label from above) with attributes:\n"
    "- NhaCungCap: Suppliers, companies. Attrs: dia_chi, ma_so_thue, dien_thoai\n"
    "- HopDong: Contracts, orders. Attrs: so_van_ban, ngay, gia_tri, hinh_thuc, thoi_han\n"
    "- VatTu: Materials, products, spare parts. Attrs: ma_vat_tu, quy_cach, don_vi_tinh, don_gia, xuat_xu\n"
    "- NguoiKiemTra: People (directors, managers, staff). Attrs: chuc_vu, phong_ban\n"
    "- ChungChi: Certificates, licenses. Attrs: so_van_ban, ngay, co_quan_ban_hanh\n"
    "- QuyDinh: Regulations, decisions. Attrs: so_van_ban, ngay, co_quan_ban_hanh\n"
    "- Kho: Warehouses, delivery locations. Attrs: dia_chi\n"
    "- ChaoGia: Quotations. Attrs: ngay, gia_tri, hieu_luc\n"
    "- SuCo: Incidents, repair reports. Attrs: ngay, gia_tri, tai_san\n\n"
    "EXTRACT THESE RELATIONSHIPS (use exact type from above):\n"
    "- CUNG_CAP: Company supplies materials/products to another\n"
    "- CO_DON_HANG: Company has an order/contract\n"
    "- BAO_GOM: Contract contains materials/products\n"
    "- GIAO_HANG: Contract/order ships to location\n"
    "- SAN_XUAT_BOI: Manufacturer produces products\n"
    "- CO_CHUNG_CHI: Company holds certificates\n"
    "- THUOC_PHONG_BAN: Person belongs to company/department\n"
    "- DUOC_DUYET_BOI: Document/person approved by another person\n"
    "- NHAP_VAT_TU: Company imports materials\n\n"
    "Return JSON with exact schema:\n"
    '{"entities": [{"name": "...", "label": "...", "attributes": {"key": "value", ...}}], '
    '"relationships": [{"source": "...", "source_label": "...", '
    '"relation": "...", "target": "...", "target_label": "..."}]}\n'
    "Rules:\n"
    "- Only use labels and relationship types from the valid lists above.\n"
    "- Extract ALL entities and relationships clearly mentioned in the text.\n"
    "- For Vietnamese company names, use the full official name.\n"
    "- Only include attributes that are explicitly stated in the text. Omit unknown attributes.\n"
    "- Attribute values must be strings. Keep monetary values with units (e.g. '322.272.000 VNĐ')."
)

_ENTITY_ONLY_PROMPT = (
    "You are an expert entity extractor for Vietnamese business documents.\n"
    "Extract ONLY entities (NO relationships) from the text below.\n\n"
    "VALID ENTITY LABELS: " + ", ".join(sorted(_DOMAIN_LABELS)) + "\n\n"
    "For each entity, provide:\n"
    "- name: The entity name (full Vietnamese name for companies, specific names for items)\n"
    "- label: One of the valid labels above\n"
    "- attributes: A dict of key-value pairs with additional info found in the text\n\n"
    "ALLOWED ATTRIBUTES PER LABEL:\n"
    "- NhaCungCap: dia_chi, ma_so_thue, dien_thoai\n"
    "- HopDong: so_van_ban, ngay, gia_tri, hinh_thuc, thoi_han\n"
    "- VatTu: ma_vat_tu, quy_cach, don_vi_tinh, don_gia, xuat_xu\n"
    "- NguoiKiemTra: chuc_vu, phong_ban\n"
    "- ChungChi: so_van_ban, ngay, co_quan_ban_hanh\n"
    "- QuyDinh: so_van_ban, ngay, co_quan_ban_hanh\n"
    "- Kho: dia_chi\n"
    "- ChaoGia: ngay, gia_tri, hieu_luc\n"
    "- SuCo: ngay, gia_tri, tai_san\n\n"
    'Return JSON: {"entities": [{"name": "...", "label": "...", "attributes": {"key": "value"}}]}\n'
    "Rules:\n"
    "- Only use labels from the valid list.\n"
    "- Only include attributes explicitly stated in the text. Omit unknown ones.\n"
    "- All attribute values must be strings.\n"
    "- Extract ALL entities mentioned in the text."
)

_RELATIONSHIP_ONLY_PROMPT = (
    "You are an expert relationship extractor for Vietnamese business documents.\n"
    "Given a list of entities and the full document text, extract ONLY relationships between them.\n\n"
    "VALID ENTITY LABELS: " + ", ".join(sorted(_DOMAIN_LABELS)) + "\n"
    "VALID RELATIONSHIP TYPES: " + ", ".join(sorted(_DOMAIN_RELS)) + "\n\n"
    "Rules:\n"
    "1. Only create relationships explicitly mentioned or clearly implied in the text\n"
    "2. Use exact entity names from the provided list\n"
    "3. Only use relationship types from the valid list above\n"
    "4. Do NOT create relationships that are not supported by the text\n\n"
    "Return JSON:\n"
    '{"relationships": [{"source": "...", "source_label": "...", '
    '"relation": "...", "target": "...", "target_label": "..."}]}\n'
    "If no relationships are found, return {\"relationships\": []}"
)


def extract_kg(chunks: list[str], doc) -> dict:
    """Two-phase extraction:
    1. Extract all entities from all chunks (aggregate)
    2. Extract relationships based on full context + known entities

    Returns {"entities": [...], "relationships": [...]} or {} on any error.
    Never raises — extraction failure must not block ingest.
    """
    if not chunks:
        logger.warning("extract_kg called with empty chunks for doc_id=%s", getattr(doc, "id", "?"))
        return {}
    try:
        from openai import OpenAI
        client = OpenAI()  # lazy: raises here if OPENAI_API_KEY missing; caught below
        logger.info("OpenAI client initialized for doc_id=%s, processing %d chunks",
                    getattr(doc, "id", "?"), len(chunks))

        # ── Phase 1: Extract entities from chunks (batched) ────────
        all_entities: list[dict] = []
        ENTITY_BATCH_SIZE = 20  # ~20 chunks × 512 chars ≈ 10K chars, well within 128K context

        total_batches = (len(chunks) + ENTITY_BATCH_SIZE - 1) // ENTITY_BATCH_SIZE
        for batch_idx in range(0, len(chunks), ENTITY_BATCH_SIZE):
            batch_chunks = chunks[batch_idx:batch_idx + ENTITY_BATCH_SIZE]
            batch_text = "\n\n---\n\n".join(batch_chunks)
            batch_num = batch_idx // ENTITY_BATCH_SIZE + 1

            user_prompt = (
                f"Document name: {doc.name}\n"
                f"Document category: {doc.category}\n\n"
                f"Extract ONLY entities (no relationships) from the following text "
                f"(batch {batch_num}/{total_batches}, {len(batch_chunks)} sections).\n\n"
                f"Text:\n{batch_text}"
            )
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": _ENTITY_ONLY_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                )                                                                                               
                data = json.loads(response.choices[0].message.content)

                batch_count = 0
                for e in data.get("entities", []):
                    if isinstance(e, dict) and e.get("name") and e.get("label") in _DOMAIN_LABELS:
                        existing = next(
                            (x for x in all_entities if x["name"] == e["name"] and x["label"] == e["label"]),
                            None,
                        )
                        if existing is None:
                            if "attributes" not in e:
                                e["attributes"] = {}
                            all_entities.append(e)
                            batch_count += 1
                        else:
                            # Merge new attributes into existing entity
                            new_attrs = e.get("attributes", {})
                            if new_attrs and isinstance(new_attrs, dict):
                                existing.setdefault("attributes", {})
                                for k, v in new_attrs.items():
                                    if v and k not in existing["attributes"]:
                                        existing["attributes"][k] = v

                logger.info("Phase 1 - Batch %d/%d (%d chunks): +%d entities (total: %d)",
                            batch_num, total_batches, len(batch_chunks), batch_count, len(all_entities))
            except Exception as e:
                logger.warning("Phase 1 batch %d failed: %s", batch_num, e)
                continue

        if not all_entities:
            logger.warning("No entities extracted for doc_id=%s", getattr(doc, "id", "?"))
            return {}

        # ── Phase 2: Extract relationships in batches to avoid token limit ─────
        # Split chunks into groups of 50 for relationship extraction
        BATCH_SIZE = 50
        all_chunks = chunks
        relationships: list[dict] = []

        for batch_idx in range(0, len(all_chunks), BATCH_SIZE):
            batch_chunks = all_chunks[batch_idx:batch_idx + BATCH_SIZE]
            batch_text = "\n\n".join(batch_chunks)

            entity_info = "\n".join([f"- {e['name']} ({e['label']})" for e in all_entities])

            rel_prompt = (
                f"Document name: {doc.name}\n"
                f"Document category: {doc.category}\n\n"
                f"KNOWN ENTITIES:\n{entity_info}\n\n"
                f"TEXT SECTION:\n{batch_text}\n\n"
                f"Find relationships between known entities in this section only.\n"
                f"Only create relationships explicitly mentioned in this section."
            )

            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": _RELATIONSHIP_ONLY_PROMPT},
                        {"role": "user", "content": rel_prompt},
                    ],
                )
                data = json.loads(response.choices[0].message.content)

                for r in data.get("relationships", []):
                    if (isinstance(r, dict)
                        and r.get("source")
                        and r.get("source_label") in _DOMAIN_LABELS
                        and r.get("relation") in _DOMAIN_RELS
                        and r.get("target")
                        and r.get("target_label") in _DOMAIN_LABELS):
                        if not any(x["source"] == r["source"] and x["relation"] == r["relation"] and x["target"] == r["target"]
                                   for x in relationships):
                            relationships.append(r)
            except Exception as e:
                logger.warning("Phase 2 batch %d failed: %s", batch_idx // BATCH_SIZE + 1, e)
                continue

            logger.info("Phase 2 - Batch %d/%d: extracted relationships",
                        batch_idx // BATCH_SIZE + 1, (len(all_chunks) + BATCH_SIZE - 1) // BATCH_SIZE)

        logger.info("Phase 2: extracted %d relationships from %d entities",
                    len(relationships), len(all_entities))
        logger.info("Total for doc_id=%s: %d entities, %d relationships",
                     getattr(doc, "id", "?"), len(all_entities), len(relationships))

        return {"entities": all_entities, "relationships": relationships}
    except Exception as exc:
        logger.warning(
            "KG extraction failed for doc %s: %s",
            getattr(doc, "id", "?"), exc,
        )
        return {}
