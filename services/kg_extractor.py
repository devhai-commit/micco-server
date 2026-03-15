import json
import logging
from kg.ontology import NodeLabel, RelType

logger = logging.getLogger(__name__)

_DOMAIN_LABELS: set[str] = {
    label.value for label in NodeLabel
    if label not in (NodeLabel.DOCUMENT, NodeLabel.DOCUMENT_CHUNK)
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
    "EXTRACT THESE ENTITIES (use exact label from above):\n"
    "- NhaCungCap: Suppliers, customers, companies (with name, address, tax code)\n"
    "- HopDong: Contracts, orders with numbers, dates, values\n"
    "- VatTu: Materials, products, equipment, spare parts\n"
    "- NguoiKiemTra: People with positions (directors, managers, staff)\n"
    "- ChungChi: Certificates, business licenses, authorization letters\n"
    "- Kho: Warehouses, delivery locations, storage facilities\n\n"
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
    '{"entities": [{"name": "...", "label": "..."}], '
    '"relationships": [{"source": "...", "source_label": "...", '
    '"relation": "...", "target": "...", "target_label": "..."}]}\n'
    "Only use labels and relationship types from the valid lists above. "
    "Extract ALL entities and relationships clearly mentioned in the text. "
    "For Vietnamese company names, use the full official name."
)

_ENTITY_ONLY_PROMPT = (
    "You are an expert entity extractor for Vietnamese business documents.\n"
    "Extract ONLY entities (NO relationships) from the text below.\n\n"
    "VALID ENTITY LABELS: " + ", ".join(sorted(_DOMAIN_LABELS)) + "\n\n"
    "For each entity, provide:\n"
    "- name: The entity name (full Vietnamese name for companies, specific names for items)\n"
    "- label: One of the valid labels above\n\n"
    'Return JSON: {"entities": [{"name": "...", "label": "..."}]}\n'
    "Only use labels from the valid list. Extract ALL entities mentioned in the text."
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

        # ── Phase 1: Extract entities from all chunks ───────────────
        all_entities: list[dict] = []

        for i, chunk_text in enumerate(chunks):
            user_prompt = (
                f"Document name: {doc.name}\n"
                f"Document category: {doc.category}\n\n"
                f"Extract ONLY entities (no relationships) from this text chunk.\n\n"
                f"Text chunk {i+1}/{len(chunks)}:\n{chunk_text}"
            )
            response = client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _ENTITY_ONLY_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            data = json.loads(response.choices[0].message.content)

            for e in data.get("entities", []):
                if isinstance(e, dict) and e.get("name") and e.get("label") in _DOMAIN_LABELS:
                    if not any(x["name"] == e["name"] and x["label"] == e["label"] for x in all_entities):
                        all_entities.append(e)

            logger.info("Phase 1 - Chunk %d/%d: extracted %d entities",
                        i+1, len(chunks), len(data.get("entities", [])))

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
