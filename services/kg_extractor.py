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


def extract_kg(chunks: list[str], doc) -> dict:
    """Send first 5 chunks to GPT-4o and return extracted EKG data.

    Returns {"entities": [...], "relationships": [...]} or {} on any error.
    Never raises — extraction failure must not block ingest.
    """
    if not chunks:
        logger.warning("extract_kg called with empty chunks for doc_id=%s", getattr(doc, "id", "?"))
        return {}
    try:
        from openai import OpenAI
        client = OpenAI()  # lazy: raises here if OPENAI_API_KEY missing; caught below
        logger.info("OpenAI client initialized successfully for doc_id=%s", getattr(doc, "id", "?"))
        text = "\n\n".join(chunks[:5])
        user_prompt = (
            f"Document name: {doc.name}\n"
            f"Document category: {doc.category}\n\n"
            f"Text:\n{text}"
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        data = json.loads(response.choices[0].message.content)

        entities = [
            e for e in data.get("entities", [])
            if isinstance(e, dict)
            and e.get("name")
            and e.get("label") in _DOMAIN_LABELS
        ]
        relationships = [
            r for r in data.get("relationships", [])
            if isinstance(r, dict)
            and r.get("source")
            and r.get("source_label") in _DOMAIN_LABELS
            and r.get("relation") in _DOMAIN_RELS
            and r.get("target")
            and r.get("target_label") in _DOMAIN_LABELS
        ]
        if not entities and not relationships:
            return {}
        return {"entities": entities, "relationships": relationships}
    except Exception as exc:
        logger.warning(
            "KG extraction failed for doc %s: %s",
            getattr(doc, "id", "?"), exc,
        )
        return {}
