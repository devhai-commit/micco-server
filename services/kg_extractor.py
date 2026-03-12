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
    if rel != RelType.HAS_CHUNK
}

_SYSTEM_PROMPT = (
    "You are an expert entity and relationship extractor for a Vietnamese document management system.\n"
    "Given document text, extract named entities and their relationships.\n\n"
    "Valid entity labels: " + ", ".join(sorted(_DOMAIN_LABELS)) + "\n"
    "Valid relationship types: " + ", ".join(sorted(_DOMAIN_RELS)) + "\n\n"
    "Return JSON with this exact schema:\n"
    '{"entities": [{"name": "...", "label": "..."}], '
    '"relationships": [{"source": "...", "source_label": "...", '
    '"relation": "...", "target": "...", "target_label": "..."}]}\n'
    "Only use labels and relation types from the lists above. "
    "Extract only entities clearly mentioned in the text."
)


def extract_kg(chunks: list[str], doc) -> dict:
    """Send first 5 chunks to GPT-4o and return extracted EKG data.

    Returns {"entities": [...], "relationships": [...]} or {} on any error.
    Never raises — extraction failure must not block ingest.
    """
    if not chunks:
        return {}
    try:
        from openai import OpenAI
        client = OpenAI()  # lazy: raises here if OPENAI_API_KEY missing; caught below
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
