import json
import logging
import numpy as np
from kg.ontology import NodeLabel, RelType

logger = logging.getLogger(__name__)

_DOMAIN_LABELS: set[str] = {
    label.value for label in NodeLabel
    if label is not NodeLabel.DOCUMENT
}

_DOMAIN_RELS: set[str] = {
    rel.value for rel in RelType
}

# ── Prompts ──────────────────────────────────────────────────────────────────

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

_GLEANING_PROMPT = (
    "You previously extracted entities from a document section. "
    "Re-read the text carefully and identify any entities you MISSED.\n\n"
    "ALREADY EXTRACTED:\n{extracted}\n\n"
    "TEXT:\n{text}\n\n"
    "Return ONLY new entities not already in the list above.\n"
    "Use the same JSON format: {{\"entities\": [...]}}\n"
    "If nothing was missed, return {{\"entities\": []}}\n"
    "Rules: same valid labels and attribute rules as before."
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

# ── Constants ─────────────────────────────────────────────────────────────────

_ENTITY_BATCH_SIZE = 20   # chunks per Phase 1 batch
_REL_BATCH_SIZE    = 8    # chunks per Phase 2 batch (tight context)
_REL_CONTEXT_PAD  = 2    # extra chunks before/after each Phase 2 batch for context
_NORM_THRESHOLD   = 0.92  # cosine similarity to merge entity names


# ── Entity normalization ──────────────────────────────────────────────────────

def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 1e-9 else 0.0


def _normalize_entities(entities: list[dict]) -> list[dict]:
    """Merge entities whose names are semantically near-identical (same label).

    Uses text-embedding-3-small cosine similarity. Entities with sim >= threshold
    within the same label are merged: shorter name folds into longer (more specific).
    Attributes are union-merged onto the surviving entity.
    """
    if len(entities) <= 1:
        return entities

    try:
        from services.embedding_service import embed

        # Group indices by label
        by_label: dict[str, list[int]] = {}
        for i, e in enumerate(entities):
            by_label.setdefault(e["label"], []).append(i)

        # merged_into[i] = canonical index for entity i
        merged_into: dict[int, int] = {}

        def _canonical(i: int) -> int:
            while i in merged_into:
                i = merged_into[i]
            return i

        for label, idxs in by_label.items():
            if len(idxs) < 2:
                continue
            names = [entities[i]["name"] for i in idxs]
            vectors = embed(names)

            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    ca = _canonical(idxs[a])
                    cb = _canonical(idxs[b])
                    if ca == cb:
                        continue
                    # Use the position in the original idxs list for vectors
                    sim = _cosine(vectors[a], vectors[b])
                    if sim >= _NORM_THRESHOLD:
                        # Keep the longer (more specific) name as canonical
                        if len(entities[ca]["name"]) >= len(entities[cb]["name"]):
                            winner, loser = ca, cb
                        else:
                            winner, loser = cb, ca
                        merged_into[loser] = winner
                        # Merge attributes: winner keeps its values; loser fills gaps
                        winner_attrs = entities[winner].setdefault("attributes", {})
                        for k, v in entities[loser].get("attributes", {}).items():
                            if v and k not in winner_attrs:
                                winner_attrs[k] = v
                        logger.debug(
                            "Merged entity '%s' → '%s' (sim=%.3f)",
                            entities[loser]["name"], entities[winner]["name"], sim,
                        )

        # Keep only canonical entities (preserve original order)
        seen_canonical: set[int] = set()
        result = []
        for i in range(len(entities)):
            c = _canonical(i)
            if c not in seen_canonical:
                seen_canonical.add(c)
                result.append(entities[c])
        logger.info("Entity normalization: %d → %d entities", len(entities), len(result))
        return result

    except Exception as exc:
        logger.warning("Entity normalization failed, returning original list: %s", exc)
        return entities


# ── Helpers ───────────────────────────────────────────────────────────────────

def _merge_entity_into_list(e: dict, all_entities: list[dict]) -> bool:
    """Merge entity into all_entities. Returns True if it was new, False if merged."""
    existing = next(
        (x for x in all_entities if x["name"] == e["name"] and x["label"] == e["label"]),
        None,
    )
    if existing is None:
        e.setdefault("attributes", {})
        all_entities.append(e)
        return True
    # Merge new attributes into existing
    for k, v in e.get("attributes", {}).items():
        if v and k not in existing.get("attributes", {}):
            existing.setdefault("attributes", {})[k] = v
    return False


def _call_entity_extraction(client, system_prompt: str, user_prompt: str) -> list[dict]:
    """Call OpenAI for entity extraction and return validated entity dicts."""
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )
    data = json.loads(response.choices[0].message.content)
    return [
        e for e in data.get("entities", [])
        if isinstance(e, dict) and e.get("name") and e.get("label") in _DOMAIN_LABELS
    ]


# ── Main extractor ────────────────────────────────────────────────────────────

def extract_kg(chunks: list[str], doc) -> dict:
    """Three-phase KG extraction with gleaning and entity normalization.

    Phase 1 — Entity extraction (batched, 20 chunks/batch)
      + Gleaning: one follow-up pass per batch to catch missed entities

    Phase 1b — Entity normalization via embedding similarity dedup

    Phase 2 — Relationship extraction (8 chunks/batch, ±2 chunk context window)

    Returns {"entities": [...], "relationships": [...]} or {} on any error.
    Never raises — extraction failure must not block ingest.
    """
    if not chunks:
        logger.warning("extract_kg called with empty chunks for doc_id=%s", getattr(doc, "id", "?"))
        return {}
    try:
        from openai import OpenAI
        client = OpenAI()
        logger.info(
            "KG extraction start: doc_id=%s, %d chunks",
            getattr(doc, "id", "?"), len(chunks),
        )

        # ── Phase 1: Entity extraction + gleaning ─────────────────────────────
        all_entities: list[dict] = []
        total_batches = (len(chunks) + _ENTITY_BATCH_SIZE - 1) // _ENTITY_BATCH_SIZE

        for batch_idx in range(0, len(chunks), _ENTITY_BATCH_SIZE):
            batch_num   = batch_idx // _ENTITY_BATCH_SIZE + 1
            batch_chunks = chunks[batch_idx: batch_idx + _ENTITY_BATCH_SIZE]
            batch_text   = "\n\n---\n\n".join(batch_chunks)

            user_prompt = (
                f"Document name: {doc.name}\n"
                f"Document category: {doc.category}\n\n"
                f"Extract ONLY entities (no relationships) from the following text "
                f"(batch {batch_num}/{total_batches}, {len(batch_chunks)} sections).\n\n"
                f"Text:\n{batch_text}"
            )
            try:
                extracted = _call_entity_extraction(client, _ENTITY_ONLY_PROMPT, user_prompt)
                new_count = sum(1 for e in extracted if _merge_entity_into_list(e, all_entities))
                logger.info(
                    "Phase 1 batch %d/%d: +%d new entities (total: %d)",
                    batch_num, total_batches, new_count, len(all_entities),
                )
            except Exception as exc:
                logger.warning("Phase 1 batch %d failed: %s", batch_num, exc)
                continue

            # ── Gleaning: one additional pass to recover missed entities ──────
            try:
                already = "\n".join(
                    f"- {e['name']} ({e['label']})" for e in all_entities
                )
                gleaning_user = _GLEANING_PROMPT.format(
                    extracted=already, text=batch_text
                )
                gleaned = _call_entity_extraction(client, _ENTITY_ONLY_PROMPT, gleaning_user)
                glean_count = sum(1 for e in gleaned if _merge_entity_into_list(e, all_entities))
                if glean_count:
                    logger.info(
                        "Gleaning batch %d: +%d additional entities (total: %d)",
                        batch_num, glean_count, len(all_entities),
                    )
            except Exception as exc:
                logger.warning("Gleaning batch %d failed (non-fatal): %s", batch_num, exc)

        if not all_entities:
            logger.warning("No entities extracted for doc_id=%s", getattr(doc, "id", "?"))
            return {}

        # ── Phase 1b: Entity normalization — merge near-duplicate names ───────
        all_entities = _normalize_entities(all_entities)

        # ── Phase 2: Relationship extraction (sliding window) ─────────────────
        # Each batch of _REL_BATCH_SIZE chunks gets _REL_CONTEXT_PAD chunks of
        # padding on each side so cross-boundary relationships are captured.
        relationships: list[dict] = []
        entity_info = "\n".join(f"- {e['name']} ({e['label']})" for e in all_entities)
        rel_batches  = (len(chunks) + _REL_BATCH_SIZE - 1) // _REL_BATCH_SIZE

        for batch_idx in range(0, len(chunks), _REL_BATCH_SIZE):
            batch_num  = batch_idx // _REL_BATCH_SIZE + 1

            # Core window for this batch
            core_start = batch_idx
            core_end   = min(len(chunks), batch_idx + _REL_BATCH_SIZE)

            # Expanded window with padding for context
            ctx_start  = max(0, core_start - _REL_CONTEXT_PAD)
            ctx_end    = min(len(chunks), core_end + _REL_CONTEXT_PAD)
            window     = chunks[ctx_start:ctx_end]
            batch_text = "\n\n---\n\n".join(window)

            rel_prompt = (
                f"Document name: {doc.name}\n"
                f"Document category: {doc.category}\n\n"
                f"KNOWN ENTITIES:\n{entity_info}\n\n"
                f"TEXT (focus on sections {core_start + 1}–{core_end}, "
                f"context sections {ctx_start + 1}–{ctx_end}):\n{batch_text}\n\n"
                f"Find relationships between known entities in this section only.\n"
                f"Only create relationships explicitly mentioned in this section."
            )

            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": _RELATIONSHIP_ONLY_PROMPT},
                        {"role": "user",   "content": rel_prompt},
                    ],
                )
                data = json.loads(response.choices[0].message.content)

                new_rels = 0
                for r in data.get("relationships", []):
                    if (
                        isinstance(r, dict)
                        and r.get("source")
                        and r.get("source_label") in _DOMAIN_LABELS
                        and r.get("relation")      in _DOMAIN_RELS
                        and r.get("target")
                        and r.get("target_label")  in _DOMAIN_LABELS
                        and not any(
                            x["source"]   == r["source"]
                            and x["relation"] == r["relation"]
                            and x["target"]   == r["target"]
                            for x in relationships
                        )
                    ):
                        relationships.append(r)
                        new_rels += 1

                logger.info(
                    "Phase 2 batch %d/%d (chunks %d–%d, ctx %d–%d): +%d relationships",
                    batch_num, rel_batches,
                    core_start + 1, core_end,
                    ctx_start + 1, ctx_end,
                    new_rels,
                )
            except Exception as exc:
                logger.warning("Phase 2 batch %d failed: %s", batch_num, exc)
                continue

        logger.info(
            "KG extraction done: doc_id=%s → %d entities, %d relationships",
            getattr(doc, "id", "?"), len(all_entities), len(relationships),
        )
        return {"entities": all_entities, "relationships": relationships}

    except Exception as exc:
        logger.warning(
            "KG extraction failed for doc %s: %s",
            getattr(doc, "id", "?"), exc,
        )
        return {}
