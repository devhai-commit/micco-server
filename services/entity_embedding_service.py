"""Embed KG entities and upsert into PostgreSQL entity_embeddings table.

Called after KG extraction during document ingest to keep entity vectors
in sync with the Neo4j knowledge graph. Entity descriptions are built
from the entity name, label, and extracted attributes so that vector
similarity captures both identity and context.
"""

import logging
from sqlalchemy import text
from sqlalchemy.orm import Session

from services.embedding_service import embed

logger = logging.getLogger(__name__)


def _build_description(entity: dict) -> str:
    """Build a searchable description from entity fields."""
    label = entity.get("label", "")
    name = entity.get("name", "")
    attrs = entity.get("attributes", {})
    parts = [f"{label}: {name}"]
    for k, v in attrs.items():
        if v:
            parts.append(f"{k}: {v}")
    return ". ".join(parts)


def upsert_entity_embeddings(db: Session, entities: list[dict]) -> int:
    """Embed and upsert a list of entities into entity_embeddings.

    Args:
        db: Active SQLAlchemy session (caller manages commit/rollback).
        entities: List of entity dicts with keys: name, label, attributes.

    Returns:
        Number of entities successfully upserted.
    """
    if not entities:
        return 0

    descriptions = [_build_description(e) for e in entities]
    try:
        vectors = embed(descriptions)
    except Exception as exc:
        logger.warning("Failed to embed %d entities: %s", len(entities), exc)
        return 0

    count = 0
    for entity, desc, vec in zip(entities, descriptions, vectors):
        name = entity.get("name", "")
        label = entity.get("label", "")
        if not name or not label:
            continue
        try:
            db.execute(
                text("""
                    INSERT INTO entity_embeddings
                        (entity_name, entity_label, description, embedding, updated_at)
                    VALUES
                        (:name, :label, :desc, CAST(:embedding AS vector), NOW())
                    ON CONFLICT (entity_name, entity_label) DO UPDATE SET
                        description = EXCLUDED.description,
                        embedding   = EXCLUDED.embedding,
                        updated_at  = NOW()
                """),
                {
                    "name": name,
                    "label": label,
                    "desc": desc,
                    "embedding": str(vec),
                },
            )
            count += 1
        except Exception as exc:
            logger.warning("Failed to upsert entity %s/%s: %s", label, name, exc)
    return count
