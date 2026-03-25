import logging
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import text
from typing import Optional

from database import get_db, SessionLocal
from models import User, KnowledgeEntry
from schemas import KnowledgeCreateRequest, KnowledgeUpdateRequest, KnowledgeResponse
from auth import get_current_user
from services.chunker_service import chunk_text
from services.embedding_service import embed
from services.neo4j_service import neo4j_service, category_to_label
from services import kg_extractor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/knowledge", tags=["Knowledge"])


# ─── Ingest background task ─────────────────────────────────

def ingest_knowledge(entry_id: int) -> None:
    """Background task: chunk → embed → store in document_chunks (unified) + Neo4j.

    Writes to the same document_chunks table as documents, with source_type='knowledge'.
    Single ivfflat index = single search path for chatbot.
    KnowledgeEntry exposes .name property (alias for .title) for kg_extractor compatibility.
    """
    db = SessionLocal()
    entry = None
    try:
        entry = db.query(KnowledgeEntry).filter(KnowledgeEntry.id == entry_id).first()
        if entry is None:
            logger.warning("Knowledge ingest: entry_id=%d not found", entry_id)
            return

        entry.ingest_status = "processing"
        entry.ingest_error = None
        db.commit()

        text_content = entry.content_text.strip()
        if not text_content:
            entry.ingest_status = "completed"
            db.commit()
            return

        # ── MERGE Neo4j node ────────────────────────────────────
        label = category_to_label(entry.category)
        neo4j_service.merge_document_node({
            "document_id": entry.id,
            "label": label,
            "ten": entry.title,
            "owner": entry.owner_name,
            "created_at": entry.created_at.isoformat() if entry.created_at else "",
            "department_id": entry.department_id,
        })

        # ── Chunk → Embed → Store in document_chunks (unified) ─
        chunks = chunk_text([text_content])
        if not chunks:
            entry.ingest_status = "completed"
            db.commit()
            return

        contents = [c["content"] for c in chunks]
        vectors = embed(contents)

        # Clear old knowledge chunks before re-inserting
        db.execute(
            text("DELETE FROM document_chunks WHERE source_type = 'knowledge' AND source_id = :sid"),
            {"sid": entry.id},
        )

        for chunk, vector in zip(chunks, vectors):
            db.execute(
                text("""
                    INSERT INTO document_chunks
                        (source_type, source_id, chunk_index, content, embedding, token_count, department_id)
                    VALUES
                        ('knowledge', :source_id, :chunk_index, :content,
                         CAST(:embedding AS vector), :token_count, :department_id)
                """),
                {
                    "source_id": entry.id,
                    "chunk_index": chunk["chunk_index"],
                    "content": chunk["content"],
                    "embedding": str(vector),
                    "token_count": len(chunk["content"].split()),
                    "department_id": entry.department_id,
                },
            )
        db.commit()

        # ── EKG extraction ──────────────────────────────────────
        if neo4j_service.available:
            chunk_texts = [c["content"] for c in chunks]
            kg = kg_extractor.extract_kg(chunk_texts, entry)
            if kg:
                neo4j_service.create_entity_graph(
                    entry.id,
                    kg.get("entities", []),
                    kg.get("relationships", []),
                    source_label=label,
                )

        entry.ingest_status = "completed"
        db.commit()
        logger.info("Knowledge ingest completed for entry_id=%d", entry_id)

    except Exception as exc:
        logger.error("Knowledge ingest failed for entry_id=%d: %s", entry_id, exc, exc_info=True)
        try:
            if entry is not None:
                entry.ingest_status = "failed"
                entry.ingest_error = str(exc)
                db.commit()
        except Exception:
            db.rollback()
    finally:
        db.close()


# ─── Helpers ─────────────────────────────────────────────────

def _to_response(entry: KnowledgeEntry) -> dict:
    return {
        "id": entry.id,
        "title": entry.title,
        "content_html": entry.content_html,
        "content_text": entry.content_text,
        "category": entry.category,
        "tags": entry.tags or [],
        "owner": entry.owner_name,
        "department": entry.department_name,
        "visibility": getattr(entry, "visibility", "internal") or "internal",
        "approval_status": getattr(entry, "approval_status", "pending_approval") or "pending_approval",
        "approval_note": getattr(entry, "approval_note", None),
        "status": entry.status,
        "ingest_status": entry.ingest_status,
        "created_at": entry.created_at,
        "updated_at": entry.updated_at,
    }


# ─── CRUD Endpoints ─────────────────────────────────────────

@router.get("")
def list_knowledge(
    search: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = db.query(KnowledgeEntry).options(
        joinedload(KnowledgeEntry.owner),
        joinedload(KnowledgeEntry.department),
    )

    # Visibility + department scoping:
    # Admin sees all; non-admin sees public entries + internal entries from own department
    if current_user.role != "Admin":
        from sqlalchemy import or_
        query = query.filter(
            or_(
                KnowledgeEntry.visibility == "public",
                KnowledgeEntry.department_id == current_user.department_id,
            )
        )

    if search:
        query = query.filter(
            KnowledgeEntry.title.ilike(f"%{search}%")
            | KnowledgeEntry.content_text.ilike(f"%{search}%")
        )
    if category:
        query = query.filter(KnowledgeEntry.category == category)
    if status:
        query = query.filter(KnowledgeEntry.status == status)

    total = query.count()
    entries = (
        query.order_by(KnowledgeEntry.updated_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return {
        "items": [_to_response(e) for e in entries],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/{entry_id}")
def get_knowledge(
    entry_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    entry = (
        db.query(KnowledgeEntry)
        .options(joinedload(KnowledgeEntry.owner), joinedload(KnowledgeEntry.department))
        .filter(KnowledgeEntry.id == entry_id)
        .first()
    )
    if not entry:
        raise HTTPException(status_code=404, detail="Knowledge entry not found")
    # Access check: non-admin can only read public or own-department entries
    if current_user.role != "Admin":
        vis = getattr(entry, "visibility", "internal") or "internal"
        if vis != "public" and entry.department_id != current_user.department_id:
            raise HTTPException(status_code=403, detail="Permission denied")
    return _to_response(entry)


@router.post("", status_code=201)
def create_knowledge(
    body: KnowledgeCreateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    entry = KnowledgeEntry(
        title=body.title,
        content_html=body.content_html,
        content_text=body.content_text,
        category=body.category,
        tags=body.tags,
        visibility=body.visibility if body.visibility in ("internal", "public") else "internal",
        status=body.status,
        owner_id=current_user.id,
        department_id=current_user.department_id,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)

    # Ingest is deferred until a Trưởng phòng/Admin approves the entry
    return _to_response(entry)


@router.put("/{entry_id}")
def update_knowledge(
    entry_id: int,
    body: KnowledgeUpdateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    entry = (
        db.query(KnowledgeEntry)
        .options(joinedload(KnowledgeEntry.owner), joinedload(KnowledgeEntry.department))
        .filter(KnowledgeEntry.id == entry_id)
        .first()
    )
    if not entry:
        raise HTTPException(status_code=404, detail="Knowledge entry not found")
    if entry.owner_id != current_user.id and current_user.role != "Admin":
        raise HTTPException(status_code=403, detail="Permission denied")

    content_changed = False
    for field, value in body.model_dump(exclude_unset=True).items():
        if field in ("content_html", "content_text") and value is not None:
            content_changed = True
        setattr(entry, field, value)

    db.commit()
    db.refresh(entry)

    # Re-ingest if content changed and status is Active
    if content_changed and entry.status == "Active":
        entry.ingest_status = "pending"
        db.commit()
        background_tasks.add_task(ingest_knowledge, entry.id)

    return _to_response(entry)


@router.delete("/{entry_id}", status_code=204)
def delete_knowledge(
    entry_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    entry = db.query(KnowledgeEntry).filter(KnowledgeEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Knowledge entry not found")
    if entry.owner_id != current_user.id and current_user.role != "Admin":
        raise HTTPException(status_code=403, detail="Permission denied")

    # Clean up chunks (no FK cascade since unified table)
    db.execute(
        text("DELETE FROM document_chunks WHERE source_type = 'knowledge' AND source_id = :sid"),
        {"sid": entry.id},
    )
    db.delete(entry)
    db.commit()
    return None
