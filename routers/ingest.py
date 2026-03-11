from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy import or_
from sqlalchemy.orm import Session

from auth import get_current_user
from database import get_db
from models import Document, User
from services import ingest_pipeline

router = APIRouter(prefix="/api/ingest", tags=["Ingestion"])


# ── batch route MUST be registered before {document_id} to avoid route shadowing ──

@router.post("/batch", summary="Trigger ingestion for all pending documents (admin only)")
def batch_ingest(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role != "Admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin only")

    docs = (
        db.query(Document)
        .filter(or_(Document.ingest_status != "completed", Document.ingest_status.is_(None)))
        .all()
    )
    for doc in docs:
        background_tasks.add_task(ingest_pipeline.run, doc.id)

    return {"queued": len(docs), "document_ids": [d.id for d in docs]}


@router.post("/{document_id}", summary="Trigger ingestion for a single document")
def trigger_ingest(
    document_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    background_tasks.add_task(ingest_pipeline.run, document_id)
    return {"document_id": document_id, "status": "queued"}


@router.get("/{document_id}/status", summary="Get ingestion status for a document")
def get_ingest_status(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    return {
        "document_id": document_id,
        "status": doc.ingest_status,
        "error": doc.ingest_error,
    }