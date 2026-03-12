import os
import uuid
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, Form
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session, joinedload
from typing import Optional

from database import get_db
from models import User, Document
from schemas import DocumentResponse
from auth import get_current_user
from config import UPLOAD_DIR, MAX_FILE_SIZE
from fastapi import BackgroundTasks
from services import ingest_pipeline

router = APIRouter(prefix="/api/documents", tags=["Documents"])


# ─── Helpers ─────────────────────────────────────────────────

def format_file_size(size_bytes: int) -> str:
    """Convert bytes to human-readable size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.0f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def get_file_type(filename: str) -> str:
    """Extract file type from filename."""
    ext = filename.rsplit(".", 1)[-1].upper() if "." in filename else "FILE"
    return ext


def check_department_access(user: User, doc: Document):
    """Raise 403 if a non-admin user tries to access a doc outside their department."""
    if user.role == "Admin":
        return  # Admins can access everything
    if doc.department_id is not None and user.department_id != doc.department_id:
        raise HTTPException(
            status_code=403,
            detail="You do not have permission to access this document",
        )


def build_document_response(doc: Document) -> DocumentResponse:
    """Build a DocumentResponse from a Document model instance."""
    return DocumentResponse(
        id=doc.id,
        name=doc.name,
        type=doc.type,
        category=doc.category,
        size=doc.size,
        owner=doc.owner_name,
        department=doc.department_name,
        date=doc.date,
        tags=doc.tags,
        status=doc.status,
    )


# ─── List Documents ─────────────────────────────────────────

@router.get("", response_model=list[DocumentResponse])
def list_documents(
    search: Optional[str] = Query(None),
    type_filter: Optional[str] = Query(None, alias="type"),
    category: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List documents scoped to the user's department (admins see all)."""
    query = db.query(Document).options(
        joinedload(Document.owner),
        joinedload(Document.department),
    )

    # Department scoping: non-admin users only see their department's docs
    if current_user.role != "Admin" and current_user.department_id is not None:
        query = query.filter(Document.department_id == current_user.department_id)

    if search:
        search_lower = f"%{search.lower()}%"
        query = query.filter(Document.name.ilike(search_lower))

    if type_filter and type_filter != "All":
        query = query.filter(Document.type == type_filter)

    if category and category != "All":
        query = query.filter(Document.category == category)

    query = query.order_by(Document.created_at.desc())
    docs = query.all()

    return [build_document_response(doc) for doc in docs]


# ─── Upload Documents ───────────────────────────────────────

@router.post("/upload", response_model=list[DocumentResponse])
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    tags: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    department_id: Optional[int] = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Upload one or more files with category and department assignment."""
    uploaded = []
    tag_list = [t.strip() for t in tags.split(",")] if tags else []

    # Auto-assign department from user if not explicitly provided
    effective_department_id = department_id if department_id is not None else current_user.department_id

    # Default category
    effective_category = category if category else "Tài liệu"

    for file in files:
        # Read file content
        content = await file.read()
        size_bytes = len(content)

        if size_bytes > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit",
            )

        # Save file to disk
        file_ext = file.filename.rsplit(".", 1)[-1] if "." in file.filename else "bin"
        stored_name = f"{uuid.uuid4().hex}.{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, stored_name)

        with open(file_path, "wb") as f:
            f.write(content)

        # Create DB record
        doc = Document(
            name=file.filename,
            type=get_file_type(file.filename),
            category=effective_category,
            size=format_file_size(size_bytes),
            size_bytes=size_bytes,
            owner_id=current_user.id,
            department_id=effective_department_id,
            file_path=stored_name,
            status="Active",
        )
        doc.tags = tag_list
        db.add(doc)
        db.commit()
        db.refresh(doc)

        # Eagerly load relationships for response
        db.refresh(doc, attribute_names=["owner", "department"])

        # Enqueue ingest pipeline as background task
        background_tasks.add_task(ingest_pipeline.run, doc.id)

        uploaded.append(build_document_response(doc))

    return uploaded


# ─── Download Document ──────────────────────────────────────

@router.get("/{doc_id}/download")
def download_document(
    doc_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Download a document file (department-scoped access)."""
    doc = db.query(Document).options(
        joinedload(Document.owner),
        joinedload(Document.department),
    ).filter(Document.id == doc_id).first()

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Department access check
    check_department_access(current_user, doc)

    if not doc.file_path:
        raise HTTPException(status_code=404, detail="File not available for download")

    file_path = os.path.join(UPLOAD_DIR, doc.file_path)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")

    return FileResponse(
        path=file_path,
        filename=doc.name,
        media_type="application/octet-stream",
    )


# ─── Delete Document ────────────────────────────────────────

@router.delete("/{doc_id}")
def delete_document(
    doc_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a document (owner or admin only, department-scoped)."""
    doc = db.query(Document).options(
        joinedload(Document.owner),
        joinedload(Document.department),
    ).filter(Document.id == doc_id).first()

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Department access check
    check_department_access(current_user, doc)

    # Ownership check: only the doc owner or an admin can delete
    if current_user.role != "Admin" and doc.owner_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="Only the document owner or an admin can delete this document",
        )

    # Remove file from disk
    if doc.file_path:
        file_path = os.path.join(UPLOAD_DIR, doc.file_path)
        if os.path.exists(file_path):
            os.remove(file_path)

    db.delete(doc)
    db.commit()

    return {"message": "Document deleted successfully"}
