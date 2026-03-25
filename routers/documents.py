import os
import uuid
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, Form
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session, joinedload
from typing import Optional

from database import get_db
from models import User, Document, DocumentVersion
from schemas import DocumentResponse, DocumentVersionResponse
from auth import get_current_user
from config import UPLOAD_DIR, MAX_FILE_SIZE

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
    """Raise 403 if a non-admin user tries to access a non-public doc outside their department."""
    if user.role == "Admin":
        return  # Admins can access everything
    if getattr(doc, "visibility", "internal") == "public":
        return  # Public docs are accessible to all
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
        thumbnail=doc.thumbnail,
        visibility=doc.visibility or "internal",
        approval_status=doc.approval_status or "pending_approval",
        approval_note=doc.approval_note,
        status=doc.status,
    )


# ─── List Documents ─────────────────────────────────────────

@router.get("", response_model=list[DocumentResponse])
def list_documents(
    search: Optional[str] = Query(None),
    type_filter: Optional[str] = Query(None, alias="type"),
    category: Optional[str] = Query(None),
    department_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List documents scoped to the user's department (admins see all)."""
    query = db.query(Document).options(
        joinedload(Document.owner),
        joinedload(Document.department),
    )

    # Visibility + department scoping:
    # - Admin sees everything (optionally filtered by department_id param)
    # - Non-admin sees: public docs + internal docs from own department
    if current_user.role != "Admin":
        from sqlalchemy import or_
        query = query.filter(
            or_(
                Document.visibility == "public",
                Document.department_id == current_user.department_id,
            )
        )
    if department_id is not None:
        query = query.filter(Document.department_id == department_id)

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
    files: list[UploadFile] = File(...),
    tags: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    visibility: Optional[str] = Form("internal"),
    department_id: Optional[int] = Form(None),
    thumbnail: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Upload one or more files with category and department assignment."""
    uploaded = []
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    # Auto-assign department from user if not explicitly provided
    effective_department_id = department_id if department_id is not None else current_user.department_id

    # Default category
    effective_category = category if category else "Tài liệu"

    # Handle thumbnail upload
    thumbnail_path = None
    if thumbnail:
        thumb_content = await thumbnail.read()
        thumb_ext = thumbnail.filename.rsplit(".", 1)[-1] if "." in thumbnail.filename else "jpg"
        thumb_name = f"thumb_{uuid.uuid4().hex}.{thumb_ext}"
        thumb_path = os.path.join(UPLOAD_DIR, thumb_name)
        with open(thumb_path, "wb") as f:
            f.write(thumb_content)
        thumbnail_path = thumb_name

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
        # Validate visibility value
        effective_visibility = visibility if visibility in ("internal", "public") else "internal"

        doc = Document(
            name=file.filename,
            type=get_file_type(file.filename),
            category=effective_category,
            size=format_file_size(size_bytes),
            size_bytes=size_bytes,
            owner_id=current_user.id,
            department_id=effective_department_id,
            file_path=stored_name,
            thumbnail=thumbnail_path,
            visibility=effective_visibility,
            status="Active",
        )
        doc.tags = tag_list
        db.add(doc)
        db.commit()
        db.refresh(doc)

        # Create initial version V1.0
        initial_version = DocumentVersion(
            document_id=doc.id,
            version_number=1,
            version_label="V 1.0",
            file_path=stored_name,
            size=format_file_size(size_bytes),
            size_bytes=size_bytes,
            change_note="Phiên bản gốc",
            created_by=current_user.id,
            is_current=True,
        )
        db.add(initial_version)
        db.commit()

        # Eagerly load relationships for response
        db.refresh(doc, attribute_names=["owner", "department"])

        # Ingest is deferred until a Trưởng phòng/Admin approves the document

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


# ─── Get Thumbnail ──────────────────────────────────────────────

@router.get("/{doc_id}/thumbnail")
def get_thumbnail(
    doc_id: int,
    db: Session = Depends(get_db),
):
    """Serve document thumbnail image (public access)."""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc or not doc.thumbnail:
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    file_path = os.path.join(UPLOAD_DIR, doc.thumbnail)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Thumbnail file not found")

    # Determine media type from extension
    ext = doc.thumbnail.rsplit(".", 1)[-1].lower()
    media_types = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "gif": "image/gif", "webp": "image/webp"}
    media_type = media_types.get(ext, "image/jpeg")

    return FileResponse(
        path=file_path,
        media_type=media_type,
    )


# ─── Get Single Document ────────────────────────────────────────

@router.get("/{doc_id}", response_model=DocumentResponse)
def get_document(
    doc_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a single document by ID."""
    doc = db.query(Document).options(
        joinedload(Document.owner),
        joinedload(Document.department),
    ).filter(Document.id == doc_id).first()

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Department access check
    check_department_access(current_user, doc)

    return build_document_response(doc)


# ─── Get Document Versions ──────────────────────────────────

@router.get("/{doc_id}/versions", response_model=list[DocumentVersionResponse])
def get_document_versions(
    doc_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get all versions of a document."""
    doc = db.query(Document).options(
        joinedload(Document.owner),
        joinedload(Document.department),
    ).filter(Document.id == doc_id).first()

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    check_department_access(current_user, doc)

    versions = (
        db.query(DocumentVersion)
        .filter(DocumentVersion.document_id == doc_id)
        .order_by(DocumentVersion.version_number.desc())
        .all()
    )

    return [
        DocumentVersionResponse(
            id=v.id,
            document_id=v.document_id,
            version_number=v.version_number,
            version_label=v.version_label,
            size=v.size,
            change_note=v.change_note,
            created_by_name=v.creator_name,
            is_current=bool(v.is_current),
            created_at=v.created_at,
        )
        for v in versions
    ]


# ─── Upload New Version ─────────────────────────────────────

@router.post("/{doc_id}/versions", response_model=DocumentVersionResponse)
async def upload_new_version(
    doc_id: int,
    file: UploadFile = File(...),
    change_note: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Upload a new version of a document."""
    doc = db.query(Document).options(
        joinedload(Document.owner),
        joinedload(Document.department),
    ).filter(Document.id == doc_id).first()

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    check_department_access(current_user, doc)

    # Read file content
    content = await file.read()
    size_bytes = len(content)

    if size_bytes > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit",
        )

    # Save file to disk
    file_ext = file.filename.rsplit(".", 1)[-1] if "." in file.filename else "bin"
    stored_name = f"{uuid.uuid4().hex}.{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, stored_name)

    with open(file_path, "wb") as f:
        f.write(content)

    # Get next version number
    max_version = (
        db.query(DocumentVersion.version_number)
        .filter(DocumentVersion.document_id == doc_id)
        .order_by(DocumentVersion.version_number.desc())
        .first()
    )
    next_version = (max_version[0] + 1) if max_version else 1

    # Generate version label
    version_label = f"V {next_version}.0"

    # Mark all previous versions as not current
    db.query(DocumentVersion).filter(
        DocumentVersion.document_id == doc_id,
        DocumentVersion.is_current == True,
    ).update({"is_current": False})

    # Create new version
    new_version = DocumentVersion(
        document_id=doc_id,
        version_number=next_version,
        version_label=version_label,
        file_path=stored_name,
        size=format_file_size(size_bytes),
        size_bytes=size_bytes,
        change_note=change_note or f"Phiên bản {version_label}",
        created_by=current_user.id,
        is_current=True,
    )
    db.add(new_version)

    # Update the main document to point to the latest file
    doc.file_path = stored_name
    doc.size = format_file_size(size_bytes)
    doc.size_bytes = size_bytes
    doc.name = file.filename  # Update name if the new file has a different name

    db.commit()
    db.refresh(new_version)

    return DocumentVersionResponse(
        id=new_version.id,
        document_id=new_version.document_id,
        version_number=new_version.version_number,
        version_label=new_version.version_label,
        size=new_version.size,
        change_note=new_version.change_note,
        created_by_name=new_version.creator_name,
        is_current=True,
        created_at=new_version.created_at,
    )


# ─── Download Specific Version ──────────────────────────────

@router.get("/{doc_id}/versions/{version_id}/download")
def download_version(
    doc_id: int,
    version_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Download a specific version of a document."""
    doc = db.query(Document).options(
        joinedload(Document.owner),
        joinedload(Document.department),
    ).filter(Document.id == doc_id).first()

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    check_department_access(current_user, doc)

    version = db.query(DocumentVersion).filter(
        DocumentVersion.id == version_id,
        DocumentVersion.document_id == doc_id,
    ).first()

    if not version:
        raise HTTPException(status_code=404, detail="Version not found")

    if not version.file_path:
        raise HTTPException(status_code=404, detail="File not available for this version")

    file_path = os.path.join(UPLOAD_DIR, version.file_path)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")

    return FileResponse(
        path=file_path,
        filename=f"{doc.name} ({version.version_label})",
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
