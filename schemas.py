from pydantic import BaseModel
from typing import Optional
from datetime import datetime


# ─── Auth Schemas ────────────────────────────────────────────

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str
    department_id: Optional[int] = None


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    role: str
    department_id: Optional[int] = None
    department_name: Optional[str] = None
    avatar: Optional[str] = None

    class Config:
        from_attributes = True


# ─── Department Schemas ──────────────────────────────────────

class DepartmentResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None

    class Config:
        from_attributes = True


# ─── Document Schemas ────────────────────────────────────────

class DocumentResponse(BaseModel):
    id: int
    name: str
    type: str
    category: str = "Tài liệu"
    size: str
    owner: str
    department: Optional[str] = None
    date: str
    tags: list[str]
    thumbnail: Optional[str] = None
    visibility: str = "internal"
    approval_status: str = "pending_approval"
    approval_note: Optional[str] = None
    status: str

    class Config:
        from_attributes = True


# ─── Document Version Schemas ────────────────────────────────

class DocumentVersionResponse(BaseModel):
    id: int
    document_id: int
    version_number: int
    version_label: str
    size: Optional[str] = None
    change_note: Optional[str] = None
    created_by_name: str
    is_current: bool
    created_at: datetime

    class Config:
        from_attributes = True


# ─── Document Chunk Schemas ──────────────────────────────────

class DocumentChunkResponse(BaseModel):
    id: int
    document_id: int
    chunk_index: int
    content: str
    token_count: int = 0

    class Config:
        from_attributes = True


# ─── Dashboard Schemas ───────────────────────────────────────

class DashboardStats(BaseModel):
    totalFiles: int
    storageUsed: str
    recentUploads: int
    teamMembers: int


class UploadDataPoint(BaseModel):
    month: str
    uploads: int


class StorageDataPoint(BaseModel):
    type: str
    size: float
    fill: str


# ─── Chat Schemas ────────────────────────────────────────────

# ─── Knowledge Schemas ──────────────────────────────────────

class KnowledgeCreateRequest(BaseModel):
    title: str
    content_html: str
    content_text: str
    category: str = "Chung"
    tags: list[str] = []
    visibility: str = "internal"
    status: str = "Active"


class KnowledgeUpdateRequest(BaseModel):
    title: Optional[str] = None
    content_html: Optional[str] = None
    content_text: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[list[str]] = None
    visibility: Optional[str] = None
    status: Optional[str] = None


class KnowledgeResponse(BaseModel):
    id: int
    title: str
    content_html: str
    content_text: str
    category: str
    tags: list[str]
    owner: str
    department: Optional[str] = None
    visibility: str = "internal"
    approval_status: str = "pending_approval"
    approval_note: Optional[str] = None
    status: str
    ingest_status: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ─── Chat Schemas ────────────────────────────────────────────

class ChatSendRequest(BaseModel):
    message: str
    document_ids: list[int] = []


class ChatMessageResponse(BaseModel):
    id: int
    role: str
    content: str
    sources: list[str]
    graph_data: dict | None = None

    class Config:
        from_attributes = True
