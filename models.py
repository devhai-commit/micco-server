from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, BigInteger, Float, Boolean
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from database import Base


class Department(Base):
    __tablename__ = "departments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    users = relationship("User", back_populates="department")
    documents = relationship("Document", back_populates="department")
    knowledge_entries = relationship("KnowledgeEntry", back_populates="department")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, default="Nhân viên")
    department_id = Column(Integer, ForeignKey("departments.id", ondelete="SET NULL"), nullable=True)
    avatar = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    department = relationship("Department", back_populates="users")
    documents = relationship("Document", foreign_keys="Document.owner_id", back_populates="owner")
    chat_messages = relationship("ChatMessage", back_populates="user")
    knowledge_entries = relationship("KnowledgeEntry", foreign_keys="KnowledgeEntry.owner_id", back_populates="owner")


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    type = Column(String(10), nullable=False)
    category = Column(String(50), nullable=False, default="Tài liệu")
    size = Column(String(50), nullable=False)
    size_bytes = Column(BigInteger, default=0)
    owner_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    department_id = Column(Integer, ForeignKey("departments.id", ondelete="SET NULL"), nullable=True)
    tags = Column(JSONB, nullable=False, default=[])
    thumbnail = Column(String(500), nullable=True)  # Cover image / avatar for document
    visibility = Column(String(20), nullable=False, default="internal")  # 'internal' or 'public'
    approval_status = Column(String(20), nullable=False, default="pending_approval")  # pending_approval | approved | rejected
    approved_by_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    approved_at = Column(DateTime(timezone=True), nullable=True)
    approval_note = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, default="Active")
    file_path = Column(String(500), nullable=True)
    ingest_status = Column(String(20), nullable=True, default="pending")
    ingest_error  = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    owner = relationship("User", foreign_keys=[owner_id], back_populates="documents")
    department = relationship("Department", back_populates="documents")
    versions = relationship("DocumentVersion", back_populates="document", order_by="DocumentVersion.version_number.desc()")
    # chunks: queried via DocumentChunk.source_type='document', source_id=self.id

    @property
    def owner_name(self):
        return self.owner.name if self.owner else "Unknown"

    @property
    def department_name(self):
        return self.department.name if self.department else None

    @property
    def date(self):
        return self.created_at.strftime("%Y-%m-%d") if self.created_at else ""


class DocumentVersion(Base):
    """Version history for documents — each re-upload creates a new version."""
    __tablename__ = "document_versions"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    version_number = Column(Integer, nullable=False)
    version_label = Column(String(50), nullable=False, default="V 1.0")
    file_path = Column(String(500), nullable=True)
    size = Column(String(50), nullable=True)
    size_bytes = Column(BigInteger, default=0)
    change_note = Column(Text, nullable=True)
    created_by = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    is_current = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    document = relationship("Document", back_populates="versions")
    creator = relationship("User", foreign_keys=[created_by])

    @property
    def creator_name(self):
        return self.creator.name if self.creator else "Unknown"


class DocumentChunk(Base):
    """Unified chunk table for both documents and knowledge entries.

    Uses polymorphic source_type + source_id instead of a hard FK,
    so one ivfflat index covers all embeddings and chatbot search
    hits a single table.
    """
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    source_type = Column(String(20), nullable=False, default="document")  # 'document' | 'knowledge'
    source_id = Column(Integer, nullable=False)                            # documents.id or knowledge_entries.id
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    token_count = Column(Integer, default=0)
    department_id = Column(Integer, ForeignKey("departments.id", ondelete="SET NULL"), nullable=True)
    chunk_metadata = Column("metadata", JSONB, default={})
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Note: 'embedding' vector column is managed directly via SQL/pgvector,
    # not mapped here since SQLAlchemy needs pgvector extension for vector type.
    # Note: FK removed — polymorphic source_type/source_id pattern.
    #       Cascade deletes handled by application-level cleanup.


class KnowledgeEntry(Base):
    __tablename__ = "knowledge_entries"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500), nullable=False)
    content_html = Column(Text, nullable=False)          # WYSIWYG HTML content
    content_text = Column(Text, nullable=False)           # Plain text for search/embedding
    category = Column(String(100), nullable=False, default="Chung")
    tags = Column(JSONB, nullable=False, default=[])
    owner_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    department_id = Column(Integer, ForeignKey("departments.id", ondelete="SET NULL"), nullable=True)
    visibility = Column(String(20), nullable=False, default="internal")  # 'internal' | 'public'
    approval_status = Column(String(20), nullable=False, default="pending_approval")  # pending_approval | approved | rejected
    approved_by_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    approved_at = Column(DateTime(timezone=True), nullable=True)
    approval_note = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, default="Active")  # Active, Draft, Archived
    ingest_status = Column(String(20), nullable=True, default="pending")  # pending, processing, completed, failed
    ingest_error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    owner = relationship("User", foreign_keys=[owner_id], back_populates="knowledge_entries")
    department = relationship("Department", back_populates="knowledge_entries")
    # chunks: queried via DocumentChunk.source_type='knowledge', source_id=self.id

    @property
    def owner_name(self):
        return self.owner.name if self.owner else "Unknown"

    @property
    def department_name(self):
        return self.department.name if self.department else None

    # Compatibility properties for kg_extractor (expects doc.name, doc.category)
    @property
    def name(self):
        return self.title


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(10), nullable=False)
    content = Column(Text, nullable=False)
    sources = Column(JSONB, nullable=False, default=[])
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="chat_messages")
