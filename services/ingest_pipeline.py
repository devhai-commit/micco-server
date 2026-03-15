import logging
import os
from pathlib import Path
from sqlalchemy import text

from database import SessionLocal
from models import Document
from services.neo4j_service import neo4j_service, category_to_label
from services.ocr_pipeline import extract_text
from services.chunker_service import chunk_text
from services.embedding_service import embed
from services import kg_extractor

logger = logging.getLogger(__name__)

_OCR_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
_UPLOADS_ROOT = Path(os.getenv("UPLOAD_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "uploads"))).resolve()


def _safe_file_path(file_path: str) -> Path:
    """Resolve path and verify it stays within the uploads root (prevents path traversal)."""
    resolved = (_UPLOADS_ROOT / file_path).resolve()
    if not str(resolved).startswith(str(_UPLOADS_ROOT)):
        raise ValueError(f"Path traversal detected: {file_path!r}")
    return resolved


def run(document_id: int) -> None:
    """Main ingest pipeline orchestrator.

    Opens its own DB session. Designed to be dispatched via
    FastAPI BackgroundTasks (runs in a thread pool, not the event loop).

    Stages:
      1. Load document record from PostgreSQL.
      2. MERGE typed Neo4j node (Document node).
      3. If PDF/PNG/JPG: OCR → markdown → chunk → embed → INSERT document_chunks.
         If .md/.txt: read → chunk → embed → INSERT document_chunks.
      4. Extract entities/relationships via LLM → insert into Neo4j graph.

    Status transitions: pending → processing → completed | failed
    """
    db = SessionLocal()
    doc = None  # initialize before try so except block can safely check it
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc is None:
            logger.warning("Ingest called for nonexistent document_id=%d", document_id)
            return

        # Mark as processing
        doc.ingest_status = "processing"
        doc.ingest_error = None
        db.commit()

        # ── Structured path: MERGE Neo4j node ──────────────────────────────
        label = category_to_label(doc.category)
        neo4j_service.merge_document_node({
            "document_id": doc.id,
            "label": label,
            "ten": doc.name,
            "owner": doc.owner_name,
            "created_at": doc.created_at.isoformat() if doc.created_at else "",
        })

        # ── Unstructured path: text extraction → chunk → embed ─────────────
        raw_path = doc.file_path or ""
        ext = "." + raw_path.rsplit(".", 1)[-1].lower() if "." in raw_path else ""

        # Determine which chunks to process
        chunks_to_process = None
        file_path = None

        if ext in _OCR_EXTENSIONS:
            # PDF/Image: use OCR
            file_path = str(_safe_file_path(raw_path))
            pages, md_path = extract_text(file_path)
            logger.info("OCR completed. Markdown saved to: %s", md_path)
            if pages:
                chunks_to_process = chunk_text(pages)
        elif ext in {".md", ".txt"}:
            # Text files: read directly
            file_path = str(_safe_file_path(raw_path))
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text_content = f.read()
                if text_content.strip():
                    chunks_to_process = chunk_text([text_content])
                    logger.info("Text file read directly: %s", file_path)
            except Exception as exc:
                logger.warning("Failed to read text file %s: %s", file_path, exc)

        # Process chunks if we have them
        if chunks_to_process:
            contents = [c["content"] for c in chunks_to_process]
            vectors = embed(contents)

            for chunk, vector in zip(chunks_to_process, vectors):
                db.execute(
                    text("""
                        INSERT INTO document_chunks
                            (document_id, chunk_index, content, embedding, token_count)
                        VALUES
                            (:document_id, :chunk_index, :content,
                             CAST(:embedding AS vector), :token_count)
                        ON CONFLICT (document_id, chunk_index)
                        DO UPDATE SET
                            content   = EXCLUDED.content,
                            embedding = EXCLUDED.embedding
                    """),
                    {
                        "document_id": document_id,
                        "chunk_index": chunk["chunk_index"],
                        "content": chunk["content"],
                        "embedding": str(vector),
                        "token_count": len(chunk["content"].split()),
                    },
                )
                # Sync DocumentChunk to Neo4j with embedding
                neo4j_service.merge_document_chunk(
                    document_id=document_id,
                    chunk_index=chunk["chunk_index"],
                    content=chunk["content"],
                    embedding=vector,
                )
            db.commit()

            # ── EKG extraction ─────────────────────────────────────
            logger.info("EKG extraction: neo4j_service.available=%s", neo4j_service.available)
            if neo4j_service.available:
                chunk_texts = [c["content"] for c in chunks_to_process]
                logger.info("EKG extraction: extracting from %d chunks", len(chunk_texts))
                kg = kg_extractor.extract_kg(chunk_texts, doc)
                logger.info("EKG extraction: result=%s", kg)
                if kg:
                    neo4j_service.create_entity_graph(
                        doc.id,
                        kg.get("entities", []),
                        kg.get("relationships", []),
                    )
                else:
                    logger.warning("EKG extraction returned empty for doc_id=%d", doc.id)
            else:
                logger.warning("Neo4j unavailable, skipping EKG extraction for doc_id=%d", document_id)

        doc.ingest_status = "completed"
        db.commit()
        logger.info("Ingest completed for document_id=%d", document_id)

    except Exception as exc:
        logger.error("Ingest failed for document_id=%d: %s", document_id, exc, exc_info=True)
        try:
            if doc is not None:
                doc.ingest_status = "failed"
                doc.ingest_error = str(exc)
                db.commit()
        except Exception:
            db.rollback()
    finally:
        db.close()