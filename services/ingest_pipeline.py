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

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
_UPLOADS_ROOT = Path(os.getenv("UPLOAD_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "uploads"))).resolve()

# Minimum characters per page to consider a PDF as having extractable text
_PDF_TEXT_THRESHOLD = 50


def _extract_pdf_native(file_path: str) -> list[str] | None:
    """Try to extract text directly from a text-based PDF (no OCR).

    Returns a list of non-empty page strings if the PDF has sufficient
    embedded text, or None if the PDF appears to be scanned/image-only.
    Falls back gracefully if pdfplumber/pypdf is not installed.
    """
    pages_text: list[str] = []

    # Try pdfplumber first (best layout-aware extraction)
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                pages_text.append((page.extract_text() or "").strip())
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("pdfplumber failed for %s: %s", file_path, exc)

    # Fallback to pypdf if pdfplumber gave nothing
    if not any(pages_text):
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            pages_text = [(page.extract_text() or "").strip() for page in reader.pages]
        except ImportError:
            pass
        except Exception as exc:
            logger.warning("pypdf failed for %s: %s", file_path, exc)

    if not pages_text:
        return None

    # A PDF is considered text-based when at least half its pages
    # contain more than the threshold number of characters.
    pages_with_text = [p for p in pages_text if len(p) >= _PDF_TEXT_THRESHOLD]
    if len(pages_with_text) >= max(1, len(pages_text) // 2):
        non_empty = [p for p in pages_text if p]
        logger.info(
            "PDF native extraction: %d/%d pages have text (>= %d chars)",
            len(pages_with_text), len(pages_text), _PDF_TEXT_THRESHOLD,
        )
        return non_empty if non_empty else None

    return None


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
            "department_id": doc.department_id,
        })

        # ── Unstructured path: text extraction → chunk → embed ─────────────
        raw_path = doc.file_path or ""
        ext = "." + raw_path.rsplit(".", 1)[-1].lower() if "." in raw_path else ""

        # Determine which chunks to process
        chunks_to_process = None
        file_path = None

        if ext == ".pdf":
            file_path = str(_safe_file_path(raw_path))
            # Try native text extraction first (fast, no GPU needed)
            native_pages = _extract_pdf_native(file_path)
            if native_pages:
                logger.info("PDF has embedded text — using native extraction (no OCR)")
                chunks_to_process = chunk_text(native_pages)
            else:
                # Scanned/image PDF — fall back to OCR
                logger.info("PDF appears scanned — falling back to OCR")
                pages, md_path = extract_text(file_path)
                logger.info("OCR completed. Markdown saved to: %s", md_path)
                if pages:
                    chunks_to_process = chunk_text(pages)
        elif ext in _IMAGE_EXTENSIONS:
            # Images always use OCR
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
        elif ext == ".docx":
            # Word documents: extract text via python-docx
            file_path = str(_safe_file_path(raw_path))
            try:
                import docx
                doc_obj = docx.Document(file_path)
                paragraphs = [p.text for p in doc_obj.paragraphs if p.text.strip()]
                text_content = "\n".join(paragraphs)
                if text_content.strip():
                    chunks_to_process = chunk_text([text_content])
                    logger.info("DOCX extracted: %s (%d paragraphs)", file_path, len(paragraphs))
            except Exception as exc:
                logger.warning("Failed to read docx file %s: %s", file_path, exc)

        # Process chunks if we have them
        if chunks_to_process:
            contents = [c["content"] for c in chunks_to_process]
            vectors = embed(contents)

            for chunk, vector in zip(chunks_to_process, vectors):
                db.execute(
                    text("""
                        INSERT INTO document_chunks
                            (source_type, source_id, chunk_index, content, embedding, token_count, department_id)
                        VALUES
                            ('document', :source_id, :chunk_index, :content,
                             CAST(:embedding AS vector), :token_count, :department_id)
                        ON CONFLICT (source_type, source_id, chunk_index)
                        DO UPDATE SET
                            content       = EXCLUDED.content,
                            embedding     = EXCLUDED.embedding,
                            department_id = EXCLUDED.department_id
                    """),
                    {
                        "source_id": document_id,
                        "chunk_index": chunk["chunk_index"],
                        "content": chunk["content"],
                        "embedding": str(vector),
                        "token_count": len(chunk["content"].split()),
                        "department_id": doc.department_id,
                    },
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
                    entities = kg.get("entities", [])
                    neo4j_service.create_entity_graph(
                        doc.id,
                        entities,
                        kg.get("relationships", []),
                    )
                    # Embed entities for GraphRAG Local Search
                    if entities:
                        from services.entity_embedding_service import upsert_entity_embeddings
                        n = upsert_entity_embeddings(db, entities)
                        logger.info("Upserted %d entity embeddings for doc_id=%d", n, doc.id)
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