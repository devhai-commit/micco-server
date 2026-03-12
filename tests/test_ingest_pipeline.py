import pytest
import sys
import importlib
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone


@pytest.fixture(autouse=True)
def _reload_ingest_pipeline():
    """Reload ingest_pipeline module before each test to clear stubs."""
    # Remove module from cache if it's a stub (types.ModuleType without __file__)
    if "services.ingest_pipeline" in sys.modules:
        mod = sys.modules["services.ingest_pipeline"]
        if not hasattr(mod, "__file__"):
            del sys.modules["services.ingest_pipeline"]
    if "services.kg_extractor" in sys.modules:
        mod = sys.modules["services.kg_extractor"]
        if not hasattr(mod, "__file__"):
            del sys.modules["services.kg_extractor"]
    if "services" in sys.modules:
        # Also reload services to clear cached reference
        try:
            importlib.reload(sys.modules["services"])
        except (ImportError, AttributeError):
            pass


def _make_mock_db(doc):
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = doc
    return db


def _make_doc(doc_id=1, category="Tài liệu", file_path="/uploads/test.pdf", name="test.pdf"):
    doc = MagicMock()
    doc.id = doc_id
    doc.name = name
    doc.category = category
    doc.file_path = file_path
    doc.owner_name = "admin"
    doc.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    doc.ingest_status = "pending"
    doc.ingest_error = None
    return doc


def test_run_sets_processing_then_completed():
    doc = _make_doc()
    with patch("services.ingest_pipeline.SessionLocal") as mock_session_cls, \
         patch("services.ingest_pipeline.neo4j_service") as mock_neo4j, \
         patch("services.ingest_pipeline._safe_file_path", return_value="/uploads/test.pdf"), \
         patch("services.ingest_pipeline.extract_text", return_value=["Page 1"]), \
         patch("services.ingest_pipeline.chunk_text", return_value=[{"chunk_index": 0, "content": "text", "char_start": 0}]), \
         patch("services.ingest_pipeline.embed", return_value=[[0.1] * 1024]), \
         patch("services.ingest_pipeline.text", return_value=MagicMock()), \
         patch("services.ingest_pipeline.kg_extractor") as mock_kg:
        mock_neo4j.available = True
        mock_kg.extract_kg.return_value = {}
        db = _make_mock_db(doc)
        mock_session_cls.return_value = db
        from services import ingest_pipeline
        ingest_pipeline.run(1)
        assert doc.ingest_status == "completed"


def test_run_document_not_found_does_not_crash():
    with patch("services.ingest_pipeline.SessionLocal") as mock_session_cls:
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None
        mock_session_cls.return_value = db
        from services import ingest_pipeline
        ingest_pipeline.run(999)


def test_run_skips_ocr_for_non_pdf_category():
    doc = _make_doc(category="HopDong", file_path="/uploads/contract.docx")
    with patch("services.ingest_pipeline.SessionLocal") as mock_session_cls, \
         patch("services.ingest_pipeline.neo4j_service") as mock_neo4j, \
         patch("services.ingest_pipeline.extract_text") as mock_ocr, \
         patch("services.ingest_pipeline.kg_extractor") as mock_kg:
        db = _make_mock_db(doc)
        mock_session_cls.return_value = db
        from services import ingest_pipeline
        ingest_pipeline.run(1)
        mock_ocr.assert_not_called()


def test_run_ocr_path_for_pdf():
    doc = _make_doc(category="Tài liệu", file_path="/uploads/material.pdf")
    with patch("services.ingest_pipeline.SessionLocal") as mock_session_cls, \
         patch("services.ingest_pipeline.neo4j_service") as mock_neo4j, \
         patch("services.ingest_pipeline._safe_file_path", return_value="/uploads/material.pdf"), \
         patch("services.ingest_pipeline.extract_text", return_value=["page text"]) as mock_ocr, \
         patch("services.ingest_pipeline.chunk_text", return_value=[{"chunk_index": 0, "content": "c", "char_start": 0}]), \
         patch("services.ingest_pipeline.embed", return_value=[[0.0] * 1024]), \
         patch("services.ingest_pipeline.text", return_value=MagicMock()), \
         patch("services.ingest_pipeline.kg_extractor") as mock_kg:
        mock_neo4j.available = False
        db = _make_mock_db(doc)
        mock_session_cls.return_value = db
        from services import ingest_pipeline
        ingest_pipeline.run(1)
        mock_ocr.assert_called_once()


def test_run_sets_failed_status_on_exception():
    doc = _make_doc()
    with patch("services.ingest_pipeline.SessionLocal") as mock_session_cls, \
         patch("services.ingest_pipeline.neo4j_service") as mock_neo4j, \
         patch("services.ingest_pipeline.extract_text", side_effect=RuntimeError("OCR crash")), \
         patch("services.ingest_pipeline.kg_extractor") as mock_kg:
        db = _make_mock_db(doc)
        mock_session_cls.return_value = db
        from services import ingest_pipeline
        ingest_pipeline.run(1)
        assert doc.ingest_status == "failed"


def test_run_calls_kg_extractor_when_neo4j_available():
    """After chunk insert, extract_kg and create_entity_graph must be called."""
    doc = _make_doc()
    kg_result = {
        "entities": [{"name": "ABC", "label": "HopDong"}],
        "relationships": [],
    }
    with patch("services.ingest_pipeline.SessionLocal") as mock_session_cls, \
         patch("services.ingest_pipeline.neo4j_service") as mock_neo4j, \
         patch("services.ingest_pipeline._safe_file_path", return_value="/uploads/test.pdf"), \
         patch("services.ingest_pipeline.extract_text", return_value=["Page 1"]), \
         patch("services.ingest_pipeline.chunk_text",
               return_value=[{"chunk_index": 0, "content": "text", "char_start": 0}]), \
         patch("services.ingest_pipeline.embed", return_value=[[0.1] * 1024]), \
         patch("services.ingest_pipeline.text", return_value=MagicMock()), \
         patch("services.ingest_pipeline.kg_extractor") as mock_kg:
        mock_neo4j.available = True
        mock_kg.extract_kg.return_value = kg_result
        db = _make_mock_db(doc)
        mock_session_cls.return_value = db
        from services import ingest_pipeline
        ingest_pipeline.run(1)
        mock_kg.extract_kg.assert_called_once()
        mock_neo4j.create_entity_graph.assert_called_once_with(
            doc.id,
            kg_result["entities"],
            kg_result["relationships"],
        )


def test_run_skips_kg_extractor_when_neo4j_unavailable():
    """If Neo4j is unavailable, kg_extractor must not be called."""
    doc = _make_doc()
    with patch("services.ingest_pipeline.SessionLocal") as mock_session_cls, \
         patch("services.ingest_pipeline.neo4j_service") as mock_neo4j, \
         patch("services.ingest_pipeline._safe_file_path", return_value="/uploads/test.pdf"), \
         patch("services.ingest_pipeline.extract_text", return_value=["Page 1"]), \
         patch("services.ingest_pipeline.chunk_text",
               return_value=[{"chunk_index": 0, "content": "text", "char_start": 0}]), \
         patch("services.ingest_pipeline.embed", return_value=[[0.1] * 1024]), \
         patch("services.ingest_pipeline.text", return_value=MagicMock()), \
         patch("services.ingest_pipeline.kg_extractor") as mock_kg:
        mock_neo4j.available = False
        db = _make_mock_db(doc)
        mock_session_cls.return_value = db
        from services import ingest_pipeline
        ingest_pipeline.run(1)
        mock_kg.extract_kg.assert_not_called()


def test_run_completes_even_if_kg_extractor_returns_empty():
    """Empty extract_kg result must not block ingest completion."""
    doc = _make_doc()
    with patch("services.ingest_pipeline.SessionLocal") as mock_session_cls, \
         patch("services.ingest_pipeline.neo4j_service") as mock_neo4j, \
         patch("services.ingest_pipeline._safe_file_path", return_value="/uploads/test.pdf"), \
         patch("services.ingest_pipeline.extract_text", return_value=["Page 1"]), \
         patch("services.ingest_pipeline.chunk_text",
               return_value=[{"chunk_index": 0, "content": "text", "char_start": 0}]), \
         patch("services.ingest_pipeline.embed", return_value=[[0.1] * 1024]), \
         patch("services.ingest_pipeline.text", return_value=MagicMock()), \
         patch("services.ingest_pipeline.kg_extractor") as mock_kg:
        mock_neo4j.available = True
        mock_kg.extract_kg.return_value = {}
        db = _make_mock_db(doc)
        mock_session_cls.return_value = db
        from services import ingest_pipeline
        ingest_pipeline.run(1)
        assert doc.ingest_status == "completed"
        mock_neo4j.create_entity_graph.assert_not_called()