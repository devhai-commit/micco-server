# backend/tests/test_documents_router.py
"""Tests for document upload BackgroundTask enqueue behavior."""
import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

# Stub heavy modules before any import of routers.documents
def _stub(name, obj=None):
    sys.modules.setdefault(name, obj or types.ModuleType(name))

_stub("database"); _stub("auth"); _stub("config")
_stub("services"); _stub("services.ingest_pipeline")

sys.modules["config"].UPLOAD_DIR = "/tmp"
sys.modules["config"].MAX_FILE_SIZE = 100 * 1024 * 1024

# Add stub run function to ingest_pipeline module
sys.modules["services.ingest_pipeline"].run = MagicMock()


def test_upload_enqueues_ingest_run_for_each_file():
    """Each uploaded file must result in ingest_pipeline.run being added to BackgroundTasks."""
    from fastapi import BackgroundTasks
    from routers.documents import upload_documents

    # Build a minimal mock Document whose attrs satisfy build_document_response()
    doc = MagicMock()
    doc.id = 42
    doc.name = "test.pdf"; doc.type = "PDF"; doc.category = "Tài liệu"
    doc.size = "5 B"; doc.owner_name = "admin"; doc.department_name = None
    doc.date = "01/01/2026"; doc.tags = []; doc.status = "Active"

    mock_db = MagicMock()
    mock_user = MagicMock()
    mock_user.id = 1; mock_user.department_id = None; mock_user.role = "Member"

    mock_file = AsyncMock()
    mock_file.filename = "test.pdf"
    mock_file.read = AsyncMock(return_value=b"hello")

    bg = BackgroundTasks()

    with patch("routers.documents.Document", return_value=doc), \
         patch("routers.documents.ingest_pipeline.run") as mock_run, \
         patch("builtins.open", MagicMock()), \
         patch("routers.documents.os.path.join", return_value="/tmp/x.pdf"), \
         patch("routers.documents.uuid.uuid4", return_value=MagicMock(hex="abc")):
        asyncio.run(upload_documents(
            background_tasks=bg,
            files=[mock_file],
            tags=None,
            category=None,
            department_id=None,
            db=mock_db,
            current_user=mock_user,
        ))

    # BackgroundTasks stores tasks as a list of BackgroundTask objects with .func / .args
    task_funcs = [task.func for task in bg.tasks]
    assert mock_run in task_funcs, "ingest_pipeline.run must be added to BackgroundTasks"
    # Verify it was enqueued with the correct doc.id
    ingest_task = next(t for t in bg.tasks if t.func is mock_run)
    assert ingest_task.args == (doc.id,), f"Expected (42,), got {ingest_task.args}"
