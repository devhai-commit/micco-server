import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


def _make_app_with_overrides(db_override, user_override):
    """Create a fresh FastAPI app with dependency overrides for testing."""
    # Remove cached module so router is re-imported with current env each time
    import sys
    sys.modules.pop("routers.ingest", None)

    from routers.ingest import router, get_db, get_current_user
    app = FastAPI()
    app.include_router(router)

    app.dependency_overrides[get_db] = db_override
    app.dependency_overrides[get_current_user] = user_override
    return app


def _mock_user(role="Admin"):
    user = MagicMock()
    user.id = 1
    user.role = role
    return user


def _mock_doc(doc_id=1, status="pending"):
    doc = MagicMock()
    doc.id = doc_id
    doc.ingest_status = status
    doc.ingest_error = None
    return doc


def test_trigger_returns_queued():
    doc = _mock_doc(1)
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = doc

    def override_db():
        yield db

    def override_user():
        return _mock_user()

    with patch("routers.ingest.ingest_pipeline.run") as mock_run:
        app = _make_app_with_overrides(override_db, override_user)
        with TestClient(app) as c:
            resp = c.post("/api/ingest/1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["document_id"] == 1
            assert data["status"] == "queued"


def test_trigger_404_for_missing_doc():
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None

    def override_db():
        yield db

    def override_user():
        return _mock_user()

    app = _make_app_with_overrides(override_db, override_user)
    with TestClient(app) as c:
        resp = c.post("/api/ingest/999")
        assert resp.status_code == 404


def test_status_returns_current_status():
    doc = _mock_doc(1, "completed")
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = doc

    def override_db():
        yield db

    def override_user():
        return _mock_user()

    app = _make_app_with_overrides(override_db, override_user)
    with TestClient(app) as c:
        resp = c.get("/api/ingest/1/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"


def test_batch_requires_admin():
    db = MagicMock()

    def override_db():
        yield db

    def override_user():
        return _mock_user(role="Member")

    app = _make_app_with_overrides(override_db, override_user)
    with TestClient(app) as c:
        resp = c.post("/api/ingest/batch")
        assert resp.status_code == 403


def test_batch_route_not_shadowed_by_document_id_route():
    """Verify /api/ingest/batch is matched as the literal route, not as document_id='batch'."""
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = []

    def override_db():
        yield db

    def override_user():
        return _mock_user(role="Admin")

    # or_() is called with MagicMock columns from the stubbed Document model,
    # which confuses SQLAlchemy's expression coercion. Patch or_ in the ingest
    # router module to avoid coercion so the mock filter chain proceeds normally.
    # Must patch BEFORE the module is re-imported in _make_app_with_overrides.
    import sys
    sys.modules.pop("routers.ingest", None)
    with patch.dict("sys.modules"):  # isolate sys.modules changes
        with patch("sqlalchemy.or_", return_value=MagicMock()):
            app = _make_app_with_overrides(override_db, override_user)
            with TestClient(app) as c:
                resp = c.post("/api/ingest/batch")
                assert resp.status_code != 422
                assert resp.status_code == 200