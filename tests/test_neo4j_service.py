import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_driver():
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return driver, session


def test_connect_sets_available_true(mock_driver):
    driver, _ = mock_driver
    with patch("services.neo4j_service.GraphDatabase.driver", return_value=driver):
        from services.neo4j_service import Neo4jService
        svc = Neo4jService.__new__(Neo4jService)
        svc.available = False
        svc._driver = None
        svc.uri = "bolt://localhost:7687"
        svc.user = "neo4j"
        svc.password = "test"
        svc.connect()
        assert svc.available is True


def test_connect_sets_available_false_on_error():
    with patch("services.neo4j_service.GraphDatabase.driver", side_effect=Exception("conn refused")):
        from services.neo4j_service import Neo4jService
        svc = Neo4jService.__new__(Neo4jService)
        svc.available = True
        svc._driver = None
        svc.uri = "bolt://localhost:7687"
        svc.user = "neo4j"
        svc.password = "wrong"
        svc.connect()
        assert svc.available is False


def test_merge_document_node_runs_cypher(mock_driver):
    driver, session = mock_driver
    with patch("services.neo4j_service.GraphDatabase.driver", return_value=driver):
        from services.neo4j_service import Neo4jService
        svc = Neo4jService.__new__(Neo4jService)
        svc.available = True
        svc._driver = driver
        doc = {
            "document_id": 42,
            "label": "VatTu",
            "ten": "Thep CT3",
            "owner": "admin",
            "created_at": "2026-01-01",
        }
        svc.merge_document_node(doc)
        assert session.run.called
        cypher_call = session.run.call_args
        assert "MERGE" in cypher_call[0][0]
        assert cypher_call[1]["document_id"] == 42


def test_merge_document_node_noop_when_unavailable(mock_driver):
    driver, session = mock_driver
    from services.neo4j_service import Neo4jService
    svc = Neo4jService.__new__(Neo4jService)
    svc.available = False
    svc._driver = driver
    svc.merge_document_node({"document_id": 1, "label": "VatTu", "ten": "", "owner": "", "created_at": ""})
    session.run.assert_not_called()


def test_create_entity_graph_merges_entity_nodes(mock_driver):
    driver, session = mock_driver
    from services.neo4j_service import Neo4jService
    svc = Neo4jService.__new__(Neo4jService)
    svc.available = True
    svc._driver = driver

    entities = [{"name": "Công ty ABC", "label": "NhaCungCap"}]
    svc.create_entity_graph(document_id=1, entities=entities, relationships=[])

    assert session.run.called
    cypher_call = session.run.call_args_list[0]
    assert "MERGE" in cypher_call[0][0]
    assert "NhaCungCap" in cypher_call[0][0]


def test_create_entity_graph_noop_when_unavailable(mock_driver):
    driver, session = mock_driver
    from services.neo4j_service import Neo4jService
    svc = Neo4jService.__new__(Neo4jService)
    svc.available = False
    svc._driver = driver

    svc.create_entity_graph(
        document_id=1,
        entities=[{"name": "X", "label": "HopDong"}],
        relationships=[],
    )
    session.run.assert_not_called()


def test_create_entity_graph_skips_invalid_label(mock_driver):
    driver, session = mock_driver
    from services.neo4j_service import Neo4jService
    svc = Neo4jService.__new__(Neo4jService)
    svc.available = True
    svc._driver = driver

    entities = [{"name": "Bad", "label": "NotAValidLabel"}]
    svc.create_entity_graph(document_id=1, entities=entities, relationships=[])
    # Should not call session.run for the entity MERGE
    for call in session.run.call_args_list:
        assert "NotAValidLabel" not in call[0][0]


def test_create_entity_graph_merges_mentions_edge(mock_driver):
    driver, session = mock_driver
    from services.neo4j_service import Neo4jService
    svc = Neo4jService.__new__(Neo4jService)
    svc.available = True
    svc._driver = driver

    entities = [{"name": "HĐ-001", "label": "HopDong"}]
    svc.create_entity_graph(document_id=7, entities=entities, relationships=[])

    all_cyphers = [call[0][0] for call in session.run.call_args_list]
    assert any("MENTIONS" in c for c in all_cyphers), \
        "Expected a MENTIONS edge Cypher call"


def test_create_entity_graph_sets_attributes(mock_driver):
    driver, session = mock_driver
    from services.neo4j_service import Neo4jService
    svc = Neo4jService.__new__(Neo4jService)
    svc.available = True
    svc._driver = driver

    entities = [{
        "name": "Công ty ABC",
        "label": "NhaCungCap",
        "attributes": {"dia_chi": "Hà Nội", "ma_so_thue": "012345"},
    }]
    svc.create_entity_graph(document_id=1, entities=entities, relationships=[])

    # Find the MERGE entity call (first session.run call)
    cypher_call = session.run.call_args_list[0]
    cypher = cypher_call[0][0]
    kwargs = cypher_call[1]
    assert "e.dia_chi" in cypher
    assert "e.ma_so_thue" in cypher
    assert kwargs["dia_chi"] == "Hà Nội"
    assert kwargs["ma_so_thue"] == "012345"


def test_create_entity_graph_filters_disallowed_attributes(mock_driver):
    driver, session = mock_driver
    from services.neo4j_service import Neo4jService
    svc = Neo4jService.__new__(Neo4jService)
    svc.available = True
    svc._driver = driver

    entities = [{
        "name": "HĐ-001",
        "label": "HopDong",
        "attributes": {"ngay": "15/05/2025", "evil_key": "DROP DATABASE"},
    }]
    svc.create_entity_graph(document_id=1, entities=entities, relationships=[])

    cypher_call = session.run.call_args_list[0]
    cypher = cypher_call[0][0]
    kwargs = cypher_call[1]
    assert "e.ngay" in cypher
    assert "evil_key" not in cypher
    assert "evil_key" not in kwargs


def test_allowed_labels_covers_all_domain_labels():
    """_ALLOWED_LABELS must include all domain NodeLabel values."""
    from services.neo4j_service import _ALLOWED_LABELS
    from kg.ontology import NodeLabel
    domain_labels = {
        label.value for label in NodeLabel
        if label is not NodeLabel.DOCUMENT
    }
    assert domain_labels.issubset(_ALLOWED_LABELS), (
        f"Missing labels: {domain_labels - _ALLOWED_LABELS}"
    )
