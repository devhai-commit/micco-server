import pytest
from kg.ontology import NodeLabel, RelType


def test_node_labels_count():
    # 17 domain nodes + Document + DocumentChunk (structural nodes for HAS_CHUNK path) = 19
    assert len(list(NodeLabel)) == 19


def test_rel_types_count():
    assert len(list(RelType)) == 38


def test_node_label_values_are_strings():
    for label in NodeLabel:
        assert isinstance(label.value, str)
        assert label.value[0].isupper(), f"{label.value} should start with uppercase"


def test_key_node_labels_present():
    labels = {nl.value for nl in NodeLabel}
    assert "VatTu" in labels
    assert "NhaCungCap" in labels
    assert "HopDong" in labels
    assert "DocumentChunk" in labels
    assert "Document" in labels


def test_key_rel_types_present():
    rels = {r.value for r in RelType}
    assert "BAO_GOM" in rels
    assert "CUNG_CAP" in rels
    assert "HAS_CHUNK" in rels
    assert "TUAN_THU_THEO" in rels