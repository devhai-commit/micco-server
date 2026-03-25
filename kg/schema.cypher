// MICCO EKG Schema — run once at Neo4j setup
CREATE CONSTRAINT vat_tu_unique IF NOT EXISTS
    FOR (n:VatTu) REQUIRE n.document_id IS UNIQUE;
CREATE CONSTRAINT hop_dong_unique IF NOT EXISTS
    FOR (n:HopDong) REQUIRE n.document_id IS UNIQUE;
CREATE CONSTRAINT quy_dinh_unique IF NOT EXISTS
    FOR (n:QuyDinh) REQUIRE n.document_id IS UNIQUE;
CREATE CONSTRAINT su_co_unique IF NOT EXISTS
    FOR (n:SuCo) REQUIRE n.document_id IS UNIQUE;
CREATE CONSTRAINT ke_hoach_unique IF NOT EXISTS
    FOR (n:KeHoachMuaSam) REQUIRE n.document_id IS UNIQUE;
CREATE CONSTRAINT phieu_nhap_unique IF NOT EXISTS
    FOR (n:PhieuNhapKho) REQUIRE n.document_id IS UNIQUE;
CREATE CONSTRAINT chung_chi_unique IF NOT EXISTS
    FOR (n:ChungChi) REQUIRE n.document_id IS UNIQUE;
CREATE INDEX doc_node_idx IF NOT EXISTS
    FOR (n:Document) ON (n.document_id);