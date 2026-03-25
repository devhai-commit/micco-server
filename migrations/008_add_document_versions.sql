-- ═══════════════════════════════════════════════════════════════
-- Migration 008: Add document_versions table
-- Lưu lịch sử phiên bản cho tài liệu
-- ═══════════════════════════════════════════════════════════════

-- ─── Drop if exists (safe re-run) ───────────────────────────
DROP TABLE IF EXISTS document_versions CASCADE;

-- ─── Table ──────────────────────────────────────────────────
CREATE TABLE document_versions (
    id              SERIAL PRIMARY KEY,
    document_id     INTEGER       NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    version_number  INTEGER       NOT NULL,
    version_label   VARCHAR(50)   NOT NULL DEFAULT 'V 1.0',
    file_path       VARCHAR(500),
    size            VARCHAR(50),
    size_bytes      BIGINT        DEFAULT 0,
    change_note     TEXT,
    created_by      INTEGER       NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    is_current      BOOLEAN       NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_document_version UNIQUE (document_id, version_number)
);

CREATE INDEX idx_doc_versions_document ON document_versions (document_id);
CREATE INDEX idx_doc_versions_current  ON document_versions (document_id) WHERE is_current IS TRUE;
CREATE INDEX idx_doc_versions_created  ON document_versions (document_id, created_at DESC);

COMMENT ON TABLE document_versions IS 'Lịch sử phiên bản tài liệu — mỗi lần upload lại tạo version mới';

-- ─── Backfill: Create V1.0 for all existing documents ───────
INSERT INTO document_versions (document_id, version_number, version_label, file_path, size, size_bytes, change_note, created_by, is_current, created_at)
SELECT
    d.id,
    1,
    'V 1.0',
    d.file_path,
    d.size,
    d.size_bytes,
    'Phiên bản gốc',
    d.owner_id,
    TRUE,
    d.created_at
FROM documents d
WHERE NOT EXISTS (
    SELECT 1 FROM document_versions dv WHERE dv.document_id = d.id
);

DO $$
BEGIN
    RAISE NOTICE '✅ Migration 008: document_versions table created, existing documents backfilled with V1.0';
END $$;
