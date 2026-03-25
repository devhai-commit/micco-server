-- ═══════════════════════════════════════════════════════════════
-- Migration 007: Switch embedding from vector(1024) to vector(512)
-- Model: text-embedding-3-small with 512 dimensions
-- ═══════════════════════════════════════════════════════════════

-- 1. Drop the ivfflat index (it's dimension-bound)
DROP INDEX IF EXISTS idx_chunks_embedding;

-- 2. Clear old embeddings (they are 1024-dim, incompatible with new 512-dim)
UPDATE document_chunks SET embedding = NULL;

-- 3. Alter column to vector(512)
ALTER TABLE document_chunks
    ALTER COLUMN embedding TYPE vector(512);

-- 4. Recreate ivfflat index
CREATE INDEX idx_chunks_embedding
    ON document_chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- 5. Update search function signature
CREATE OR REPLACE FUNCTION search_chunks_by_embedding(
    p_query_embedding vector(512),
    p_department_id   INTEGER DEFAULT NULL,
    p_limit           INTEGER DEFAULT 10
)
RETURNS TABLE (
    chunk_id        INTEGER,
    source_type     VARCHAR(20),
    source_id       INTEGER,
    source_name     TEXT,
    chunk_content   TEXT,
    similarity      FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        dc.id               AS chunk_id,
        dc.source_type,
        dc.source_id,
        COALESCE(d.name, ke.title, 'Unknown')::TEXT AS source_name,
        dc.content          AS chunk_content,
        1 - (dc.embedding <=> p_query_embedding)::FLOAT AS similarity
    FROM document_chunks dc
    LEFT JOIN documents d
        ON dc.source_type = 'document' AND dc.source_id = d.id
    LEFT JOIN knowledge_entries ke
        ON dc.source_type = 'knowledge' AND dc.source_id = ke.id
    WHERE
        dc.embedding IS NOT NULL
        AND (p_department_id IS NULL OR dc.department_id = p_department_id)
    ORDER BY dc.embedding <=> p_query_embedding ASC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- ═══════════════════════════════════════════════════════════════
-- NOTE: After running this migration, re-ingest all documents
-- to regenerate embeddings with the new 512-dim model.
-- ═══════════════════════════════════════════════════════════════
