-- ═══════════════════════════════════════════════════════════════
-- Migration 008: GraphRAG Local + Global search support
-- Adds: entity_embeddings, communities, community_entities
-- ═══════════════════════════════════════════════════════════════

-- ─── Entity embeddings for Local Search ──────────────────────
CREATE TABLE IF NOT EXISTS entity_embeddings (
    id              SERIAL PRIMARY KEY,
    entity_name     TEXT        NOT NULL,
    entity_label    TEXT        NOT NULL,
    description     TEXT        NOT NULL DEFAULT '',
    embedding       vector(512),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_entity UNIQUE (entity_name, entity_label)
);

CREATE INDEX IF NOT EXISTS idx_entity_emb_embedding
    ON entity_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 50);

CREATE INDEX IF NOT EXISTS idx_entity_emb_label
    ON entity_embeddings (entity_label);

COMMENT ON TABLE entity_embeddings IS
    'Vector embeddings for KG entities — used by GraphRAG Local Search';

-- ─── Communities for Global Search ───────────────────────────
CREATE TABLE IF NOT EXISTS communities (
    id              SERIAL PRIMARY KEY,
    level           INTEGER     NOT NULL DEFAULT 0,
    title           TEXT,
    summary         TEXT,
    full_content    TEXT,
    embedding       vector(512),
    entity_count    INTEGER     NOT NULL DEFAULT 0,
    relationship_count INTEGER  NOT NULL DEFAULT 0,
    rank            FLOAT       NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_communities_level
    ON communities (level);

CREATE INDEX IF NOT EXISTS idx_communities_embedding
    ON communities USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 20);

COMMENT ON TABLE communities IS
    'Leiden community clusters with LLM-generated summaries — used by GraphRAG Global Search';

-- ─── Community membership ────────────────────────────────────
CREATE TABLE IF NOT EXISTS community_entities (
    community_id    INTEGER NOT NULL REFERENCES communities(id) ON DELETE CASCADE,
    entity_name     TEXT    NOT NULL,
    entity_label    TEXT    NOT NULL,
    PRIMARY KEY (community_id, entity_name, entity_label)
);

CREATE INDEX IF NOT EXISTS idx_community_entities_entity
    ON community_entities (entity_name, entity_label);

COMMENT ON TABLE community_entities IS
    'Maps entities to their detected communities';

-- ─── Search function: entity vector similarity ───────────────
CREATE OR REPLACE FUNCTION search_entities_by_embedding(
    p_query_embedding vector(512),
    p_limit           INTEGER DEFAULT 10
)
RETURNS TABLE (
    entity_name     TEXT,
    entity_label    TEXT,
    description     TEXT,
    similarity      FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        ee.entity_name,
        ee.entity_label,
        ee.description,
        1 - (ee.embedding <=> p_query_embedding)::FLOAT AS similarity
    FROM entity_embeddings ee
    WHERE ee.embedding IS NOT NULL
    ORDER BY ee.embedding <=> p_query_embedding ASC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION search_entities_by_embedding IS
    'Semantic search over entity embeddings for GraphRAG Local Search';
