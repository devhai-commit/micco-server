-- ═══════════════════════════════════════════════════════════════
-- DocVault AI — Database Schema v3
-- Target: PostgreSQL (TimescaleDB)
-- Changes: + knowledge_entries, unified document_chunks
--          (source_type/source_id), department scoping
-- ═══════════════════════════════════════════════════════════════

-- ─── Extensions ─────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";  -- pgvector for embeddings

-- ─── Drop existing tables (safe re-run) ─────────────────────
DROP TABLE IF EXISTS document_chunks CASCADE;
DROP TABLE IF EXISTS chat_messages CASCADE;
DROP TABLE IF EXISTS knowledge_entries CASCADE;
DROP TABLE IF EXISTS documents CASCADE;
DROP TABLE IF EXISTS users CASCADE;
DROP TABLE IF EXISTS departments CASCADE;

-- ═══════════════════════════════════════════════════════════════
-- TABLES
-- ═══════════════════════════════════════════════════════════════

-- ─── Departments ────────────────────────────────────────────
CREATE TABLE departments (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(100)  NOT NULL UNIQUE,
    description     TEXT,
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_departments_name ON departments (name);

COMMENT ON TABLE departments IS 'Organizational departments for access control';

-- ─── Users ──────────────────────────────────────────────────
CREATE TABLE users (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(100)  NOT NULL,
    email           VARCHAR(255)  NOT NULL UNIQUE,
    hashed_password VARCHAR(255)  NOT NULL,
    role            VARCHAR(50)   NOT NULL DEFAULT 'Nhân viên',
    department_id   INTEGER       REFERENCES departments(id) ON DELETE SET NULL,
    avatar          VARCHAR(500),
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_users_email      ON users (email);
CREATE INDEX idx_users_role       ON users (role);
CREATE INDEX idx_users_department  ON users (department_id);

COMMENT ON TABLE users IS 'Application users with authentication credentials and department membership';

-- ─── Documents ──────────────────────────────────────────────
CREATE TABLE documents (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(255)  NOT NULL,
    type            VARCHAR(10)   NOT NULL,
    category        VARCHAR(50)   NOT NULL,
    size            VARCHAR(50)   NOT NULL,
    size_bytes      BIGINT        NOT NULL DEFAULT 0,
    owner_id        INTEGER       NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    department_id   INTEGER       REFERENCES departments(id) ON DELETE SET NULL,
    tags            JSONB         NOT NULL DEFAULT '[]'::jsonb,
    thumbnail       VARCHAR(500),
    visibility      VARCHAR(20)   NOT NULL DEFAULT 'internal',
    status          VARCHAR(20)   NOT NULL DEFAULT 'Active',
    file_path       VARCHAR(500),
    ingest_status   VARCHAR(20)   DEFAULT 'pending',
    ingest_error    TEXT,
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_documents_owner      ON documents (owner_id);
CREATE INDEX idx_documents_department ON documents (department_id);
CREATE INDEX idx_documents_type       ON documents (type);
CREATE INDEX idx_documents_category   ON documents (category);
CREATE INDEX idx_documents_status     ON documents (status);
CREATE INDEX idx_documents_created    ON documents (created_at DESC);
CREATE INDEX idx_documents_tags       ON documents USING GIN (tags);
CREATE INDEX idx_documents_name_trgm  ON documents (LOWER(name) varchar_pattern_ops);

COMMENT ON TABLE documents IS 'Uploaded documents with metadata, file references, and department ownership';

-- ─── Knowledge Entries ──────────────────────────────────────
CREATE TABLE knowledge_entries (
    id              SERIAL PRIMARY KEY,
    title           VARCHAR(500)  NOT NULL,
    content_html    TEXT          NOT NULL,
    content_text    TEXT          NOT NULL,
    category        VARCHAR(100)  NOT NULL DEFAULT 'Chung',
    tags            JSONB         NOT NULL DEFAULT '[]'::jsonb,
    owner_id        INTEGER       NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    department_id   INTEGER       REFERENCES departments(id) ON DELETE SET NULL,
    visibility      VARCHAR(20)   NOT NULL DEFAULT 'internal',
    status          VARCHAR(20)   NOT NULL DEFAULT 'Active',
    ingest_status   VARCHAR(20)   DEFAULT 'pending',
    ingest_error    TEXT,
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_knowledge_owner      ON knowledge_entries (owner_id);
CREATE INDEX idx_knowledge_department  ON knowledge_entries (department_id);
CREATE INDEX idx_knowledge_category    ON knowledge_entries (category);
CREATE INDEX idx_knowledge_visibility  ON knowledge_entries (visibility);
CREATE INDEX idx_knowledge_status      ON knowledge_entries (status);
CREATE INDEX idx_knowledge_updated     ON knowledge_entries (updated_at DESC);
CREATE INDEX idx_knowledge_tags        ON knowledge_entries USING GIN (tags);

COMMENT ON TABLE knowledge_entries IS 'Manually entered knowledge base entries with WYSIWYG HTML content';

-- ─── Document Chunks (unified: documents + knowledge) ───────
-- Polymorphic: source_type ('document'|'knowledge') + source_id
-- Single ivfflat index covers all embeddings for chatbot search.
CREATE TABLE document_chunks (
    id              SERIAL PRIMARY KEY,
    source_type     VARCHAR(20)   NOT NULL DEFAULT 'document',
    source_id       INTEGER       NOT NULL,
    chunk_index     INTEGER       NOT NULL,
    content         TEXT          NOT NULL,
    embedding       vector(512),   -- text-embedding-3-small (512-dim)
    token_count     INTEGER       DEFAULT 0,
    department_id   INTEGER       REFERENCES departments(id) ON DELETE SET NULL,
    metadata        JSONB         DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_chunks_source UNIQUE (source_type, source_id, chunk_index)
);

CREATE INDEX idx_chunks_source_type ON document_chunks (source_type);
CREATE INDEX idx_chunks_source      ON document_chunks (source_type, source_id);
CREATE INDEX idx_chunks_department   ON document_chunks (department_id);
CREATE INDEX idx_chunks_embedding   ON document_chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

COMMENT ON TABLE document_chunks IS 'Unified text chunks with vector embeddings for semantic search (RAG) — covers both documents and knowledge entries';

-- ─── Chat Messages ──────────────────────────────────────────
CREATE TABLE chat_messages (
    id              SERIAL PRIMARY KEY,
    user_id         INTEGER       NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role            VARCHAR(10)   NOT NULL CHECK (role IN ('user', 'ai')),
    content         TEXT          NOT NULL,
    sources         JSONB         NOT NULL DEFAULT '[]'::jsonb,
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_chat_user      ON chat_messages (user_id);
CREATE INDEX idx_chat_created   ON chat_messages (user_id, created_at ASC);

COMMENT ON TABLE chat_messages IS 'Chat history between users and AI assistant';


-- ═══════════════════════════════════════════════════════════════
-- STORED PROCEDURES & FUNCTIONS
-- ═══════════════════════════════════════════════════════════════

-- ─── Get Dashboard Stats (department-scoped) ────────────────
CREATE OR REPLACE FUNCTION get_dashboard_stats(p_department_id INTEGER DEFAULT NULL)
RETURNS TABLE (
    total_files     BIGINT,
    storage_used    BIGINT,
    recent_uploads  BIGINT,
    team_members    BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        (SELECT COUNT(*) FROM documents
            WHERE (p_department_id IS NULL OR department_id = p_department_id))::BIGINT AS total_files,
        (SELECT COALESCE(SUM(size_bytes), 0) FROM documents
            WHERE (p_department_id IS NULL OR department_id = p_department_id))::BIGINT AS storage_used,
        (SELECT COUNT(*) FROM documents
            WHERE created_at >= NOW() - INTERVAL '7 days'
            AND (p_department_id IS NULL OR department_id = p_department_id))::BIGINT AS recent_uploads,
        (SELECT COUNT(*) FROM users
            WHERE (p_department_id IS NULL OR department_id = p_department_id))::BIGINT AS team_members;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_dashboard_stats IS 'Returns dashboard statistics, optionally scoped to a department';


-- ─── Get Storage By Type (department-scoped) ────────────────
CREATE OR REPLACE FUNCTION get_storage_by_type(p_department_id INTEGER DEFAULT NULL)
RETURNS TABLE (
    doc_type    VARCHAR(10),
    total_bytes BIGINT,
    file_count  BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        d.type          AS doc_type,
        COALESCE(SUM(d.size_bytes), 0)::BIGINT AS total_bytes,
        COUNT(*)::BIGINT AS file_count
    FROM documents d
    WHERE (p_department_id IS NULL OR d.department_id = p_department_id)
    GROUP BY d.type
    ORDER BY total_bytes DESC;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_storage_by_type IS 'Returns storage usage grouped by document type, optionally scoped to a department';


-- ─── Get Uploads Over Time (department-scoped) ──────────────
CREATE OR REPLACE FUNCTION get_uploads_over_time(p_department_id INTEGER DEFAULT NULL)
RETURNS TABLE (
    month_name  TEXT,
    upload_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        TO_CHAR(gs.month, 'Mon')  AS month_name,
        COALESCE(cnt.total, 0)::BIGINT    AS upload_count
    FROM (
        SELECT generate_series(
            DATE_TRUNC('month', NOW()) - INTERVAL '5 months',
            DATE_TRUNC('month', NOW()),
            '1 month'::interval
        ) AS month
    ) gs
    LEFT JOIN (
        SELECT
            DATE_TRUNC('month', created_at) AS month,
            COUNT(*) AS total
        FROM documents
        WHERE (p_department_id IS NULL OR department_id = p_department_id)
        GROUP BY DATE_TRUNC('month', created_at)
    ) cnt ON gs.month = cnt.month
    ORDER BY gs.month ASC;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_uploads_over_time IS 'Returns upload counts per month for the last 6 months, optionally scoped to a department';


-- ─── Search Documents (department-scoped) ───────────────────
CREATE OR REPLACE FUNCTION search_documents(
    p_search        TEXT DEFAULT NULL,
    p_type          VARCHAR(10) DEFAULT NULL,
    p_category      VARCHAR(50) DEFAULT NULL,
    p_department_id INTEGER DEFAULT NULL,
    p_owner_id      INTEGER DEFAULT NULL,
    p_status        VARCHAR(20) DEFAULT NULL,
    p_limit         INTEGER DEFAULT 100,
    p_offset        INTEGER DEFAULT 0
)
RETURNS TABLE (
    doc_id           INTEGER,
    doc_name         VARCHAR(255),
    doc_type         VARCHAR(10),
    doc_category     VARCHAR(50),
    doc_size         VARCHAR(50),
    owner_name       VARCHAR(100),
    department_name  VARCHAR(100),
    doc_tags         JSONB,
    doc_status       VARCHAR(20),
    doc_date         TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        d.id,
        d.name,
        d.type,
        d.category,
        d.size,
        u.name            AS owner_name,
        dep.name          AS department_name,
        d.tags,
        d.status,
        d.created_at      AS doc_date
    FROM documents d
    JOIN users u ON d.owner_id = u.id
    LEFT JOIN departments dep ON d.department_id = dep.id
    WHERE
        (p_search IS NULL OR LOWER(d.name) LIKE '%' || LOWER(p_search) || '%'
            OR d.tags::text ILIKE '%' || p_search || '%')
        AND (p_type IS NULL OR d.type = p_type)
        AND (p_category IS NULL OR d.category = p_category)
        AND (p_department_id IS NULL OR d.department_id = p_department_id)
        AND (p_owner_id IS NULL OR d.owner_id = p_owner_id)
        AND (p_status IS NULL OR d.status = p_status)
    ORDER BY d.created_at DESC
    LIMIT p_limit OFFSET p_offset;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION search_documents IS 'Search and filter documents with pagination, optionally scoped to a department';


-- ─── Delete Document (with cleanup) ─────────────────────────
CREATE OR REPLACE FUNCTION delete_document(p_doc_id INTEGER, p_user_id INTEGER)
RETURNS TABLE (
    deleted_id  INTEGER,
    file_path   VARCHAR(500)
) AS $$
BEGIN
    RETURN QUERY
    DELETE FROM documents
    WHERE id = p_doc_id
    RETURNING id AS deleted_id, documents.file_path;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION delete_document IS 'Delete a document and return its file path for filesystem cleanup';


-- ─── Semantic Search via Embeddings (unified + dept-scoped) ─
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

COMMENT ON FUNCTION search_chunks_by_embedding IS 'Semantic search across documents + knowledge — filtered by department_id on chunks';


-- ─── Get User Chat History ──────────────────────────────────
CREATE OR REPLACE FUNCTION get_chat_history(
    p_user_id   INTEGER,
    p_limit     INTEGER DEFAULT 100
)
RETURNS TABLE (
    msg_id      INTEGER,
    msg_role    VARCHAR(10),
    msg_content TEXT,
    msg_sources JSONB,
    msg_date    TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        cm.id,
        cm.role,
        cm.content,
        cm.sources,
        cm.created_at
    FROM chat_messages cm
    WHERE cm.user_id = p_user_id
    ORDER BY cm.created_at ASC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_chat_history IS 'Get chat history for a specific user';


-- ─── Insert Chat Message Pair (user + AI) ───────────────────
CREATE OR REPLACE FUNCTION insert_chat_pair(
    p_user_id       INTEGER,
    p_user_message  TEXT,
    p_ai_response   TEXT,
    p_ai_sources    JSONB DEFAULT '[]'::jsonb
)
RETURNS TABLE (
    ai_msg_id   INTEGER,
    ai_content  TEXT,
    ai_sources  JSONB
) AS $$
DECLARE
    v_ai_id INTEGER;
BEGIN
    -- Insert user message
    INSERT INTO chat_messages (user_id, role, content, sources)
    VALUES (p_user_id, 'user', p_user_message, '[]'::jsonb);

    -- Insert AI response
    INSERT INTO chat_messages (user_id, role, content, sources)
    VALUES (p_user_id, 'ai', p_ai_response, p_ai_sources)
    RETURNING id INTO v_ai_id;

    RETURN QUERY
    SELECT v_ai_id, p_ai_response, p_ai_sources;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION insert_chat_pair IS 'Insert a user message and AI response as a pair, returns the AI message';


-- ─── Entity Embeddings (GraphRAG Local Search) ───────────────
CREATE TABLE entity_embeddings (
    id              SERIAL PRIMARY KEY,
    entity_name     TEXT        NOT NULL,
    entity_label    TEXT        NOT NULL,
    description     TEXT        NOT NULL DEFAULT '',
    embedding       vector(512),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_entity UNIQUE (entity_name, entity_label)
);

CREATE INDEX idx_entity_emb_embedding
    ON entity_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 50);
CREATE INDEX idx_entity_emb_label ON entity_embeddings (entity_label);

COMMENT ON TABLE entity_embeddings IS 'Vector embeddings for KG entities — GraphRAG Local Search';

-- ─── Communities (GraphRAG Global Search) ─────────────────────
CREATE TABLE communities (
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

CREATE INDEX idx_communities_level ON communities (level);
CREATE INDEX idx_communities_embedding
    ON communities USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 20);

COMMENT ON TABLE communities IS 'Leiden community clusters with LLM summaries — GraphRAG Global Search';

-- ─── Community Membership ─────────────────────────────────────
CREATE TABLE community_entities (
    community_id    INTEGER NOT NULL REFERENCES communities(id) ON DELETE CASCADE,
    entity_name     TEXT    NOT NULL,
    entity_label    TEXT    NOT NULL,
    PRIMARY KEY (community_id, entity_name, entity_label)
);

CREATE INDEX idx_community_entities_entity ON community_entities (entity_name, entity_label);

COMMENT ON TABLE community_entities IS 'Maps entities to their detected communities';

-- ─── Entity vector similarity search ──────────────────────────
CREATE OR REPLACE FUNCTION search_entities_by_embedding(
    p_query_embedding vector(512),
    p_limit           INTEGER DEFAULT 10
)
RETURNS TABLE (
    entity_name TEXT, entity_label TEXT, description TEXT, similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT ee.entity_name, ee.entity_label, ee.description,
           1 - (ee.embedding <=> p_query_embedding)::FLOAT AS similarity
    FROM entity_embeddings ee
    WHERE ee.embedding IS NOT NULL
    ORDER BY ee.embedding <=> p_query_embedding ASC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;


-- ═══════════════════════════════════════════════════════════════
-- SEED DATA
-- ═══════════════════════════════════════════════════════════════

-- ─── Departments ────────────────────────────────────────────
INSERT INTO departments (name, description) VALUES
    ('Kế toán',     'Phòng Kế toán - Tài chính'),
    ('Nhân sự',     'Phòng Nhân sự'),
    ('Kỹ thuật',    'Phòng Kỹ thuật - Công nghệ'),
    ('Kinh doanh',  'Phòng Kinh doanh - Marketing'),
    ('Pháp chế',    'Phòng Pháp chế - Hợp đồng'),
    ('Ban Giám đốc','Ban lãnh đạo công ty')
ON CONFLICT (name) DO NOTHING;


-- ═══════════════════════════════════════════════════════════════
-- DONE
-- ═══════════════════════════════════════════════════════════════
DO $$
BEGIN
    RAISE NOTICE '✅ Schema v3 created: 6 tables (departments, users, documents, knowledge_entries, document_chunks, chat_messages), 9 functions, seed data inserted';
END $$;
