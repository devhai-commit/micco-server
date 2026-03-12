-- ═══════════════════════════════════════════════════════════════
-- DocVault AI — Database Schema v2
-- Target: PostgreSQL (TimescaleDB)
-- Changes: + departments, + department access control,
--          + document_chunks (embedding/RAG)
-- ═══════════════════════════════════════════════════════════════

-- ─── Extensions ─────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";  -- pgvector for embeddings

-- ─── Drop existing tables (safe re-run) ─────────────────────
DROP TABLE IF EXISTS document_chunks CASCADE;
DROP TABLE IF EXISTS chat_messages CASCADE;
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
    role            VARCHAR(50)   NOT NULL DEFAULT 'Member',
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

-- ─── Document Chunks (for text embedding / RAG) ─────────────
CREATE TABLE document_chunks (
    id              SERIAL PRIMARY KEY,
    document_id     INTEGER       NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index     INTEGER       NOT NULL,
    content         TEXT          NOT NULL,
    embedding       vector(1536),   -- OpenAI ada-002 dimension; adjust as needed
    token_count     INTEGER       DEFAULT 0,
    metadata        JSONB         DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),

    UNIQUE (document_id, chunk_index)
);

CREATE INDEX idx_chunks_document   ON document_chunks (document_id);
CREATE INDEX idx_chunks_embedding  ON document_chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

COMMENT ON TABLE document_chunks IS 'Text chunks with vector embeddings for semantic search (RAG)';

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


-- ─── Semantic Search via Embeddings ─────────────────────────
CREATE OR REPLACE FUNCTION search_chunks_by_embedding(
    p_query_embedding vector(1536),
    p_department_id   INTEGER DEFAULT NULL,
    p_limit           INTEGER DEFAULT 10
)
RETURNS TABLE (
    chunk_id        INTEGER,
    document_id     INTEGER,
    document_name   VARCHAR(255),
    chunk_content   TEXT,
    similarity      FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        dc.id               AS chunk_id,
        dc.document_id,
        d.name              AS document_name,
        dc.content          AS chunk_content,
        1 - (dc.embedding <=> p_query_embedding)::FLOAT AS similarity
    FROM document_chunks dc
    JOIN documents d ON dc.document_id = d.id
    WHERE
        dc.embedding IS NOT NULL
        AND (p_department_id IS NULL OR d.department_id = p_department_id)
    ORDER BY dc.embedding <=> p_query_embedding ASC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION search_chunks_by_embedding IS 'Semantic search: find most similar text chunks by vector distance, optionally scoped to a department';


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
    RAISE NOTICE '✅ Schema v2 created: 5 tables (departments, users, documents, document_chunks, chat_messages), 9 functions, seed data inserted';
END $$;
