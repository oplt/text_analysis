"""RAG research assistant: vector/FTS, conversations, traces, scope snapshots.

Revision ID: e7f8a9b0c1d2
Revises: d5e6f7a8b9c0
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "e7f8a9b0c1d2"
down_revision = "d5e6f7a8b9c0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    is_pg = bind.dialect.name == "postgresql"

    if is_pg:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
        # embedding column may already exist from runtime DDL; add if missing
        op.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'rag_chunks' AND column_name = 'embedding'
                ) THEN
                    ALTER TABLE rag_chunks ADD COLUMN embedding vector(1536);
                END IF;
            END $$;
            """
        )
        op.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1
                    FROM pg_indexes
                    WHERE tablename = 'rag_chunks'
                      AND (
                        indexdef ILIKE '%USING hnsw%embedding%'
                        OR indexdef ILIKE '%USING ivfflat%embedding%'
                      )
                ) THEN
                    BEGIN
                        CREATE INDEX ix_rag_chunks_embedding_hnsw
                        ON rag_chunks USING hnsw (embedding vector_cosine_ops);
                    EXCEPTION WHEN OTHERS THEN
                        -- HNSW may be unavailable; IVFFlat fallback
                        BEGIN
                            CREATE INDEX ix_rag_chunks_embedding_ivfflat
                            ON rag_chunks USING ivfflat (embedding vector_cosine_ops)
                            WITH (lists = 100);
                        EXCEPTION WHEN OTHERS THEN
                            RAISE EXCEPTION
                                'Unable to create an HNSW or IVFFlat ANN index on '
                                'rag_chunks.embedding';
                        END;
                    END;
                END IF;
            END $$;
            """
        )
        op.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'rag_chunks' AND column_name = 'content_tsv'
                ) THEN
                    ALTER TABLE rag_chunks ADD COLUMN content_tsv tsvector;
                END IF;
            END $$;
            """
        )
        op.execute(
            """
            UPDATE rag_chunks
            SET content_tsv = to_tsvector('simple', coalesce(content, ''))
            WHERE content_tsv IS NULL
            """
        )
        op.execute(
            """
            CREATE OR REPLACE FUNCTION rag_chunks_content_tsv_trigger()
            RETURNS trigger AS $$
            BEGIN
                NEW.content_tsv := to_tsvector('simple', coalesce(NEW.content, ''));
                RETURN NEW;
            END
            $$ LANGUAGE plpgsql;
            """
        )
        op.execute("DROP TRIGGER IF EXISTS trg_rag_chunks_content_tsv ON rag_chunks")
        op.execute(
            """
            CREATE TRIGGER trg_rag_chunks_content_tsv
            BEFORE INSERT OR UPDATE OF content ON rag_chunks
            FOR EACH ROW EXECUTE PROCEDURE rag_chunks_content_tsv_trigger();
            """
        )
        op.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_indexes WHERE indexname = 'ix_rag_chunks_content_tsv'
                ) THEN
                    CREATE INDEX ix_rag_chunks_content_tsv ON rag_chunks USING gin (content_tsv);
                END IF;
            END $$;
            """
        )

    op.add_column(
        "rag_chunks",
        sa.Column("content_hash", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "rag_chunks",
        sa.Column("parent_chunk_id", sa.String(), nullable=True),
    )
    op.add_column(
        "rag_chunks",
        sa.Column("chunker_version", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "rag_chunks",
        sa.Column("parser_version", sa.String(length=64), nullable=True),
    )
    op.create_index("ix_rag_chunks_parent_chunk_id", "rag_chunks", ["parent_chunk_id"])
    op.create_index("ix_rag_chunks_content_hash", "rag_chunks", ["content_hash"])

    op.create_table(
        "rag_conversations",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("project_id", sa.String(length=128), nullable=True),
        sa.Column("organization_id", sa.String(length=128), nullable=True),
        sa.Column("title", sa.String(length=512), nullable=False, server_default="Untitled"),
        sa.Column("scope_snapshot_json", sa.Text(), nullable=True),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_rag_conversations_user_id", "rag_conversations", ["user_id"])
    op.create_index("ix_rag_conversations_project_id", "rag_conversations", ["project_id"])
    op.create_index("ix_rag_conversations_updated_at", "rag_conversations", ["updated_at"])

    op.create_table(
        "rag_retrieval_traces",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("project_id", sa.String(length=128), nullable=True),
        sa.Column("conversation_id", sa.String(), sa.ForeignKey("rag_conversations.id", ondelete="SET NULL"), nullable=True),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("intent", sa.String(length=64), nullable=True),
        sa.Column("scope_hash", sa.String(length=64), nullable=True),
        sa.Column("document_ids_json", sa.Text(), nullable=True),
        sa.Column("retrieved_chunks_json", sa.Text(), nullable=True),
        sa.Column("coverage_json", sa.Text(), nullable=True),
        sa.Column("config_json", sa.Text(), nullable=True),
        sa.Column("degraded", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("degradation_reason", sa.String(length=256), nullable=True),
        sa.Column("no_matches", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("injection_chunks_filtered", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("latency_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_rag_retrieval_traces_user_id", "rag_retrieval_traces", ["user_id"])
    op.create_index("ix_rag_retrieval_traces_conversation_id", "rag_retrieval_traces", ["conversation_id"])
    op.create_index("ix_rag_retrieval_traces_created_at", "rag_retrieval_traces", ["created_at"])

    op.create_table(
        "rag_messages",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "conversation_id",
            sa.String(),
            sa.ForeignKey("rag_conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("model_name", sa.String(length=128), nullable=True),
        sa.Column("prompt_template_id", sa.String(length=128), nullable=True),
        sa.Column("prompt_version_id", sa.String(length=128), nullable=True),
        sa.Column(
            "retrieval_trace_id",
            sa.String(),
            sa.ForeignKey("rag_retrieval_traces.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("citations_json", sa.Text(), nullable=True),
        sa.Column("claims_json", sa.Text(), nullable=True),
        sa.Column("metadata_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_rag_messages_conversation_id", "rag_messages", ["conversation_id"])
    op.create_index("ix_rag_messages_retrieval_trace_id", "rag_messages", ["retrieval_trace_id"])
    op.create_index("ix_rag_messages_created_at", "rag_messages", ["created_at"])

    op.create_table(
        "research_assistant_threads",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "corpus_id",
            sa.String(),
            sa.ForeignKey("research_corpora.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("project_id", sa.String(), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column(
            "rag_conversation_id",
            sa.String(),
            sa.ForeignKey("rag_conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("title", sa.String(length=512), nullable=False, server_default="Ask Corpus"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_research_assistant_threads_corpus_id", "research_assistant_threads", ["corpus_id"])
    op.create_index("ix_research_assistant_threads_project_id", "research_assistant_threads", ["project_id"])
    op.create_index(
        "ix_research_assistant_threads_rag_conversation_id",
        "research_assistant_threads",
        ["rag_conversation_id"],
        unique=True,
    )

    op.create_table(
        "research_assistant_scope_snapshots",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "thread_id",
            sa.String(),
            sa.ForeignKey("research_assistant_threads.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "rag_message_id",
            sa.String(),
            sa.ForeignKey("rag_messages.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "retrieval_trace_id",
            sa.String(),
            sa.ForeignKey("rag_retrieval_traces.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("corpus_id", sa.String(), nullable=False),
        sa.Column("project_id", sa.String(), nullable=False),
        sa.Column("scope_hash", sa.String(length=64), nullable=False),
        sa.Column("rag_document_ids_json", sa.Text(), nullable=False),
        sa.Column("corpus_document_ids_json", sa.Text(), nullable=False),
        sa.Column("indexed_rag_document_ids_json", sa.Text(), nullable=False),
        sa.Column("unavailable_rag_document_ids_json", sa.Text(), nullable=False),
        sa.Column("index_version", sa.String(length=64), nullable=True),
        sa.Column("retrieval_version", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_research_assistant_scope_snapshots_thread_id",
        "research_assistant_scope_snapshots",
        ["thread_id"],
    )
    op.create_index(
        "ix_research_assistant_scope_snapshots_retrieval_trace_id",
        "research_assistant_scope_snapshots",
        ["retrieval_trace_id"],
    )


def downgrade() -> None:
    op.drop_table("research_assistant_scope_snapshots")
    op.drop_table("research_assistant_threads")
    op.drop_table("rag_messages")
    op.drop_table("rag_retrieval_traces")
    op.drop_table("rag_conversations")
    op.drop_index("ix_rag_chunks_content_hash", table_name="rag_chunks")
    op.drop_index("ix_rag_chunks_parent_chunk_id", table_name="rag_chunks")
    op.drop_column("rag_chunks", "parser_version")
    op.drop_column("rag_chunks", "chunker_version")
    op.drop_column("rag_chunks", "parent_chunk_id")
    op.drop_column("rag_chunks", "content_hash")

    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("DROP TRIGGER IF EXISTS trg_rag_chunks_content_tsv ON rag_chunks")
        op.execute("DROP FUNCTION IF EXISTS rag_chunks_content_tsv_trigger()")
        op.execute("DROP INDEX IF EXISTS ix_rag_chunks_content_tsv")
        op.execute("ALTER TABLE rag_chunks DROP COLUMN IF EXISTS content_tsv")
        # leave embedding column / indexes in place — may have been pre-existing
