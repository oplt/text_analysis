"""Replace task assignment-derived project access with explicit memberships."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "b7e4c2d8f1a6"
down_revision = "c3d7a1e9b4f2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "project_members",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("project_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False, server_default="viewer"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project_id", "user_id", name="uq_project_members_project_user"),
    )
    op.create_index("ix_project_members_project_id", "project_members", ["project_id"])
    op.create_index("ix_project_members_user_id", "project_members", ["user_id"])
    op.execute(
        """
        INSERT INTO project_members (id, project_id, user_id, role, created_at)
        SELECT lower(hex(randomblob(16))), id, owner_id, 'owner', created_at
        FROM projects
        """
        if op.get_bind().dialect.name == "sqlite"
        else """
        INSERT INTO project_members (id, project_id, user_id, role, created_at)
        SELECT md5(random()::text || clock_timestamp()::text), id, owner_id, 'owner', created_at
        FROM projects
        """
    )
    op.execute(
        """
        INSERT INTO project_members (id, project_id, user_id, role, created_at)
        SELECT lower(hex(randomblob(16))), project_id, assignee_id, 'viewer', created_at
        FROM project_tasks
        WHERE assignee_id IS NOT NULL
        ON CONFLICT (project_id, user_id) DO NOTHING
        """
        if op.get_bind().dialect.name == "sqlite"
        else """
        INSERT INTO project_members (id, project_id, user_id, role, created_at)
        SELECT md5(random()::text || clock_timestamp()::text || project_id || assignee_id), project_id, assignee_id, 'viewer', created_at
        FROM project_tasks
        WHERE assignee_id IS NOT NULL
        ON CONFLICT (project_id, user_id) DO NOTHING
        """
    )


def downgrade() -> None:
    op.drop_index("ix_project_members_user_id", table_name="project_members")
    op.drop_index("ix_project_members_project_id", table_name="project_members")
    op.drop_table("project_members")
