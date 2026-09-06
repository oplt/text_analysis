"""Record the codebook version used for each adjudication."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "f2a6c1d4e9b0"
down_revision = "e1f2a3b4c5d6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "research_adjudications",
        sa.Column("codebook_version", sa.String(length=64), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("research_adjudications", "codebook_version")
