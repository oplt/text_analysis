"""Persist campaign provenance on frozen training datasets.

Revision ID: d3e4f5a6b7c8
Revises: d2e3f4a5b6c7
"""

from alembic import op
import sqlalchemy as sa

revision = "d3e4f5a6b7c8"
down_revision = "d2e3f4a5b6c7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("research_training_dataset_snapshots", sa.Column("annotation_campaign_id", sa.String(), nullable=True))
    op.add_column("research_training_dataset_snapshots", sa.Column("annotation_campaign_snapshot_hash", sa.String(length=64), nullable=True))
    op.add_column("research_training_dataset_snapshots", sa.Column("adjudication_policy", sa.String(length=64), nullable=True))
    op.add_column("research_training_dataset_snapshots", sa.Column("gold_source", sa.String(length=64), nullable=True))
    op.create_foreign_key("fk_training_snapshot_campaign", "research_training_dataset_snapshots", "research_annotation_campaigns", ["annotation_campaign_id"], ["id"], ondelete="SET NULL")
    op.create_index("ix_training_snapshot_campaign_id", "research_training_dataset_snapshots", ["annotation_campaign_id"])
    op.create_index("ix_training_snapshot_campaign_hash", "research_training_dataset_snapshots", ["annotation_campaign_snapshot_hash"])


def downgrade() -> None:
    op.drop_index("ix_training_snapshot_campaign_hash", table_name="research_training_dataset_snapshots")
    op.drop_index("ix_training_snapshot_campaign_id", table_name="research_training_dataset_snapshots")
    op.drop_constraint("fk_training_snapshot_campaign", "research_training_dataset_snapshots", type_="foreignkey")
    op.drop_column("research_training_dataset_snapshots", "gold_source")
    op.drop_column("research_training_dataset_snapshots", "adjudication_policy")
    op.drop_column("research_training_dataset_snapshots", "annotation_campaign_snapshot_hash")
    op.drop_column("research_training_dataset_snapshots", "annotation_campaign_id")
