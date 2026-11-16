"""name, explain, group and order contexts per category

Revision ID: b7c41e2d9a03
Revises: 628ae829970a
Create Date: 2026-11-16 17:40:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b7c41e2d9a03"
down_revision: Union[str, Sequence[str], None] = "628ae829970a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "category_context_presentation",
        sa.Column("category_key", sa.Text(), nullable=False),
        sa.Column("context_key", sa.Text(), nullable=False),
        sa.Column("theme", sa.Text(), nullable=False),
        sa.Column("sort_order", sa.Integer(), nullable=False),
        sa.Column("display_name", sa.Text(), nullable=True),
        sa.Column("definition_text", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["category_key"],
            ["category.key"],
            name=op.f("fk_category_context_presentation_category_key_category"),
        ),
        sa.ForeignKeyConstraint(
            ["context_key"],
            ["context.key"],
            name=op.f("fk_category_context_presentation_context_key_context"),
        ),
        sa.PrimaryKeyConstraint(
            "category_key", "context_key", name=op.f("pk_category_context_presentation")
        ),
        sqlite_strict=True,
    )


def downgrade() -> None:
    op.drop_table("category_context_presentation")
