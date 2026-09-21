"""declare which indicators a control case may sweep

Adds ``control_mode`` and ``control_note`` to category membership.

A control case asserts that an indicator's declared direction holds across its
whole declared range. That is true for most indicators and false for some: the
relationship turns over, the sources disagree, or the question is still open.
Sweeping one of those would put a claim into the training data that the registry
does not actually make, so the registry now says which is which, and the
behavioural suite tests only what it can legitimately test.

Existing rows default to ``sweep``, which is the prior behaviour. Excluding an
indicator requires writing down why, enforced by a CHECK.

Revision ID: 628ae829970a
Revises: 56a19066e66e
Create Date: 2026-09-21 15:24:02.066534
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "628ae829970a"
down_revision: Union[str, Sequence[str], None] = "56a19066e66e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("category_indicator", schema=None) as batch_op:
        # SQLite will not add a NOT NULL column without a default to fill the
        # rows that already exist, so the default is part of the DDL rather
        # than only a Python-side default.
        batch_op.add_column(
            sa.Column(
                "control_mode",
                sa.Text(),
                nullable=False,
                server_default="sweep",
            )
        )
        batch_op.add_column(sa.Column("control_note", sa.Text(), nullable=True))
        batch_op.create_check_constraint(
            "ck_category_indicator_control_mode",
            "control_mode IN ('sweep','exclude')",
        )
        batch_op.create_check_constraint(
            "ck_category_indicator_exclusion_justified",
            "control_mode <> 'exclude' OR control_note IS NOT NULL",
        )


def downgrade() -> None:
    with op.batch_alter_table("category_indicator", schema=None) as batch_op:
        batch_op.drop_constraint(
            "ck_category_indicator_exclusion_justified", type_="check"
        )
        batch_op.drop_constraint("ck_category_indicator_control_mode", type_="check")
        batch_op.drop_column("control_note")
        batch_op.drop_column("control_mode")
