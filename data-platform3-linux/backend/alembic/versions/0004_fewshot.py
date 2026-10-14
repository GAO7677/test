"""Add few-shot jobs; compatible with startup create_all."""
from alembic import op
revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade():
    from app.models.fewshot import FewshotJob
    FewshotJob.__table__.create(op.get_bind(), checkfirst=True)


def downgrade():
    from app.models.fewshot import FewshotJob
    FewshotJob.__table__.drop(op.get_bind(), checkfirst=True)
