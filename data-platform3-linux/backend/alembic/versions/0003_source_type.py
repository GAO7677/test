"""v3.2：数据获取层五表加数据来源字段 source_type（真实采集/仿真/AIGC）

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-17
"""
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

TABLES = ["voice_data", "radar_data", "image_data", "video_data", "text_data"]


def upgrade() -> None:
    for t in TABLES:
        op.execute(
            f"ALTER TABLE {t} ADD COLUMN source_type VARCHAR(20) "
            f"NOT NULL DEFAULT '真实采集' COMMENT '真实采集/仿真/AIGC'"
        )


def downgrade() -> None:
    for t in TABLES:
        op.execute(f"ALTER TABLE {t} DROP COLUMN source_type")
