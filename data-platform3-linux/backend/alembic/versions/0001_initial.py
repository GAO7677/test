"""初始版本基线（v1.3.0）

首次部署由应用启动时 create_all + schema_version 自动建表；
本脚本作为 Alembic 迁移基线，后续结构变更在此基础上追加 revision。

Revision ID: 0001
Revises:
Create Date: 2026-07-28
"""
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 基线：标记 schema_version；表结构由应用 create_all 管理
    op.execute(
        "INSERT INTO schema_version (version, applied_at) "
        "SELECT '1.3.0', NOW() FROM DUAL "
        "WHERE NOT EXISTS (SELECT 1 FROM schema_version WHERE version = '1.3.0')"
    )


def downgrade() -> None:
    pass
