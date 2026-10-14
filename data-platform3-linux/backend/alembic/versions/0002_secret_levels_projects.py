"""v1.4.0：密级六级化 + 用户密级授权 + 项目表 + 自定义评估标准

新增列：
- users.view_level / operate_level（密级授权，访客默认 0/-1）
- datasets.secret_level（数据集密级，默认 内部）
- ref_standards.rules / is_custom（自定义评估规则）
新增表：projects（项目真实落库）
应用启动时 init_schema() 的 _migrate_columns() 与 create_all 已做幂等处理，
本脚本供使用 alembic upgrade 管理迁移的部署方式使用。

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-30
"""
from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("view_level", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("users", sa.Column("operate_level", sa.Integer(), nullable=False, server_default="-1"))
    op.add_column("datasets", sa.Column("secret_level", sa.String(10), nullable=False, server_default="内部"))
    op.add_column("ref_standards", sa.Column("rules", sa.Text(), nullable=True))
    op.add_column("ref_standards", sa.Column("is_custom", sa.Boolean(), nullable=False, server_default="0"))
    op.create_table(
        "projects",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(100), unique=True, nullable=False),
        sa.Column("description", sa.String(500), server_default=""),
        sa.Column("created_by", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    # 存量账号：管理员角色授权拉满（代码中管理员本就不受限，字段仅作展示）
    op.execute("UPDATE users SET view_level = 5, operate_level = 5 WHERE role IN ('system_admin','admin')")


def downgrade() -> None:
    op.drop_table("projects")
    op.drop_column("ref_standards", "is_custom")
    op.drop_column("ref_standards", "rules")
    op.drop_column("datasets", "secret_level")
    op.drop_column("users", "operate_level")
    op.drop_column("users", "view_level")
