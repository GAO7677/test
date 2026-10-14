"""Alembic 环境配置：从 app.database 读取连接串（MySQL 8.0 / utf8mb4）。

用法：
    cd backend && alembic revision --autogenerate -m "变更说明" && alembic upgrade head
"""
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.database import _mysql_url  # noqa: E402
from app.config import settings       # noqa: E402
from app.models import Base           # noqa: E402

config = context.config
config.set_main_option("sqlalchemy.url", _mysql_url(settings.db))
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(url=config.get_main_option("sqlalchemy.url"),
                      target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(config.get_section(config.config_ini_section, {}),
                                     prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
