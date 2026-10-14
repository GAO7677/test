"""数据库连接与初始化：自动建库（utf8mb4）+ create_all + schema_version 标记 + 种子数据。

默认使用 config.json 中的 MySQL 配置；开发联调可用环境变量 DATABASE_URL 覆盖（如 sqlite）。
"""
from __future__ import annotations

import os
from datetime import datetime

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from .config import settings
from .models import Base

SCHEMA_VERSION = "3.0.0"

# 轻量列迁移：存量库升级时补齐新增列（create_all 不会修改已存在的表）
_COLUMN_MIGRATIONS = [
    ("fewshot_jobs", "download_request_id", "INT NULL"),
    ("users", "view_level", "INT NOT NULL DEFAULT 0"),
    ("users", "operate_level", "INT NOT NULL DEFAULT -1"),
    ("datasets", "secret_level", "VARCHAR(10) NOT NULL DEFAULT '内部'"),
    ("ref_standards", "rules", "TEXT"),
    ("ref_standards", "is_custom", "TINYINT(1) NOT NULL DEFAULT 0"),
]


def _migrate_columns() -> None:
    """检查并补齐历史库缺失的列（幂等，兼容 MySQL 与 SQLite）。"""
    from sqlalchemy import inspect
    insp = inspect(engine)
    is_sqlite = engine.dialect.name == "sqlite"
    with engine.begin() as conn:
        for table, column, ddl in _COLUMN_MIGRATIONS:
            if not insp.has_table(table):
                continue
            cols = {c["name"] for c in insp.get_columns(table)}
            if column in cols:
                continue
            if is_sqlite and ddl.startswith("TINYINT(1)"):
                ddl = "BOOLEAN NOT NULL DEFAULT 0"
            conn.execute(text(f"ALTER TABLE `{table}` ADD COLUMN `{column}` {ddl}"))
    from .models import FewshotJob
    for index in FewshotJob.__table__.indexes:
        if index.name == 'ix_fewshot_jobs_download_request_id':
            index.create(engine,checkfirst=True)


def _mysql_url(db: dict, with_db: bool = True) -> str:
    base = (f"mysql+pymysql://{db['username']}:{db['password']}"
            f"@{db['host']}:{db['port']}")
    return f"{base}/{db['dbname']}?charset=utf8mb4" if with_db else base


def _build_engine():
    override = os.environ.get("DATABASE_URL")
    if override:
        return create_engine(override, pool_pre_ping=True)
    db = settings.db
    # 数据库不存在则自动创建（utf8mb4 / utf8mb4_general_ci）
    server_engine = create_engine(_mysql_url(db, with_db=False), pool_pre_ping=True)
    with server_engine.connect() as conn:
        conn.execute(text(
            f"CREATE DATABASE IF NOT EXISTS `{db['dbname']}` "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci"))
        conn.commit()
    server_engine.dispose()
    return create_engine(_mysql_url(db), pool_pre_ping=True, pool_recycle=3600)


engine = _build_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_schema() -> None:
    """create_all 建表 + schema_version 标记。"""
    Base.metadata.create_all(engine)
    _migrate_columns()
    from .models import SchemaVersion
    with SessionLocal() as s:
        row = s.query(SchemaVersion).order_by(SchemaVersion.id.desc()).first()
        if not row or row.version != SCHEMA_VERSION:
            s.add(SchemaVersion(version=SCHEMA_VERSION))
            s.commit()


def seed_data() -> None:
    """初始化默认数据（幂等）：默认账号、5 项参考标准、默认标注模板、术语库、
    system_settings 默认值（导入 config.json security 段）。"""
    from .models import (User, RefStandard, Template, SemanticTerm, SystemSetting)
    from .security import hash_password
    import json

    with SessionLocal() as s:
        # ---- 默认账号（初始密码 admin，标记 must_change_password=1 强制首登改密）----
        if s.query(User).count() == 0:
            for username, display, role in (
                ("sysadmin", "系统管理员", "system_admin"),
                ("admin", "管理员", "admin"),
                ("guest", "访客", "visitor"),
            ):
                is_admin = role in ("system_admin", "admin")
                s.add(User(username=username, password_hash=hash_password("admin"),
                           display_name=display, role=role, status="active",
                           view_level=5 if is_admin else 0,
                           operate_level=5 if is_admin else -1,
                           must_change_password=True))
            s.commit()
        else:
            # 幂等修正：历史库中的管理员账号密级授权置为最高（管理员本身不受限，仅保证数据一致）
            for u in s.query(User).filter(User.role.in_(("system_admin", "admin"))).all():
                if (u.view_level or 0) < 5 or (u.operate_level or -1) < 5:
                    u.view_level, u.operate_level = 5, 5
            s.commit()

        # ---- system_settings（导入 config.json security 段作为初始值）----
        sec = settings.security
        defaults = [
            ("max_login_attempts", str(sec["max_login_attempts"]), "连续登录失败锁定阈值（次）"),
            ("lockout_duration_minutes", str(sec["lockout_duration_minutes"]), "账号锁定时长（分钟）"),
            ("session_timeout_minutes", str(sec["session_timeout_minutes"]), "会话超时时间（分钟）"),
            ("allow_concurrent_sessions", "true" if sec["allow_concurrent_sessions"] else "false", "是否允许并发会话"),
        ]
        for key, value, desc in defaults:
            if not s.query(SystemSetting).filter_by(key=key).first():
                s.add(SystemSetting(key=key, value=value, description=desc))
        s.commit()

        # ---- 5 项内置参考标准 ----
        if s.query(RefStandard).count() == 0:
            for code, name, dims, scope in (
                ("GB/T 36344-2018", "信息技术 数据质量评价指标",
                 "规范性、完整性、准确性、一致性、时效性、可访问性", "通用数据质量评价"),
                ("GB/T 42381.61-2023", "数据质量 第61部分：数据质量管理：过程参考模型",
                 "过程参考模型、质量规划、质量保证、质量控制", "数据质量管理过程"),
                ("GB/T 36073-2018", "数据管理能力成熟度评估模型 (DCMM)",
                 "数据战略、数据治理、数据质量、数据标准", "数据管理能力评估"),
                ("ISO 8000-61:2016", "Data quality — Data quality management: Process reference model",
                 "Data quality planning, assurance, control", "国际数据质量管理"),
                ("GB/T 44406-2024", "生产过程质量数据采集系统 性能评估与校准",
                 "采集精度、稳定性、校准、性能评估", "质量数据采集系统"),
            ):
                s.add(RefStandard(code=code, name=name, standard_type="国家标准" if code.startswith("GB") else "国际标准",
                                  dimensions=dims, scope=scope))
            s.commit()

        # ---- 默认标注模板 ----
        if s.query(Template).count() == 0:
            for name, dtype, desc, fields in (
                ("语音固定切片", "voice", "按固定时长切片",
                 {"params": {"sliceDuration": 10, "overlap": 0.5}, "applicable": ["WAV", "MP3", "FLAC"]}),
                ("雷达固定网格", "radar", "固定网格划分",
                 {"params": {"gridSize": 64, "rangeMax": 500}, "applicable": ["CSV", "PLY"]}),
                ("图像均匀网格", "image", "均匀网格标注",
                 {"params": {"gridRows": 8, "gridCols": 8}, "applicable": ["JPG", "PNG", "TIFF"]}),
            ):
                s.add(Template(name=name, data_type=dtype, description=desc,
                               fields=json.dumps(fields, ensure_ascii=False), is_default=True))
            s.commit()

        # ---- 航空领域术语库 ----
        if s.query(SemanticTerm).count() == 0:
            terms = [
                ("迎角", "Angle of Attack (AOA)", "机翼弦线与相对气流方向之间的夹角", ["攻角"], "°", "飞行动力学", "内部", "现行"),
                ("侧滑角", "Sideslip Angle (β)", "飞机对称面与相对气流方向之间的夹角", ["偏流角"], "°", "飞行动力学", "内部", "现行"),
                ("马赫数", "Mach Number (Ma)", "飞行速度与当地音速之比", ["M数", "马赫"], "—", "飞行性能", "内部", "现行"),
                ("雷达散射截面积", "Radar Cross Section (RCS)", "目标反射雷达波能力的度量", ["雷达反射截面", "雷达截面积"], "m²", "雷达特性", "秘密", "现行"),
                ("升力系数", "Lift Coefficient (CL)", "升力与动压和参考面积乘积之比的无量纲参数", [], "—", "空气动力学", "内部", "现行"),
                ("推重比", "Thrust-to-Weight Ratio", "发动机推力与飞机重力之比", ["推力重量比"], "—", "动力装置", "内部", "现行"),
                ("过载", "Load Factor (n)", "升力与飞机重力之比，表征机动载荷水平", ["载荷因数", "G值"], "g", "飞行动力学", "内部", "现行"),
                ("指示空速", "Indicated Airspeed (IAS)", "按海平面标准大气条件空速表直接读出的速度", ["表速"], "km/h", "飞行性能", "内部", "现行"),
                ("地速", "Ground Speed (GS)", "飞机相对于地面的运动速度", [], "km/h", "飞行性能", "内部", "现行"),
                ("信噪比", "Signal-to-Noise Ratio (SNR)", "信号功率与噪声功率之比", [], "dB", "数据质量", "内部", "现行"),
                ("点云", "Point Cloud", "三维空间中目标表面采样点的集合", ["点云数据"], "—", "雷达数据", "内部", "现行"),
                ("标注", "Annotation", "为数据样本附加类别、位置等标签信息的过程与结果", ["打标", "标定"], "—", "数据工程", "内部", "现行"),
                ("数据集", "Dataset", "具有统一元数据描述、可独立管理的数据集合", [], "—", "数据工程", "内部", "现行"),
                ("失速速度", "Stall Speed (Vs)", "飞机维持水平飞行的最小速度", ["失速点速度"], "km/h", "飞行性能", "内部", "待审核"),
            ]
            for t, en, d, syn, unit, cat, sl, st in terms:
                s.add(SemanticTerm(term=t, en_name=en, definition=d,
                                   synonyms=json.dumps(syn, ensure_ascii=False),
                                   unit=unit, category=cat, secret_level=sl, status=st))
            s.commit()


def get_setting(db: Session, key: str, default: str = "") -> str:
    from .models import SystemSetting
    row = db.query(SystemSetting).filter_by(key=key).first()
    return row.value if row else default


def get_security_config(db: Session) -> dict:
    """四项安全策略以 system_settings 表为准（决策：配置双写）。"""
    return {
        "max_login_attempts": int(get_setting(db, "max_login_attempts", "5")),
        "lockout_duration_minutes": int(get_setting(db, "lockout_duration_minutes", "30")),
        "session_timeout_minutes": int(get_setting(db, "session_timeout_minutes", "120")),
        "allow_concurrent_sessions": get_setting(db, "allow_concurrent_sessions", "true").lower() == "true",
    }


def set_security_config(db: Session, values: dict) -> None:
    from .models import SystemSetting
    for key in ("max_login_attempts", "lockout_duration_minutes",
                "session_timeout_minutes", "allow_concurrent_sessions"):
        if key not in values:
            continue
        v = values[key]
        v = "true" if v is True else "false" if v is False else str(v)
        row = db.query(SystemSetting).filter_by(key=key).first()
        if row:
            row.value = v
            row.updated_at = datetime.now()
        else:
            db.add(SystemSetting(key=key, value=v))
    db.commit()
