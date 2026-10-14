"""认证与安全：bcrypt 哈希（成本因子12）、JWT（python-jose）、HttpOnly Cookie、
会话超时（last_activity）、并发会话（token_version）、角色权限依赖。
"""
from __future__ import annotations

from datetime import datetime, timedelta

import bcrypt as _bcrypt
from fastapi import Depends, HTTPException, Request, Response
from fastapi.security.utils import get_authorization_scheme_param
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from .config import settings
from .database import get_db, get_security_config
from .models import User

ALGORITHM = "HS256"
ACCESS_TOKEN_MINUTES = 120        # Access Token 2小时
REFRESH_TOKEN_DAYS = 7            # Refresh Token 7天
ACCESS_COOKIE = "dp_access_token"
REFRESH_COOKIE = "dp_refresh_token"
BCRYPT_ROUNDS = 12

# bcrypt 成本因子12（passlib 与新版 bcrypt 存在兼容问题，直接使用 bcrypt 库）
def hash_password(plain: str) -> str:
    return _bcrypt.hashpw(plain.encode("utf-8"), _bcrypt.gensalt(rounds=BCRYPT_ROUNDS)).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return _bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        return False


def create_token(user: User, token_type: str = "access") -> str:
    expire = datetime.utcnow() + (
        timedelta(minutes=ACCESS_TOKEN_MINUTES) if token_type == "access"
        else timedelta(days=REFRESH_TOKEN_DAYS))
    payload = {
        "sub": str(user.id),
        "username": user.username,
        "role": user.role,
        "tv": user.token_version,
        "type": token_type,
        "exp": expire,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    return jwt.decode(token, settings.jwt_secret, algorithms=[ALGORITHM])


def set_auth_cookies(response: Response, user: User) -> None:
    """Token 存 HttpOnly Cookie，SameSite=Lax（localhost 无 HTTPS，不挂 Secure）。"""
    response.set_cookie(ACCESS_COOKIE, create_token(user, "access"),
                        max_age=ACCESS_TOKEN_MINUTES * 60,
                        httponly=True, samesite="lax")
    response.set_cookie(REFRESH_COOKIE, create_token(user, "refresh"),
                        max_age=REFRESH_TOKEN_DAYS * 86400,
                        httponly=True, samesite="lax")


def clear_auth_cookies(response: Response) -> None:
    response.delete_cookie(ACCESS_COOKIE)
    response.delete_cookie(REFRESH_COOKIE)


class CurrentUser:
    """解析后的当前用户上下文"""
    def __init__(self, user: User):
        self.user = user
        self.id = user.id
        self.username = user.username
        self.role = user.role
        self.display_name = user.display_name
        self.view_level = user.view_level if user.view_level is not None else 0
        self.operate_level = user.operate_level if user.operate_level is not None else -1

    # ---- 密级权限：system_admin/admin 不受限；visitor 按授权级别 ----
    @property
    def unrestricted(self) -> bool:
        return self.role in ("system_admin", "admin")

    def can_view(self, secret_level: str) -> bool:
        from .utils.common import secret_index
        return self.unrestricted or secret_index(secret_level) <= self.view_level

    def can_operate(self, secret_level: str) -> bool:
        from .utils.common import secret_index
        return self.unrestricted or secret_index(secret_level) <= self.operate_level

    def require_view(self, secret_level: str) -> None:
        if not self.can_view(secret_level):
            raise HTTPException(status_code=403,
                                detail=f"无权访问「{secret_level}」密级数据（您的可见密级上限不足）")

    def require_operate(self, secret_level: str) -> None:
        if not self.can_operate(secret_level):
            raise HTTPException(status_code=403,
                                detail=f"无权操作「{secret_level}」密级数据（您的可操作密级上限不足）")


def _extract_token(request: Request) -> str:
    token = request.cookies.get(ACCESS_COOKIE)
    if not token:
        auth = request.headers.get("Authorization", "")
        scheme, param = get_authorization_scheme_param(auth)
        if scheme.lower() == "bearer":
            token = param
    if not token:
        raise HTTPException(status_code=401, detail="未登录或会话已失效")
    return token


def get_current_user(request: Request, db: Session = Depends(get_db)) -> CurrentUser:
    token = _extract_token(request)
    try:
        payload = decode_token(token)
    except JWTError:
        raise HTTPException(status_code=401, detail="登录凭证无效或已过期")
    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="凭证类型无效")
    user = db.query(User).filter(User.id == int(payload["sub"]),
                                 User.deleted_at.is_(None)).first()
    if not user or user.status != "active":
        raise HTTPException(status_code=401, detail="账号不存在或已禁用")
    # 并发会话关闭时：token_version 不一致即被挤下线
    if payload.get("tv") != user.token_version:
        raise HTTPException(status_code=401, detail="账号已在其他位置登录，当前会话已失效")
    # 会话超时：last_activity 超过 session_timeout_minutes 无操作即吊销会话
    sec = get_security_config(db)
    now = datetime.now()
    if user.last_activity_at and (now - user.last_activity_at).total_seconds() > sec["session_timeout_minutes"] * 60:
        raise HTTPException(status_code=401, detail="会话超时，请重新登录")
    user.last_activity_at = now
    db.commit()
    return CurrentUser(user)


ROLE_NAMES = {"system_admin": "系统管理员", "admin": "管理员", "visitor": "访客"}

# 角色权限矩阵（9.1）：module -> 允许的角色集合
_WRITE_ROLES = {"system_admin", "admin"}   # 数据/预处理/标准语义的写权限


def require_roles(*roles: str):
    def dep(cur: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if cur.role not in roles:
            raise HTTPException(status_code=403, detail="无权执行该操作")
        return cur
    return dep


require_admin = require_roles("system_admin")          # 仅系统管理员
require_write = require_roles("system_admin", "admin")  # 系统管理员+管理员
require_login = get_current_user                        # 任意登录用户


def require_task_user(cur: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    """预处理任务创建权限：admin 不受限；visitor 须被授予过可操作密级（operate_level≥0），
    具体密级是否够格由各接口按数据集密级再校验。"""
    if not cur.unrestricted and cur.operate_level < 0:
        raise HTTPException(status_code=403, detail="您未被授予数据操作权限，请联系系统管理员")
    return cur
