"""用户管理（仅系统管理员 CRUD；管理员可查看列表）与系统配置、操作日志。"""
from __future__ import annotations
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from typing import Optional
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import get_db, get_security_config, set_security_config
from ..models import OperationLog, User
from ..security import (CurrentUser, ROLE_NAMES, hash_password, require_admin,
                        require_login, require_roles)
from ..services.log_service import client_ip, write_log
from ..utils.common import SECRET_LEVELS, apply_sort, paginate, to_dict

router = APIRouter(prefix="/api", tags=["system"])

DEFAULT_PASSWORD = "admin"  # 初始/重置密码（仅后端使用，前端不展示）


class UserBody(BaseModel):
    username: Optional[str] = None
    display_name: str = ""
    role: str = "visitor"
    department: str = ""
    phone: str = ""
    email: str = ""
    password: Optional[str] = None
    view_level: Optional[int] = None      # 0-5：可见最高密级
    operate_level: Optional[int] = None   # -1..5：可操作最高密级（-1=不可操作，须≤view_level）


def _perm_levels(body: UserBody, role: str) -> tuple[int, int]:
    """计算密级授权级别：管理员固定最高；访客默认 可见=公开(0)/操作=无(-1)；
    强制 operate_level ≤ view_level（确认点a）。"""
    if role in ("system_admin", "admin"):
        return 5, 5
    vl = 0 if body.view_level is None else int(body.view_level)
    ol = -1 if body.operate_level is None else int(body.operate_level)
    if not (0 <= vl <= 5):
        raise HTTPException(status_code=400, detail="可见密级须在 0-5 之间")
    if not (-1 <= ol <= 5):
        raise HTTPException(status_code=400, detail="可操作密级须在 -1~5 之间")
    if ol > vl:
        raise HTTPException(status_code=400, detail="可操作密级不能高于可见密级")
    return vl, ol


# ---------------- 用户管理 ----------------
@router.get("/users")
def list_users(page: int = 1, page_size: int = 20, search: str = "",
               sort: str = "", cur: CurrentUser = Depends(require_roles("system_admin", "admin")),
               db: Session = Depends(get_db)):
    q = db.query(User).filter(User.deleted_at.is_(None))
    if search:
        like = f"%{search}%"
        q = q.filter(User.username.like(like) | User.display_name.like(like))
    q = apply_sort(q, User, sort)
    result = paginate(q, page, page_size)
    result["items"] = [{**to_dict(u), "role_name": ROLE_NAMES.get(u.role, u.role)} for u in result["items"]]
    return result


@router.post("/users")
def create_user(body: UserBody, request: Request,
                cur: CurrentUser = Depends(require_admin), db: Session = Depends(get_db)):
    if not body.username:
        raise HTTPException(status_code=400, detail="用户名不能为空")
    if body.role not in ROLE_NAMES:
        raise HTTPException(status_code=400, detail="角色无效")
    if db.query(User).filter(User.username == body.username, User.deleted_at.is_(None)).first():
        raise HTTPException(status_code=400, detail="用户名已存在")
    view_level, operate_level = _perm_levels(body, body.role)
    # 新增用户自动分配默认密码并标记强制首登改密
    user = User(username=body.username, display_name=body.display_name or body.username,
                role=body.role, department=body.department, phone=body.phone, email=body.email,
                password_hash=hash_password(body.password or DEFAULT_PASSWORD),
                must_change_password=True, status="active",
                view_level=view_level, operate_level=operate_level)
    db.add(user)
    db.commit()
    write_log(db, user_id=cur.id, username=cur.username, action="创建用户", module="user",
              detail=f"创建用户 {body.username}（{ROLE_NAMES.get(body.role)}）", ip=client_ip(request))
    return {**to_dict(user), "role_name": ROLE_NAMES.get(user.role)}


@router.get("/users/{uid}")
def get_user(uid: int, cur: CurrentUser = Depends(require_roles("system_admin", "admin")),
             db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == uid, User.deleted_at.is_(None)).first()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    return {**to_dict(user), "role_name": ROLE_NAMES.get(user.role, user.role)}


@router.put("/users/{uid}")
def update_user(uid: int, body: UserBody, request: Request,
                cur: CurrentUser = Depends(require_admin), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == uid, User.deleted_at.is_(None)).first()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    if body.role and body.role not in ROLE_NAMES:
        raise HTTPException(status_code=400, detail="角色无效")
    for field in ("display_name", "department", "phone", "email"):
        setattr(user, field, getattr(body, field))
    if body.role:
        user.role = body.role
    if body.view_level is not None or body.operate_level is not None:
        vl, ol = _perm_levels(body, body.role or user.role)
        user.view_level, user.operate_level = vl, ol
        if user.role == "visitor":  # 访客密级授权变更记日志
            write_log(db, user_id=cur.id, username=cur.username, action="密级授权", module="user",
                      detail=f"用户 {user.username} 可见密级≤{SECRET_LEVELS[vl]}，可操作密级"
                             + (f"≤{SECRET_LEVELS[ol]}" if ol >= 0 else "：无"),
                      ip=client_ip(request))
    user.updated_at = datetime.now()
    db.commit()
    write_log(db, user_id=cur.id, username=cur.username, action="编辑用户", module="user",
              detail=f"修改用户 {user.username} 信息", ip=client_ip(request))
    return {**to_dict(user), "role_name": ROLE_NAMES.get(user.role, user.role)}


@router.delete("/users/{uid}")
def delete_user(uid: int, request: Request,
                cur: CurrentUser = Depends(require_admin), db: Session = Depends(get_db)):
    if uid == cur.id:
        raise HTTPException(status_code=400, detail="不能删除当前登录账号")
    user = db.query(User).filter(User.id == uid, User.deleted_at.is_(None)).first()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    user.deleted_at = datetime.now()  # 软删除
    db.commit()
    write_log(db, user_id=cur.id, username=cur.username, action="删除用户", module="user",
              detail=f"软删除用户 {user.username}", ip=client_ip(request))
    return {"message": "已删除"}


@router.put("/users/{uid}/password")
def reset_password(uid: int, request: Request,
                   cur: CurrentUser = Depends(require_admin), db: Session = Depends(get_db)):
    """重置密码：重置为默认密码并重新置 must_change_password=1 强制下次登录修改。"""
    user = db.query(User).filter(User.id == uid, User.deleted_at.is_(None)).first()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    user.password_hash = hash_password(DEFAULT_PASSWORD)
    user.must_change_password = True
    user.token_version = (user.token_version or 0) + 1  # 使旧会话失效
    db.commit()
    write_log(db, user_id=cur.id, username=cur.username, action="重置密码", module="user",
              detail=f"重置用户 {user.username} 的密码", ip=client_ip(request))
    return {"message": "密码已重置，该用户下次登录须修改密码"}


@router.put("/users/{uid}/status")
def set_status(uid: int, request: Request, body: dict,
               cur: CurrentUser = Depends(require_admin), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == uid, User.deleted_at.is_(None)).first()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    status = body.get("status")
    if status not in ("active", "disabled"):
        raise HTTPException(status_code=400, detail="状态无效")
    user.status = status
    if status == "disabled":
        user.token_version = (user.token_version or 0) + 1
    db.commit()
    write_log(db, user_id=cur.id, username=cur.username,
              action="启用用户" if status == "active" else "禁用用户", module="user",
              detail=f"用户 {user.username} 状态变更为 {status}", ip=client_ip(request))
    return {"message": "已更新", "status": status}


# ---------------- 系统配置（仅系统管理员） ----------------
@router.get("/settings")
def get_settings(cur: CurrentUser = Depends(require_admin), db: Session = Depends(get_db)):
    return get_security_config(db)


@router.put("/settings")
def put_settings(body: dict, request: Request,
                 cur: CurrentUser = Depends(require_admin), db: Session = Depends(get_db)):
    allowed = {"max_login_attempts", "lockout_duration_minutes",
               "session_timeout_minutes", "allow_concurrent_sessions"}
    values = {k: v for k, v in body.items() if k in allowed}
    if not values:
        raise HTTPException(status_code=400, detail="无可更新配置项")
    for k in ("max_login_attempts", "lockout_duration_minutes", "session_timeout_minutes"):
        if k in values and (not isinstance(values[k], int) or values[k] <= 0):
            raise HTTPException(status_code=400, detail=f"{k} 必须为正整数")
    set_security_config(db, values)
    write_log(db, user_id=cur.id, username=cur.username, action="修改系统配置", module="settings",
              detail=f"更新安全策略：{values}", ip=client_ip(request))
    return get_security_config(db)


# ---------------- 操作日志（不可删除，查询+导出CSV） ----------------
@router.get("/logs")
def list_logs(page: int = 1, page_size: int = 20, module: str = "", action: str = "",
              username: str = "", sort: str = "",
              cur: CurrentUser = Depends(require_roles("system_admin", "admin")),
              db: Session = Depends(get_db)):
    q = db.query(OperationLog)
    if module:
        q = q.filter(OperationLog.module == module)
    if action:
        q = q.filter(OperationLog.action.like(f"%{action}%"))
    if username:
        q = q.filter(OperationLog.username.like(f"%{username}%"))
    q = apply_sort(q, OperationLog, sort)
    result = paginate(q, page, page_size)
    result["items"] = [to_dict(x) for x in result["items"]]
    return result


@router.get("/logs/export")
def export_logs(module: str = "", username: str = "",
                cur: CurrentUser = Depends(require_roles("system_admin", "admin")),
                db: Session = Depends(get_db)):
    import csv
    import io
    q = db.query(OperationLog).order_by(OperationLog.created_at.desc())
    if module:
        q = q.filter(OperationLog.module == module)
    if username:
        q = q.filter(OperationLog.username.like(f"%{username}%"))
    buf = io.StringIO()
    buf.write("﻿")  # BOM，Excel 兼容
    writer = csv.writer(buf)
    writer.writerow(["ID", "用户ID", "用户名", "操作", "模块", "详情", "IP", "时间"])
    for log in q.limit(10000).all():
        writer.writerow([log.id, log.user_id, log.username, log.action, log.module,
                         log.detail, log.ip_address,
                         log.created_at.strftime("%Y-%m-%d %H:%M:%S") if log.created_at else ""])
    write_log(db, user_id=cur.id, username=cur.username, action="导出日志", module="logs",
              detail="导出操作日志 CSV")
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=operation_logs.csv"})
