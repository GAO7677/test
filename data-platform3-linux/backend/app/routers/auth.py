"""认证：登录（仅账号+密码，角色由后端按账号决定）、登出、me、refresh、修改密码。
登录失败计数 + 锁定；并发会话关闭时 token_version+1 挤下线；首次登录强制改密。
"""
from __future__ import annotations
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import get_db, get_security_config
from ..models import User
from ..security import (CurrentUser, clear_auth_cookies, create_token,
                        decode_token, get_current_user, hash_password,
                        set_auth_cookies, verify_password, REFRESH_COOKIE,
                        ROLE_NAMES)
from ..services.log_service import client_ip, write_log

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginBody(BaseModel):
    username: str
    password: str


class ChangePasswordBody(BaseModel):
    old_password: str
    new_password: str


@router.post("/login")
def login(body: LoginBody, request: Request, response: Response, db: Session = Depends(get_db)):
    sec = get_security_config(db)
    user = db.query(User).filter(User.username == body.username,
                                 User.deleted_at.is_(None)).first()
    now = datetime.now()

    # 1) 锁定状态校验
    if user and user.locked_until and user.locked_until > now:
        minutes = int((user.locked_until - now).total_seconds() // 60) + 1
        raise HTTPException(status_code=423,
                            detail=f"账号已锁定，请 {minutes} 分钟后再试")

    # 2) 账号存在与禁用状态 → 3) 密码校验
    if not user or not verify_password(body.password, user.password_hash):
        if user:
            user.failed_attempts = (user.failed_attempts or 0) + 1
            remaining = sec["max_login_attempts"] - user.failed_attempts
            if user.failed_attempts >= sec["max_login_attempts"]:
                user.locked_until = now + timedelta(minutes=sec["lockout_duration_minutes"])
                user.failed_attempts = 0
                db.commit()
                write_log(db, user_id=user.id, username=user.username, action="登录锁定",
                          module="auth", detail=f"连续失败达 {sec['max_login_attempts']} 次，锁定 {sec['lockout_duration_minutes']} 分钟",
                          ip=client_ip(request))
                raise HTTPException(status_code=423,
                                    detail=f"连续登录失败次数过多，账号已锁定 {sec['lockout_duration_minutes']} 分钟")
            db.commit()
            write_log(db, user_id=user.id, username=user.username, action="登录失败",
                      module="auth", detail="密码错误", ip=client_ip(request))
            raise HTTPException(status_code=401,
                                detail=f"用户名或密码错误，剩余尝试次数 {remaining} 次")
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    if user.status != "active":
        raise HTTPException(status_code=403, detail="账号已被禁用，请联系系统管理员")

    # 登录成功：重置失败计数
    user.failed_attempts = 0
    user.locked_until = None
    # 并发会话关闭时：新登录使 token_version +1，旧 token 立即失效
    if not sec["allow_concurrent_sessions"]:
        user.token_version = (user.token_version or 0) + 1
    user.last_login_at = now
    user.last_activity_at = now
    db.commit()

    set_auth_cookies(response, user)
    write_log(db, user_id=user.id, username=user.username, action="登录成功",
              module="auth", detail=f"用户 {user.username} 登录成功", ip=client_ip(request))
    return {
        "id": user.id, "username": user.username, "display_name": user.display_name,
        "role": user.role, "role_name": ROLE_NAMES.get(user.role, user.role),
        "must_change_password": bool(user.must_change_password),
    }


@router.post("/logout")
def logout(request: Request, response: Response, db: Session = Depends(get_db)):
    token = request.cookies.get("dp_access_token")
    if token:
        try:
            payload = decode_token(token)
            write_log(db, user_id=int(payload["sub"]), username=payload.get("username", ""),
                      action="退出登录", module="auth", detail="用户退出系统", ip=client_ip(request))
        except Exception:
            pass
    clear_auth_cookies(response)
    return {"message": "已退出登录"}


@router.get("/me")
def me(cur: CurrentUser = Depends(get_current_user)):
    u = cur.user
    return {
        "id": u.id, "username": u.username, "display_name": u.display_name,
        "role": u.role, "role_name": ROLE_NAMES.get(u.role, u.role),
        "department": u.department, "phone": u.phone, "email": u.email,
        "view_level": u.view_level, "operate_level": u.operate_level,
        "must_change_password": bool(u.must_change_password),
        "last_login_at": u.last_login_at.strftime("%Y-%m-%d %H:%M:%S") if u.last_login_at else None,
    }


@router.post("/refresh")
def refresh(request: Request, response: Response, db: Session = Depends(get_db)):
    token = request.cookies.get(REFRESH_COOKIE)
    if not token:
        raise HTTPException(status_code=401, detail="无刷新凭证")
    try:
        payload = decode_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="刷新凭证无效或已过期")
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="凭证类型无效")
    user = db.query(User).filter(User.id == int(payload["sub"]),
                                 User.deleted_at.is_(None)).first()
    if not user or user.status != "active" or payload.get("tv") != user.token_version:
        raise HTTPException(status_code=401, detail="会话已失效，请重新登录")
    set_auth_cookies(response, user)
    return {"message": "ok"}


@router.post("/change-password")
def change_password(body: ChangePasswordBody, request: Request,
                    cur: CurrentUser = Depends(get_current_user),
                    db: Session = Depends(get_db)):
    """修改密码（验原密码；首次登录强制改密走此接口，成功后清除 must_change_password）。"""
    user = db.query(User).filter(User.id == cur.id).first()
    if not verify_password(body.old_password, user.password_hash):
        raise HTTPException(status_code=400, detail="原密码错误")
    if len(body.new_password) < 6:
        raise HTTPException(status_code=400, detail="新密码长度至少 6 位")
    if body.new_password == body.old_password:
        raise HTTPException(status_code=400, detail="新密码不得与当前密码相同")
    user.password_hash = hash_password(body.new_password)
    user.must_change_password = False
    db.commit()
    write_log(db, user_id=user.id, username=user.username, action="修改密码",
              module="auth", detail="用户修改密码成功", ip=client_ip(request))
    return {"message": "密码修改成功"}
