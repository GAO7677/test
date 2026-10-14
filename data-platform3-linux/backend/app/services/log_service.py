"""操作日志服务：同时写入数据库 operation_logs 表和日志文件（不可删除）。"""
from __future__ import annotations
from typing import Optional

from sqlalchemy.orm import Session

from ..models import OperationLog
from ..utils.common import get_file_logger


def write_log(db: Session, *, user_id: Optional[int], username: str,
              action: str, module: str, detail: str = "", ip: str = "") -> None:
    entry = OperationLog(user_id=user_id, username=username, action=action,
                         module=module, detail=detail, ip_address=ip)
    db.add(entry)
    db.commit()
    get_file_logger().info(f"user={username}({user_id}) action={action} module={module} "
                           f"ip={ip} detail={detail}")


def client_ip(request) -> str:
    if request and request.client:
        return request.client.host
    return ""
