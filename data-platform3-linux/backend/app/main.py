"""FastAPI 入口：启动流程（第11章）+ uvicorn.Server 内嵌启动（适配 PyInstaller 打包）。

启动流程：
1. 读取 config.json（不存在生成默认；jwt_secret 占位自动生成随机密钥写回）
2. 清理 temp 目录 7 天前临时文件
3. 检测数据库连接 → 4. 自动建库（utf8mb4）→ 5. create_all + schema_version
6. 初始化默认数据（默认账号/5项标准/默认模板/术语/system_settings）
7. 启动服务：默认 8000，被占用自动顺延 8001、8002…并控制台提示
8. 自动打开浏览器访问 http://localhost:<实际端口>
"""
from __future__ import annotations
from typing import Optional

import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from .config import settings
from .database import init_schema, seed_data
from .utils.common import (clean_temp_files, http_exception_handler,
                           unhandled_exception_handler)


def create_app() -> FastAPI:
    app = FastAPI(title="数据中台", version="3.2.0",
                  docs_url=None, redoc_url=None, openapi_url=None)  # 离线环境关闭在线文档
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    @app.middleware('http')
    async def private_api_cache(request,call_next):
        response=await call_next(request)
        if request.url.path.startswith('/api/'):
            response.headers['Cache-Control']='private, no-store'
        return response

    from .routers import auth, datasets, download, fewshot, service, standards, tasks, users, video_annotation
    from .routers.data import build_data_router, projects_router

    app.include_router(auth.router)
    app.include_router(users.router)
    app.include_router(datasets.router)
    app.include_router(tasks.router)
    app.include_router(video_annotation.router)
    app.include_router(standards.router)
    app.include_router(projects_router)
    app.include_router(service.router)
    app.include_router(download.router)
    app.include_router(fewshot.router)
    @app.on_event("startup")
    def recover_fewshot_queue():
        from .database import SessionLocal
        from .models import FewshotJob
        from .services.fewshot_service import ACTIVE, ensure_worker
        with SessionLocal() as db:
            pending = db.query(FewshotJob).filter(FewshotJob.status.in_(("queued",) + ACTIVE)).first()
        if pending:
            ensure_worker()

    @app.on_event("startup")
    def recover_interrupted_preprocessing():
        # These workers live in the web process. Do not silently leave stale running tasks.
        from .database import SessionLocal
        from .models import TASK_MODELS, Workflow
        from datetime import datetime
        with SessionLocal() as db:
            for model in set(TASK_MODELS.values()):
                states = ('running',) if model is Workflow else ('pending', 'running')
                for task in db.query(model).filter(model.status.in_(states)).all():
                    task.status = 'failed'
                    if hasattr(task, 'error_msg'): task.error_msg = '服务重启中断处理，请重新提交；原输入版本保留'
                    if hasattr(task, 'completed_at'): task.completed_at = datetime.now()
            db.commit()

    for data_type in ("voice", "radar", "image", "video", "text"):
        app.include_router(build_data_router(data_type))

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": "3.2.0"}

    # 前端构建产物挂载（打包环境路径经 sys._MEIPASS 解析）
    static_dir = _static_dir()
    if static_dir and static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
    return app


def _static_dir() -> Optional[Path]:
    candidates = []
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        candidates.append(Path(sys._MEIPASS) / "static")
    here = Path(__file__).resolve()
    candidates += [here.parents[2] / "dist" / "static",   # data-platform/dist/static（build.py 产物）
                   here.parents[2] / "frontend" / "dist",  # data-platform/frontend/dist（开发模式一键启动）
                   here.parents[3] / "dist" / "static"]
    for c in candidates:
        if c.exists() and (c / "index.html").exists():
            return c
    return None


def find_available_port(start: int) -> int:
    """端口被占用时自动顺延并提示实际端口。"""
    port = start
    while port < start + 100:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1
    raise RuntimeError("无可用端口")


def _open_browser(port: int) -> None:
    time.sleep(1.5)
    try:
        webbrowser.open(f"http://localhost:{port}")
    except Exception:
        pass


def main() -> None:
    print("=" * 56)
    print("  数据中台 v3.2.0（离线单机版）")
    print("=" * 56)
    # 1. config.json 已在导入时读取/生成
    print(f"[启动] 配置文件：{settings.path}")
    print(f"[启动] 上传目录：{settings.upload_dir}")
    print(f"[启动] 日志目录：{settings.log_dir}")
    # 2. 清理 temp 目录 7 天前临时文件
    removed = clean_temp_files(7)
    if removed:
        print(f"[启动] 已清理 {removed} 个过期临时文件")
    # 3-5. 数据库连接检测 / 自动建库 / create_all + schema_version
    print("[启动] 正在连接数据库并初始化表结构…")
    try:
        init_schema()
    except Exception as e:
        print(f"[错误] 数据库初始化失败：{e}")
        print("       请确认 MySQL 8.0 已启动，且 config.json 连接信息正确。")
        sys.exit(1)
    # 6. 初始化默认数据
    seed_data()
    print("[启动] 默认数据初始化完成")
    # 7. 端口顺延
    port = find_available_port(settings.server_port)
    if port != settings.server_port:
        print(f"[启动] 端口 {settings.server_port} 被占用，自动顺延至 {port}")
    print(f"[启动] 服务地址：http://localhost:{port}")
    # 8. 自动打开浏览器
    if sys.platform == "win32":
        threading.Thread(target=_open_browser, args=(port,), daemon=True).start()

    app = create_app()
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    main()
