"""config.json 读写：启动时读取，不存在则生成默认配置；jwt_secret 占位时自动生成随机密钥写回。

Linux 使用 /data 下独立运行目录；相对路径以项目根目录解析。Windows 保留原目录回退策略。
"""
from __future__ import annotations

import json
import os
import secrets
import sys
import threading
from pathlib import Path, PureWindowsPath

_lock = threading.Lock()

JWT_SECRET_PLACEHOLDER = "<首次启动时自动生成随机密钥并写回>"

DEFAULT_CONFIG = {
    "database": {
        "host": "localhost",
        "port": 3306,
        "username": "root",
        "password": "123456",
        "dbname": "data_platform",
    },
    "upload_dir": "/data/gaoya/agent-data/outputs_v1/data_platform3_linux/runtime/Uploads" if os.name != "nt" else r"D:\DataPlatform\Uploads",
    "log_dir": "/data/gaoya/agent-data/outputs_v1/data_platform3_linux/runtime/Logs" if os.name != "nt" else r"D:\DataPlatform\Logs",
    "jwt_secret": JWT_SECRET_PLACEHOLDER,
    "server_port": 8000,
    "security": {
        "max_login_attempts": 5,
        "lockout_duration_minutes": 30,
        "session_timeout_minutes": 120,
        "allow_concurrent_sessions": True,
    },
    "models": {
        "vad_path": "",
        "ner_path": "",
        "whisper_path": "",
        "gliner_path": "",
        "sam3_path": "",
        "sam2_path": "",
    },
}


def base_dir() -> Path:
    """程序同级目录（兼容 PyInstaller 打包环境）。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]  # project root


class AppConfig:
    def __init__(self) -> None:
        self.path = Path(os.environ.get("DP_CONFIG_PATH", base_dir() / "config.json"))
        self.data: dict = {}
        self.load()

    # ---------- 读写 ----------
    def load(self) -> None:
        with _lock:
            if not self.path.exists():
                self.data = json.loads(json.dumps(DEFAULT_CONFIG))
                self._write()
            else:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            # 补齐缺失键
            changed = False
            for k, v in DEFAULT_CONFIG.items():
                if k not in self.data:
                    self.data[k] = v
                    changed = True
                elif isinstance(v, dict):
                    for k2, v2 in v.items():
                        if k2 not in self.data[k]:
                            self.data[k][k2] = v2
                            changed = True
            # jwt_secret 占位值 → 自动生成密码学安全随机密钥并写回
            if self.data.get("jwt_secret") in (None, "", JWT_SECRET_PLACEHOLDER):
                self.data["jwt_secret"] = secrets.token_urlsafe(48)
                changed = True
            if changed:
                self._write()
            self._resolve_paths()
            self._ensure_dirs()

    def _write(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def save(self) -> None:
        with _lock:
            self._write()

    # ---------- 目录 ----------
    def _resolve_paths(self) -> None:
        """配置中的相对路径以项目根目录为基准，不随启动目录变化。"""
        for section, keys in ((self.data, ("upload_dir", "log_dir")),
                              (self.data["models"], tuple(self.data["models"])),
                              (self.data.get("detector", {}), ("model_path",))):
            for key in keys:
                value = section.get(key)
                if not value:
                    continue
                if os.name != "nt" and PureWindowsPath(value).drive:
                    raise ValueError(f"Linux 配置不能使用 Windows 路径：{key}={value}")
                path = Path(value).expanduser()
                section[key] = str(path if path.is_absolute() else base_dir() / path)

    def _ensure_dir_with_fallback(self, key: str) -> None:
        """确保目录存在；失败时回退到程序同级目录 DataPlatform\\ 并写回。"""
        p = Path(self.data[key])
        try:
            p.mkdir(parents=True, exist_ok=True)
        except OSError:
            if os.name != "nt":
                raise  # Linux 显式报告权限/路径问题，不静默写入另一目录。
            dirname = "Uploads" if "upload" in key else "Logs"
            fallback = base_dir() / "DataPlatform" / dirname
            fallback.mkdir(parents=True, exist_ok=True)
            self.data[key] = str(fallback)
            self._write()
            print(f"[配置] {p} 不可用，已回退到 {fallback} 并写回 config.json")

    def _ensure_dirs(self) -> None:
        self._ensure_dir_with_fallback("upload_dir")
        self._ensure_dir_with_fallback("log_dir")
        # 上传根目录子结构
        root = Path(self.data["upload_dir"])
        for sub in ("voice", "radar", "image", "video", "text",
                    "processed", "templates", "exports", "temp",
                    "writeback", "standards"):
            (root / sub).mkdir(parents=True, exist_ok=True)

    # ---------- 便捷访问 ----------
    @property
    def db(self) -> dict:
        return self.data["database"]

    @property
    def upload_dir(self) -> Path:
        return Path(self.data["upload_dir"])

    @property
    def log_dir(self) -> Path:
        return Path(self.data["log_dir"])

    @property
    def jwt_secret(self) -> str:
        return self.data["jwt_secret"]

    @property
    def server_port(self) -> int:
        return int(self.data.get("server_port", 8000))

    @property
    def security(self) -> dict:
        return self.data.setdefault("security", dict(DEFAULT_CONFIG["security"]))


settings = AppConfig()
