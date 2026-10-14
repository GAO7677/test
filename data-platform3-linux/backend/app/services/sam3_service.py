"""SAM 3 开放词汇分割服务（v3.2，评估项）。

图像模态的语义层自动标注：文本概念提示（如 "engine"、"起落架"）→ 掩膜 → 外接框，
输出复用现有 detections 结构（label 即提示文本，confidence 取 stability score）。

硬件门槛：需 NVIDIA GPU（显存 ≥8GB）；CPU 环境不可用，走 503 机制（见 tasks.py）。

模型准备（有网机器一次性完成，随后拷入内网）：
    按 facebookresearch/sam3 官方 README 下载权重与配置文件到 models/sam3/，
    并在 config.json 的 models.sam3_path 配置路径（留空则自动探测）。

与 vad_service 同款设计：懒加载 + 全局单例 + 线程锁 + 模型缺失抛带指引的 RuntimeError。
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

_session: Optional[Any] = None
_lock = threading.Lock()


def _default_model_dir() -> Path:
    """解析 SAM 3 模型目录：config.json → exe 同级 → _MEIPASS → 源码 models/。"""
    custom = settings.data.get("models", {}).get("sam3_path", "")
    if custom:
        p = Path(custom)
        if p.exists():
            return p
    import sys
    exe_dir = Path(sys.executable).parent
    side = exe_dir / "models" / "sam3"
    if side.exists():
        return side
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        m = Path(sys._MEIPASS) / "models" / "sam3"
        if m.exists():
            return m
    return Path(__file__).resolve().parents[3] / "models" / "sam3"


def _load_backend() -> Any:
    global _session
    if _session is not None:
        return _session
    with _lock:
        if _session is not None:
            return _session
        model_dir = _default_model_dir()
        if not model_dir.exists():
            raise RuntimeError(
                f"SAM 3 模型缺失：{model_dir}\n"
                "请按 facebookresearch/sam3 官方 README 在有网机器下载权重与配置到 models/sam3/，\n"
                "拷入内网后在 config.json 的 models.sam3_path 配置路径。\n"
                "注意：SAM 3 需 NVIDIA GPU（显存 ≥8GB），CPU 环境不可使用。"
            )
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise RuntimeError("未安装 torch（SAM 3 依赖，离线环境需随包分发 CUDA/CPU 版）") from e
        _session = True  # 占位：推理构建见 detect()，按需接入 sam3 官方推理 API
        logger.info("SAM 3 模型目录就绪：%s", model_dir)
        return _session


def model_available() -> bool:
    try:
        return _default_model_dir().exists()
    except Exception:
        return False


def detect(image_path: str, prompt: str, conf_threshold: float = 0.5,
           max_boxes: int = 300) -> list[dict]:
    """文本提示开放词汇分割，返回 [{xyxy, label, confidence}]。

    掩膜取外接框（min/max 行列），confidence 取 SAM stability score。
    prompt 为空时抛参数错误（SAM 3 必须给出概念提示）。
    """
    if not prompt:
        raise ValueError("SAM 3 检测需要文本提示（params.prompt，如 'engine'）")
    _load_backend()  # 校验模型/依赖就绪
    raise RuntimeError(
        "SAM 3 推理尚未接入实现（评估项）：请按 facebookresearch/sam3 官方推理 API "
        "在 detect() 内完成 prompt → masks → 外接框转换后移除本异常。"
    )
