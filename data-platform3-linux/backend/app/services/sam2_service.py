"""SAM 2 视频目标分割服务（v3.2，二期占位）。

视频模态的开放词汇分割：文本/点提示 → 掩膜贯穿全片。一期只注册规则与
model_ready 挂接，推理实现随二期接入（GPU 前提同 sam3_service）。

模型准备：同 sam3_service，config.json 的 models.sam2_path 指向 models/sam2/。
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


def _default_model_dir() -> Path:
    custom = settings.data.get("models", {}).get("sam2_path", "")
    if custom:
        p = Path(custom)
        if p.exists():
            return p
    import sys
    exe_dir = Path(sys.executable).parent
    side = exe_dir / "models" / "sam2"
    if side.exists():
        return side
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        m = Path(sys._MEIPASS) / "models" / "sam2"
        if m.exists():
            return m
    return Path(__file__).resolve().parents[3] / "models" / "sam2"


def model_available() -> bool:
    try:
        return _default_model_dir().exists()
    except Exception:
        return False


def segment_video(video_path: str, prompt: str, frame_interval: int = 2) -> list[dict]:
    """二期实现：返回 [{frame_index, time_offset, mask_rle}]。"""
    raise RuntimeError("SAM 2 视频分割为二期功能：待 GPU 环境确认后接入实现")
