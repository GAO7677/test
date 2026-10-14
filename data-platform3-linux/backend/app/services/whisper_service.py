"""faster-whisper 语音转写服务（v3.2）。

模型准备（有网机器一次性完成，随后拷入内网）：
    pip install faster-whisper
    python -c "from faster_whisper import WhisperModel; WhisperModel('small', download_root='models')"
    将生成的 models/models--Systran--faster-whisper-small 目录（或 symlink 目标）拷入内网
    models/ 下，并在 config.json 的 models.whisper_path 配置路径（留空则自动探测）。

推理：CPU int8 可跑（small 约 460MB）；输出整文件带时间戳的分段转写。
与 vad_service 同款设计：懒加载 + 全局单例 + 线程锁 + 模型缺失抛带指引的 RuntimeError。
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Optional

from app.config import settings

logger = logging.getLogger(__name__)

_session: Optional[Any] = None
_lock = threading.Lock()


def _default_model_dir() -> Path:
    """解析 faster-whisper 模型目录：config.json → exe 同级 → _MEIPASS → 源码 models/。"""
    custom = settings.data.get("models", {}).get("whisper_path", "")
    if custom:
        p = Path(custom)
        if p.exists():
            return p
    import sys
    exe_dir = Path(sys.executable).parent
    side = exe_dir / "models" / "faster-whisper-small"
    if side.exists():
        return side
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        m = Path(sys._MEIPASS) / "models" / "faster-whisper-small"
        if m.exists():
            return m
    return Path(__file__).resolve().parents[3] / "models" / "faster-whisper-small"


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
                f"faster-whisper 模型缺失：{model_dir}\n"
                "请在有网机器执行：\n"
                "  pip install faster-whisper\n"
                "  python -c \"from faster_whisper import WhisperModel;"
                " WhisperModel('small', download_root='models')\"\n"
                "然后将模型目录拷入内网 models/，并在 config.json 的 models.whisper_path 配置路径。"
            )
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise RuntimeError("未安装 faster-whisper（离线环境需随包分发 Windows x64 wheel）") from e
        _session = WhisperModel(str(model_dir), device="cpu", compute_type="int8")
        logger.info("faster-whisper 模型加载成功：%s", model_dir)
        return _session


def model_available() -> bool:
    try:
        from importlib.util import find_spec
        return find_spec('faster_whisper') is not None and _default_model_dir().exists()
    except Exception:
        return False


def transcribe(audio_path: str, language: str = "zh") -> list[dict]:
    """整文件转写，返回 [{start, end, text, confidence}]（start/end 单位秒）。

    confidence 由 avg_logprob 映射到 (0,1)：exp(avg_logprob)。
    vad_filter=True 自动跳过纯静音/音乐段，无需先跑 VAD。
    """
    model = _load_backend()
    segments, _info = model.transcribe(
        audio_path, language=language or None, beam_size=5,
        vad_filter=True, word_timestamps=False,
    )
    out = []
    for seg in segments:
        conf = max(0.0, min(1.0, float(2.718281828 ** seg.avg_logprob)))
        out.append({
            "start": round(float(seg.start), 3),
            "end": round(float(seg.end), 3),
            "text": (seg.text or "").strip(),
            "confidence": round(conf, 4),
        })
    return out
