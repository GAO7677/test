"""Silero VAD 语音活动检测服务（v3.0）。

设计决策（参照 detector.py）：
- 推理引擎：onnxruntime（与 YOLOv8n 共用同一后端）
- 模型文件：backend/models/silero_vad.onnx（~1MB）
- 懒加载 + 全局单例 + 线程锁
- 输出与手动标注一致的 segment 格式：{type: "segment", start, end, label, confidence}
- 模型缺失时抛带指引的异常，由调用方决定是否降级到 silence_split

Silero VAD ONNX 规范：
- 输入节点名："input"  形状：[batch, samples]  dtype：float32
- 输出节点名："output" 形状：[batch, frames]   dtype：float32（每帧语音概率，0~1）
- 采样率要求：16000 Hz
- 帧长：512 样本（32ms @ 16kHz），hop_length=256（50% 重叠）
- 后处理：概率 > threshold 判为语音，合并相邻语音段，过滤 < min_speech_duration 的短段
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional, Any

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 全局单例（懒加载）
# ---------------------------------------------------------------------------
_vad_session: Optional[Any] = None
_vad_lock = threading.Lock()


def _default_model_path() -> Path:
    """解析 Silero VAD 模型路径。

    优先级：
    1. config.json 中 models.vad_path
    2. 源码目录 backend/models/silero_vad.onnx
    3. PyInstaller exe 同级 models/silero_vad.onnx
    4. PyInstaller _MEIPASS/models/silero_vad.onnx
    """
    custom = settings.data.get("models", {}).get("vad_path", "")
    if custom:
        p = Path(custom)
        if p.exists():
            return p

    # 源码 / 开发环境
    src = Path(__file__).resolve().parents[3] / "models" / "silero_vad.onnx"
    if src.exists():
        return src

    # PyInstaller exe 同级
    import sys

    exe_dir = Path(sys.executable).parent
    exe_side = exe_dir / "models" / "silero_vad.onnx"
    if exe_side.exists():
        return exe_side

    # PyInstaller _MEIPASS
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        meipass = Path(sys._MEIPASS) / "models" / "silero_vad.onnx"
        if meipass.exists():
            return meipass

    # 兜底返回默认路径（让后续调用方判断是否存在）
    return src


def _load_backend() -> Any:
    """加载 ONNX Runtime 推理会话。"""
    global _vad_session
    if _vad_session is not None:
        return _vad_session

    model_path = _default_model_path()
    if not model_path.exists():
        raise RuntimeError(
            f"Silero VAD 模型文件缺失：{model_path}\n"
            "请从 https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx "
            "下载并放置到 backend/models/silero_vad.onnx"
        )

    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        str(model_path), sess_options, providers=["CPUExecutionProvider"]
    )

    # 校验输入/输出节点
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    if "input" not in input_names:
        raise RuntimeError(f"Silero VAD 模型输入节点名不匹配，期望 'input'，实际 {input_names}")
    if "output" not in output_names:
        raise RuntimeError(f"Silero VAD 模型输出节点名不匹配，期望 'output'，实际 {output_names}")

    _vad_session = session
    logger.info("Silero VAD ONNX 模型加载成功：%s", model_path)
    return _vad_session


def detect_segments(
    waveform: np.ndarray,
    sample_rate: int,
    threshold: float = 0.5,
    min_speech_duration: float = 0.25,
) -> list[dict]:
    """对音频波形执行 VAD，返回语音段列表。

    Args:
        waveform: 单声道音频样本，float32，范围建议 [-1, 1]
        sample_rate: 原始采样率（Hz）
        threshold: 语音概率阈值（0~1）
        min_speech_duration: 最短语音段时长（秒），低于此值的段被过滤

    Returns:
        [{"start": float, "end": float, "label": "语音段", "confidence": float}, ...]
    """
    session = _load_backend()

    # 重采样到 16kHz（Silero 要求）
    if sample_rate != 16000:
        n = len(waveform)
        new_n = int(n * 16000 / sample_rate)
        waveform = np.interp(
            np.linspace(0, n - 1, new_n),
            np.arange(n),
            waveform,
        ).astype(np.float32)
        sample_rate = 16000

    # 分帧：frame_length=512, hop_length=256
    frame_length = 512
    hop_length = 256
    num_frames = 1 + (len(waveform) - frame_length) // hop_length
    if num_frames <= 0:
        return []

    frames = np.lib.stride_tricks.sliding_window_view(waveform, frame_length)[::hop_length]
    if len(frames) > num_frames:
        frames = frames[:num_frames]

    # ONNX 推理：输入形状 [batch=1, samples=512]
    frames_batch = frames.astype(np.float32)
    probs = session.run(None, {"input": frames_batch})[0]  # [num_frames]

    # 后处理：阈值判断 -> 合并相邻段 -> 过滤短段
    segments = []
    in_speech = False
    seg_start = 0.0
    seg_confidences = []

    for i, prob in enumerate(probs):
        t = i * hop_length / sample_rate
        if prob > threshold:
            if not in_speech:
                in_speech = True
                seg_start = t
                seg_confidences = [float(prob)]
            else:
                seg_confidences.append(float(prob))
        else:
            if in_speech:
                in_speech = False
                seg_end = t + frame_length / sample_rate
                avg_conf = sum(seg_confidences) / len(seg_confidences) if seg_confidences else 0.0
                duration = seg_end - seg_start
                if duration >= min_speech_duration:
                    segments.append({
                        "start": seg_start,
                        "end": seg_end,
                        "label": "语音段",
                        "confidence": avg_conf,
                    })

    # 处理尾部
    if in_speech:
        seg_end = len(waveform) / sample_rate
        avg_conf = sum(seg_confidences) / len(seg_confidences) if seg_confidences else 0.0
        duration = seg_end - seg_start
        if duration >= min_speech_duration:
            segments.append({
                "start": seg_start,
                "end": seg_end,
                "label": "语音段",
                "confidence": avg_conf,
            })

    return segments


def model_available() -> bool:
    """检测模型文件是否存在且大小正常（>500KB）。"""
    try:
        path = _default_model_path()
        if not path.exists():
            return False
        size = path.stat().st_size
        return size > 500 * 1024  # > 500KB
    except Exception:
        return False
