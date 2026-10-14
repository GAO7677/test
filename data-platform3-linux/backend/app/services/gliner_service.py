"""GliNER 零样本实体抽取服务（v3.2）。

文本模态的"开放词汇"能力：任意实体类型名作为 labels 传入（如 ["装备","故障模式",
"试飞科目"]），无需训练、无需预定义类型表——与图像模态 SAM 3 的定位对应。

模型准备（有网机器一次性完成，随后拷入内网）：
    pip install gliner
    python -c "from gliner import GLiNER; \
GLiNER.from_pretrained('urchade/gliner_medium-v2.1').save_pretrained('models/gliner_medium')"
    将 models/gliner_medium/ 目录拷入内网，并在 config.json 的 models.gliner_path 配置路径
    （留空则自动探测源码/exe 同级 models/gliner_medium）。

推理：CPU 可跑（medium 约 900MB）。输出对齐现有 entities 结构。
与 vad_service 同款设计：懒加载 + 全局单例 + 线程锁 + 模型缺失抛带指引的 RuntimeError。
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, List, Optional

from app.config import settings

logger = logging.getLogger(__name__)

_session: Optional[Any] = None
_lock = threading.Lock()


def _default_model_dir() -> Path:
    """解析 GliNER 模型目录：config.json → exe 同级 → _MEIPASS → 源码 models/。"""
    custom = settings.data.get("models", {}).get("gliner_path", "")
    if custom:
        p = Path(custom)
        if p.exists():
            return p
    import sys
    exe_dir = Path(sys.executable).parent
    side = exe_dir / "models" / "gliner_medium"
    if side.exists():
        return side
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        m = Path(sys._MEIPASS) / "models" / "gliner_medium"
        if m.exists():
            return m
    return Path(__file__).resolve().parents[3] / "models" / "gliner_medium"


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
                f"GliNER 模型缺失：{model_dir}\n"
                "请在有网机器执行：\n"
                "  pip install gliner\n"
                "  python -c \"from gliner import GLiNER;"
                " GLiNER.from_pretrained('urchade/gliner_medium-v2.1')"
                ".save_pretrained('models/gliner_medium')\"\n"
                "然后将模型目录拷入内网 models/，并在 config.json 的 models.gliner_path 配置路径。"
            )
        try:
            from gliner import GLiNER
        except ImportError as e:
            raise RuntimeError("未安装 gliner（离线环境需随包分发 Windows x64 wheel 及其 torch 依赖）") from e
        _session = GLiNER.from_pretrained(str(model_dir))
        logger.info("GliNER 模型加载成功：%s", model_dir)
        return _session


def model_available() -> bool:
    try:
        from importlib.util import find_spec
        return find_spec('gliner') is not None and _default_model_dir().exists()
    except Exception:
        return False


def extract_entities(text: str, entity_types: List[str], threshold: float = 0.3) -> list[dict]:
    """零样本实体抽取，返回 [{text, type, start, end, confidence}]（字符偏移）。

    entity_types 为空时退化为通用类型，正常应由任务 params.entityTypes 传入。
    """
    model = _load_backend()
    if isinstance(entity_types, str):  # 防御：数组参数被序列化成字符串时按单类型处理
        entity_types = [entity_types]
    labels = [t for t in (entity_types or []) if t] or ["人名", "地名", "机构"]
    ents = model.predict_entities(text, labels, threshold=threshold)
    return [
        {
            "text": e.get("text", ""),
            "type": e.get("label", ""),
            "start": int(e.get("start", 0)),
            "end": int(e.get("end", 0)),
            "confidence": round(float(e.get("score", 0.0)), 4),
        }
        for e in ents
    ]
