"""Spacy 中文 NER 实体识别服务（v3.0）。

设计决策（参照 detector.py）：
- 推理引擎：Spacy Pipeline（非 ONNX，但同样懒加载单例）
- 模型文件：backend/models/zh_core_web_sm/（目录结构，~30MB）
- 懒加载 + 全局单例 + 线程锁
- 输出与手动标注一致的 span 格式：{type: "span", start, end, text, label}
- 模型缺失时抛带指引的异常，由调用方决定是否降级到 entity_recog

Spacy 加载方式：
- 使用 spacy.load(path) 从指定路径加载，而非 spacy.load("zh_core_web_sm")
- path 为绝对路径（通过 _default_model_path() 解析）
- 长文本（>100000 字符）自动分段处理，避免内存溢出
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional, Any

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 全局单例（懒加载）
# ---------------------------------------------------------------------------
_nlp: Optional[Any] = None
_ner_lock = threading.Lock()

_SPACY_LABEL_MAP = {
    "PERSON": "人名",
    "GPE": "地名",
    "LOC": "地名",
    "ORG": "机构",
    "DATE": "时间",
    "TIME": "时间",
    "MONEY": "数值",
    "PERCENT": "数值",
    "CARDINAL": "数值",
    "ORDINAL": "数值",
    "NORP": "其他实体",
    "FAC": "其他实体",
    "PRODUCT": "其他实体",
    "EVENT": "其他实体",
    "WORK_OF_ART": "其他实体",
    "LAW": "其他实体",
    "LANGUAGE": "其他实体",
}


def _default_model_path() -> Path:
    """解析 Spacy 中文模型路径。

    优先级：
    1. config.json 中 models.ner_path
    2. 源码目录 backend/models/zh_core_web_sm/
    3. PyInstaller exe 同级 models/zh_core_web_sm/
    4. PyInstaller _MEIPASS/models/zh_core_web_sm/
    """
    custom = settings.data.get("models", {}).get("ner_path", "")
    if custom:
        p = Path(custom)
        if p.exists() and (p / "meta.json").exists():
            return p

    # 源码 / 开发环境
    src = Path(__file__).resolve().parents[3] / "models" / "zh_core_web_sm"
    if (src / "meta.json").exists():
        return src

    # PyInstaller exe 同级
    import sys

    exe_dir = Path(sys.executable).parent
    exe_side = exe_dir / "models" / "zh_core_web_sm"
    if (exe_side / "meta.json").exists():
        return exe_side

    # PyInstaller _MEIPASS
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        meipass = Path(sys._MEIPASS) / "models" / "zh_core_web_sm"
        if (meipass / "meta.json").exists():
            return meipass

    # 兜底返回默认路径
    return src


def _load_nlp() -> Any:
    """加载 Spacy NER Pipeline。"""
    global _nlp
    if _nlp is not None:
        return _nlp

    import spacy

    path = str(_default_model_path())
    if not Path(path).exists():
        raise RuntimeError(
            f"Spacy NER 模型文件缺失：{path}\n"
            "请从 https://github.com/explosion/spacy-models/releases/download/zh_core_web_sm-3.7.0/zh_core_web_sm-3.7.0-py3-none-any.whl "
            "下载并放置到 backend/models/zh_core_web_sm/"
        )

    nlp = spacy.load(path)
    _nlp = nlp
    logger.info("Spacy NER 模型加载成功：%s", path)
    return _nlp


def extract_entities(text: str) -> list[dict]:
    """对文本执行 NER，返回实体列表。

    长文本（>100000 字符）自动分段处理，每段 50000 字符，重叠 1000 字符。
    """
    nlp = _load_nlp()

    if len(text) <= 100000:
        doc = nlp(text)
        return _doc_to_entities(doc, text)

    # 长文本分段处理
    chunk_size = 50000
    overlap = 1000
    entities = []
    seen = set()

    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        doc = nlp(chunk_text)

        for ent in doc.ents:
            global_start = start + ent.start_char
            global_end = start + ent.end_char
            key = (global_start, global_end, ent.label_)
            if key not in seen:
                seen.add(key)
                entities.append({
                    "start": global_start,
                    "end": global_end,
                    "text": text[global_start:global_end],
                    "label": _SPACY_LABEL_MAP.get(ent.label_, "其他实体"),
                })

        if end >= len(text):
            break
        start = end - overlap

    # 按 start 排序
    entities.sort(key=lambda x: x["start"])
    return entities


def _doc_to_entities(doc: Any, text: str) -> list[dict]:
    """将 Spacy Doc 转换为标准实体格式。"""
    entities = []
    for ent in doc.ents:
        entities.append({
            "start": ent.start_char,
            "end": ent.end_char,
            "text": text[ent.start_char:ent.end_char],
            "label": _SPACY_LABEL_MAP.get(ent.label_, "其他实体"),
        })
    return entities


def model_available() -> bool:
    """检测模型目录是否存在且包含 meta.json。"""
    try:
        path = _default_model_path()
        return (path / "meta.json").exists()
    except Exception:
        return False
