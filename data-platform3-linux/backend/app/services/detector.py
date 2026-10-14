"""COCO 语义目标检测服务（v2.0）：YOLOv8n ONNX 离线推理（双后端）。

设计决策：
- 推理引擎双后端：onnxruntime 优先（独立解析器，不受 OpenCV 构建批次影响；
  3.8/3.10/3.14 均有 Windows x64 离线 wheel），cv2.dnn 兜底回退。
  背景：部分 2026 年构建的 OpenCV 5.0 默认启用新 ONNX 解析器（ONNXImporter2），
  其 parseFile 阶段即拒绝合法模型，故不能以 cv2.dnn 为唯一后端。
- 模型文件 backend/models/yolov8n.onnx（12.8MB，opset 13 + onnxsim 简化，640x640 静态输入），
  随 PyInstaller 打包，完全离线可用；也可用 config.json 的 detector.model_path 覆盖。
- 懒加载 + 全局单例，避免每张图重复读模型；线程锁保证后台任务并发安全。
- 输出与手动标注/COCO 导入一致的 xyxy 像素坐标 + COCO 类别名 + 置信度。
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from ..config import base_dir, settings

# COCO 80 类词表（英文原名，与 COCO 数据集一致；前端可按需映射中文）
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

INPUT_SIZE = 640

_backend = None   # {"type": "onnxruntime"|"cv2", "obj": session|net}
_lock = threading.Lock()


def _default_model_path() -> Path:
    """模型路径：config 覆盖 > 源码目录 > exe 同级 models 目录（PyInstaller）。"""
    custom = str(settings.data.get("detector", {}).get("model_path", "")) if hasattr(settings, "data") else ""
    if custom and Path(custom).exists():
        return Path(custom)
    candidates = [
        Path(__file__).resolve().parents[2] / "models" / "yolov8n.onnx",  # backend/models/
        base_dir() / "models" / "yolov8n.onnx",                            # exe 同级
    ]
    import sys as _sys
    meipass = getattr(_sys, "_MEIPASS", None)  # PyInstaller onefile 解包目录
    if meipass:
        candidates.append(Path(meipass) / "models" / "yolov8n.onnx")
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _load_backend(p: Path) -> dict:
    """加载推理后端：onnxruntime 优先，cv2.dnn 回退。"""
    size = p.stat().st_size
    if size < 1_000_000:  # 正常模型约 12.8MB，明显偏小说明文件损坏/未下载完整
        raise RuntimeError(f"检测模型文件损坏（仅 {size} 字节，应为约 12.8MB）：{p}，请重新解压完整 zip")
    ort_err = None
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        return {"type": "onnxruntime", "obj": sess, "input": sess.get_inputs()[0].name}
    except Exception as e:  # noqa: BLE001 —— 任何 ort 失败都回退 cv2.dnn
        ort_err = e
    try:
        import cv2
        net = cv2.dnn.readNetFromONNX(str(p))
        return {"type": "cv2", "obj": net}
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"检测模型加载失败：onnxruntime 后端错误（{ort_err}）；"
            f"cv2.dnn 后端错误（OpenCV 加载失败：{e}）。路径 {p}，文件 {size} 字节。"
        ) from e


def _get_backend() -> dict:
    """懒加载全局后端单例（线程安全）。模型缺失时抛带指引的异常。"""
    global _backend
    if _backend is None:
        with _lock:
            if _backend is None:
                p = _default_model_path()
                if not p.exists():
                    raise RuntimeError(f"检测模型缺失：{p}（请将 yolov8n.onnx 放入 backend/models/）")
                _backend = _load_backend(p)
    return _backend


def _run(blob: np.ndarray) -> np.ndarray:
    """统一推理入口，返回 (8400, 84) 预测矩阵。"""
    be = _get_backend()
    if be["type"] == "onnxruntime":
        return be["obj"].run(None, {be["input"]: blob})[0][0].T
    net = be["obj"]
    net.setInput(blob)
    return net.forward()[0].T


def detect_frame(img: np.ndarray, conf: float = 0.4, nms: float = 0.45,
                 max_boxes: int = 300) -> list[dict]:
    """对单帧 BGR 图像检测，返回 [{x,y,w,h,label,confidence}]（原图像素坐标）。"""
    import cv2
    h0, w0 = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE),
                                 swapRB=True, crop=False)
    preds = _run(blob)  # (8400, 84) = xywh + 80 类分数
    boxes, scores, class_ids = [], [], []
    sx, sy = w0 / INPUT_SIZE, h0 / INPUT_SIZE
    for row in preds:
        cls_scores = row[4:]
        cid = int(np.argmax(cls_scores))
        conf_i = float(cls_scores[cid])
        if conf_i < conf:
            continue
        cx, cy, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        boxes.append([int((cx - bw / 2) * sx), int((cy - bh / 2) * sy),
                      int(bw * sx), int(bh * sy)])
        scores.append(conf_i)
        class_ids.append(cid)
    results = []
    if boxes:
        idxs = cv2.dnn.NMSBoxes(boxes, scores, conf, nms)
        for i in np.array(idxs).flatten()[:max_boxes]:
            x, y, w, h = boxes[int(i)]
            results.append({
                "x": max(0, x), "y": max(0, y),
                "w": min(w, w0), "h": min(h, h0),
                "label": COCO_CLASSES[class_ids[int(i)]] if class_ids[int(i)] < len(COCO_CLASSES) else f"class_{class_ids[int(i)]}",
                "confidence": round(scores[int(i)], 3),
            })
    return results


def detect_image(path: Path, conf: float = 0.4, nms: float = 0.45,
                 max_boxes: int = 300) -> dict:
    """检测图片文件，返回含 width/height/boxes 的结果字典。"""
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        return {"boxes": [], "error": "无法读取图像"}
    h, w = img.shape[:2]
    boxes = detect_frame(img, conf=conf, nms=nms, max_boxes=max_boxes)
    return {"width": w, "height": h, "boxes": boxes, "box_count": len(boxes)}


def model_available() -> bool:
    return _default_model_path().exists()
