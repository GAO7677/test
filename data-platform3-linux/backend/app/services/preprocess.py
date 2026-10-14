"""预处理真实执行服务（8.2）：自动标注 / 清洗 / 集成 / 转换 / 采样 / 工作流。

全部任务读取 input_version 目录文件执行，产物写入 output_version 目录
（processed/{dataset_id}/V{n}/）。后台线程执行，进度写库供前端轮询。
"""
from __future__ import annotations
from typing import Optional

import csv
import hashlib
import json
import random
import re
import shutil
import subprocess
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from ..database import SessionLocal
from ..models import (AutoLabelTask, CleanTask, Dataset, DatasetVersion,
                      IntegrateTask, SampleDataset, TASK_MODELS, TransformTask,
                      Workflow)
from . import version_service as vs


def list_files(directory: Path) -> list[Path]:
    return sorted([p for p in directory.rglob("*") if p.is_file()]) if directory.exists() else []


def data_files(directory: Path) -> list[Path]:
    """版本目录中的数据文件（排除 _label.json 标注文件与清单文件）。"""
    return [p for p in list_files(directory)
            if not p.name.endswith("_label.json") and p.name != "_manifest.json"]


def copy_all(src: Path, dst: Path) -> None:
    for p in list_files(src):
        target = dst / p.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)


def write_label(dst_dir: Path, data_file: Path, label: dict, src_dir: Optional[Path] = None) -> None:
    """写入标注 JSON；传入 src_dir 时镜像源相对目录结构（文件夹上传场景同名 stem 不冲突）。"""
    name = f"{data_file.stem}_label.json"
    target = dst_dir / name
    if src_dir:
        try:
            rel = data_file.relative_to(src_dir)
            target = dst_dir / rel.parent / name
        except ValueError:
            pass
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(label, f, ensure_ascii=False, indent=2)


# ==================== 自动标注算法 ====================
def label_voice(path: Path, algorithm: str, params: dict) -> dict:
    if algorithm == "whisper_transcribe":
        from .whisper_service import transcribe
        segments = [{**a, "label": "语音段"} for a in transcribe(str(path), language=params.get("language", "zh"))]
        return {"type": "voice", "algorithm": algorithm, "segments": segments, "segment_count": len(segments)}
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    duration = len(mono) / sr if sr else 0
    segments = []
    if algorithm in ("silence_split", "energy_gate"):
        threshold_db = float(params.get("threshold", -40))
        min_dur = float(params.get("minDuration", 200)) / 1000
        max_dur = float(params.get("maxDuration", 10000)) / 1000
        win = int(sr * 0.02) or 1  # 20ms 帧
        frames = len(mono) // win
        energies = np.array([float(np.sqrt(np.mean(mono[i * win:(i + 1) * win] ** 2)) + 1e-10)
                             for i in range(frames)])
        db = 20 * np.log10(energies)
        active = db > threshold_db
        start = None
        for i, a in enumerate(active):
            if a and start is None:
                start = i
            elif not a and start is not None:
                seg_dur = (i - start) * win / sr
                if seg_dur >= min_dur:
                    segments.append({"start": round(start * win / sr, 3),
                                     "end": round(min(i * win / sr, start * win / sr + max_dur), 3),
                                     "label": "有效语音段"})
                start = None
        if start is not None:
            segments.append({"start": round(start * win / sr, 3),
                             "end": round(duration, 3), "label": "有效语音段"})
    elif algorithm == "fixed_slice":
        slice_dur = float(params.get("sliceDuration", 10))
        overlap = float(params.get("overlap", 0.5))
        if not slice_dur > 0 or not 0 <= overlap < 1:
            raise ValueError("切片时长须大于0，重叠比例须在[0,1)内")
        step = slice_dur * (1 - overlap)
        t = 0.0
        idx = 1
        while t < duration:
            segments.append({"start": round(t, 3), "end": round(min(t + slice_dur, duration), 3),
                             "label": f"切片_{idx}"})
            t += step
            idx += 1
    elif algorithm == "silero_vad":
        from .vad_service import detect_segments, model_available
        if not model_available():
            raise RuntimeError("Silero VAD 模型文件缺失")
        segments = detect_segments(mono, sr, threshold=params.get("threshold", 0.5),
                                   min_speech_duration=params.get("min_speech_duration", 0.25))
        segments = [{"start": round(s["start"], 3), "end": round(s["end"], 3),
                     "label": "语音段", "confidence": round(s.get("confidence", 0), 3)}
                    for s in segments]
    elif algorithm == "whisper_transcribe":
        from .whisper_service import transcribe
        texts = transcribe(str(path), language=params.get("language", "zh"))
        segments = [{"start": s["start"], "end": s["end"], "label": "语音段",
                     "confidence": s.get("confidence", 0.0), "text": s["text"]}
                    for s in texts]
    else:  # pitch_track 等其他规则 → 全段标注
        segments.append({"start": 0, "end": round(duration, 3), "label": "全段"})
    return {"type": "voice", "algorithm": algorithm, "duration": round(duration, 2),
            "sample_rate": sr, "segments": segments, "segment_count": len(segments)}


def label_radar(path: Path, algorithm: str, params: dict) -> dict:
    from .file_service import _load_radar_points
    pts = _load_radar_points(path, max_points=100000)
    detections = []
    if len(pts):
        if pts.shape[1] < 4: raise ValueError("强度检测需要真实 intensity 字段，不能用坐标代替")
        intensity = pts[:, 3]
        if algorithm == "intensity_threshold":
            thr = float(params.get("rcsThreshold", 15))
            mask = intensity >= thr
            detections = pts[mask][:, :3].tolist()
        elif algorithm == "clustering":
            eps = float(params.get("eps", 2.5))
            min_pts = int(params.get("minPts", 5))
            if eps <= 0 or min_pts < 1:
                raise ValueError(f"聚类参数非法：eps 须为正数、minPts 须为正整数（当前 eps={eps}, minPts={min_pts}）")
            detections = _dbscan(pts[:, :3], eps, min_pts)
        elif algorithm == "cfar":
            guard = int(params.get("guardCells", 4))
            train = int(params.get("trainingCells", 16))
            if guard < 1 or train < 1:
                raise ValueError(f"CFAR 参数非法：guardCells 与 trainingCells 须为正整数（当前 guardCells={guard}, trainingCells={train}）")
            detections = _cfar(intensity, pts[:, :3], guard, train)
        else:  # doppler_filter：按 z 维速度近似过滤
            vmin = float(params.get("velocityMin", 0.5))
            vmax = float(params.get("velocityMax", 100))
            mask = (np.abs(pts[:, 2]) >= vmin) & (np.abs(pts[:, 2]) <= vmax)
            detections = pts[mask][:, :3].tolist()
    return {"type": "radar", "algorithm": algorithm,
            "point_count": int(len(pts)), "detections": detections[:5000],
            "detection_count": len(detections)}


def _dbscan(points: np.ndarray, eps: float, min_pts: int) -> list:
    """DBSCAN 聚类（numpy 实现，返回各簇中心作为检测目标）。"""
    n = len(points)
    if n == 0:
        return []
    labels = np.full(n, -1, dtype=int)
    cluster_id = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        dists = np.linalg.norm(points - points[i], axis=1)
        neighbors = np.where(dists <= eps)[0]
        if len(neighbors) < min_pts:
            continue
        labels[neighbors] = cluster_id
        cluster_id += 1
    centers = []
    for c in range(cluster_id):
        members = points[labels == c]
        if len(members) >= min_pts:
            centers.append({"center": np.round(members.mean(axis=0), 3).tolist(),
                            "size": int(len(members)), "label": f"目标簇_{c + 1}"})
    return centers


def _cfar(intensity: np.ndarray, points: np.ndarray, guard: int, train: int) -> list:
    """一维 CFAR（恒虚警率）目标检测：滑窗噪声估计 + 全局显著性补充。

    修正要点：此前先按强度降序排序再滑窗，滑动窗口从 pos=window 起扫描，
    导致排序最前的强目标（真实高亮目标）被直接跳过。现改为
    ① 在原始顺序上滑窗估计局部噪声底；② 用「全局均值 + 3σ」补回显著高亮目标，
    确保样本中预设的高 RCS 目标能被检出且坐标与原始数据一致。
    """
    n = len(intensity)
    if n < 2 * (guard + train) + 1:
        return []
    window = guard + train
    detections, seen = [], set()
    # 用中位数 + MAD 估计噪声水平，避免少量极端值（异常点）抬高门限而漏掉真实目标
    median_i = float(np.median(intensity))
    mad_i = float(np.median(np.abs(intensity - median_i)))
    robust_sigma = 1.4826 * mad_i if mad_i > 0 else max(float(np.std(intensity)), 1e-6)
    global_thr = median_i + 3.0 * robust_sigma
    for pos in range(window, n - window):
        noise = np.concatenate([intensity[pos - window:pos - guard],
                                intensity[pos + guard:pos + window]])
        med_n = float(np.median(noise))
        mad_n = float(np.median(np.abs(noise - med_n)))
        sigma_n = 1.4826 * mad_n if mad_n > 0 else max(float(np.std(noise)), 1e-6)
        threshold = max(med_n + 4.0 * sigma_n, global_thr)
        if intensity[pos] > threshold:
            detections.append(points[pos].tolist())
            seen.add(pos)
    # 全局显著性补充：强度超过稳健门限的强目标点（确保高亮目标不漏检）
    for pos in np.where(intensity > global_thr)[0]:
        pos = int(pos)
        if pos not in seen:
            detections.append(points[pos].tolist())
            seen.add(pos)
    detections.sort(key=lambda d: -float(d[0] ** 2 + d[1] ** 2 + d[2] ** 2))
    return detections[:200]


def label_image(path: Path, algorithm: str, params: dict) -> dict:
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        return {"type": "image", "algorithm": algorithm, "boxes": [], "error": "无法读取图像"}
    h, w = img.shape[:2]
    boxes = []
    if algorithm == "coco_detect":  # v2.0：YOLOv8n 离线语义检测（COCO 80 类）
        from . import detector
        res = detector.detect_image(path, conf=float(params.get("confThreshold", 0.4)),
                                    nms=float(params.get("nmsThreshold", 0.45)),
                                    max_boxes=int(params.get("maxBoxes", 300)))
        if res.get("error"):
            return {"type": "image", "algorithm": algorithm, "boxes": [], "error": res["error"]}
        return {"type": "image", "algorithm": algorithm, "model": "yolov8n-coco",
                "width": w, "height": h, "boxes": res["boxes"], "box_count": res["box_count"]}
    if algorithm == "sam3_detect":
        from .sam3_service import detect as sam3_detect
        sam3_dets = sam3_detect(str(path), params.get("prompt", ""),
                                conf_threshold=float(params.get("confThreshold", 0.5)))
        for d in sam3_dets:
            x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
            boxes.append({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1,
                          "label": d.get("label", "目标"), "confidence": round(float(d.get("confidence", 0.0)), 4)})
        return {"type": "image", "algorithm": algorithm, "model": "sam3",
                "width": w, "height": h, "boxes": boxes, "box_count": len(boxes)}
    if algorithm in ("edge_detect", "contour_detect"):
        min_area = float(params.get("minArea", 100))
        thr = float(params.get("edgeThreshold", 50))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, thr, thr * 2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) >= min_area:
                x, y, bw, bh = cv2.boundingRect(c)
                boxes.append({"x": x, "y": y, "w": bw, "h": bh, "label": "候选目标"})
                if len(boxes) >= 200:
                    break
    elif algorithm == "color_cluster":
        k = int(params.get("k", 5))
        data = img.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    int(params.get("maxIter", 100)), 0.2)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        for c in range(k):
            mask = (labels.flatten() == c).reshape(h, w).astype(np.uint8)
            ys, xs = np.where(mask > 0)
            if len(xs):
                boxes.append({"x": int(xs.min()), "y": int(ys.min()),
                              "w": int(xs.max() - xs.min()), "h": int(ys.max() - ys.min()),
                              "label": f"色彩区域_{c + 1}",
                              "color": [int(v) for v in centers[c]]})
    else:
        raise ValueError("未支持的图像标注算法")
    return {"type": "image", "algorithm": algorithm, "width": w, "height": h,
            "boxes": boxes, "box_count": len(boxes)}


def label_video(path: Path, algorithm: str, params: dict) -> dict:
    import cv2
    if algorithm == "sam2_video":  # v3.2 二期占位：segment_video 抛"二期功能"
        from .sam2_service import segment_video
        segment_video(str(path), params.get("prompt", ""),
                      frame_interval=int(params.get("frameInterval", 2)))
    if algorithm == "coco_detect":  # v2.0：抽帧语义检测，框回写到对应时间点
        from . import detector
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        interval = max(1, int(float(params.get("frameInterval", 2)) * fps))  # 默认每 2 秒一帧
        conf = float(params.get("confThreshold", 0.4))
        nms = float(params.get("nmsThreshold", 0.45))
        max_frames = int(params.get("maxFrames", 200))
        if not max_frames > 0 or not float(params.get("frameInterval", 2)) > 0:
            raise ValueError("抽帧间隔和最大帧数须大于0")
        items, idx, scanned = [], 0, 0
        while scanned < max_frames and (not total or idx < total):
            ok = cap.grab()
            if not ok:
                break
            if idx % interval == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                boxes = detector.detect_frame(frame, conf=conf, nms=nms)
                items.append({"frame": idx, "time": idx / fps,
                              "boxes": boxes, "status": "detected" if boxes else "empty",
                              "label": f"{len(boxes)} 个目标"})
                scanned += 1
            idx += 1
        cap.release()
        if not scanned: raise ValueError("视频无法解码或没有可处理帧")
        return {"type": "video", "algorithm": algorithm, "model": "yolov8n-coco",
                "fps": round(fps, 2), "frames_detected": scanned,
                "items": items, "item_count": len(items),
                "sampling": "interval", "frame_interval_seconds": interval / fps,
                "total_frames": total, "processed_end_time": items[-1]['time'],
                "truncated": bool(total and items[-1]['frame'] + interval < total),
                "coverage_note": "仅标注采样帧；达到最大帧数后停止，不代表全片逐帧标注"}
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    keyframes, motions = [], []
    prev = None
    idx = 0
    threshold = float(params.get("threshold", 8))
    if threshold <= 0:
        raise ValueError(f"关键帧参数非法：threshold 须为正数（当前 {threshold}）")
    sens_cfg = float(params.get("sensitivity", 5))
    if sens_cfg <= 0:
        raise ValueError(f"运动检测参数非法：sensitivity 须为正数（当前 {sens_cfg}）")
    min_interval = max(1, int(params.get("minInterval", 1))) * fps
    last_kf = -min_interval
    max_scan = min(total, 2000) if total else 2000
    while idx < max_scan:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.cvtColor(cv2.resize(frame, (64, 36)), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            delta = np.abs(small.astype(np.int16) - prev.astype(np.int16))
            diff = float(delta.mean())
            # 变化像素占比：移动目标只占画面一小部分时，平均帧差会偏小，用占比补充判定
            ratio = float((delta > 12).mean())
            if algorithm in ("keyframe_extract", "scene_segment"):
                if (diff > threshold or ratio >= 0.01) and idx - last_kf >= min_interval:
                    keyframes.append({"frame": idx, "time": round(idx / fps, 2),
                                      "diff": round(diff, 2), "change_ratio": round(ratio, 4),
                                      "label": "关键帧"})
                    last_kf = idx
            elif algorithm in ("motion_detect", "object_track"):
                sens = sens_cfg
                if diff > sens or ratio >= 0.01:
                    motions.append({"frame": idx, "time": round(idx / fps, 2),
                                    "motion": round(diff, 2), "change_ratio": round(ratio, 4),
                                    "label": "运动目标"})
        prev = small
        idx += 1
    cap.release()
    items = keyframes if algorithm in ("keyframe_extract", "scene_segment") else motions
    return {"type": "video", "algorithm": algorithm, "fps": round(fps, 2),
            "frames_scanned": idx, "total_frames": total, "truncated": idx < total or len(items) > 1000,
            "processed_end_time": idx / fps, "items": items[:1000], "item_count": min(len(items), 1000)}


def label_text(path: Path, algorithm: str, params: dict) -> dict:
    from .label_transform import read_text
    text = read_text(path)
    result: dict = {"type": "text", "algorithm": algorithm,
                    "line_count": len(text.splitlines())}
    if algorithm == "keyword_extract":
        top_k = int(params.get("topK", 10))
        if top_k < 1:
            raise ValueError(f"关键词参数非法：topK 须为正整数（当前 {top_k}）")
        words = re.findall(r"[A-Za-z]{2,}|[一-鿿]{2,}", text)
        freq = Counter(words)
        # 简化 TF-IDF：词频 * 长度权重
        scored = sorted(freq.items(), key=lambda kv: kv[1] * min(len(kv[0]), 6), reverse=True)
        result["keywords"] = [{"word": w, "score": round(c / max(len(words), 1) * 100, 2)}
                              for w, c in scored[:top_k]]
    elif algorithm == "entity_recog":
        # 规则实体识别：机构/地名/人名（词表 + 边界规则），编号/数值/日期（正则）
        entities = []
        _surname = ("赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳唐费廉岑薛雷贺倪汤"
                    "滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄和穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞熊纪舒屈项祝董梁杜阮蓝闵席季麻强贾路娄危江童颜郭"
                    "梅盛林刁钟徐邱骆高夏蔡田樊胡凌霍虞万支柯昝管卢莫房裘缪干解应宗丁宣邓郁单杭洪包诸左石崔吉钮龚程嵇邢滑裴陆荣翁荀羊惠甄曲家封芮羿储靳汲邴糜松井段富巫"
                    "乌焦巴弓牧隗山谷车侯宓蓬全郗班仰秋仲伊宫宁仇栾暴甘钭厉戎祖武符刘景詹束龙叶幸司韶郜黎蓟薄印宿白怀蒲台从鄂索咸籍赖卓蔺屠蒙池乔阴胥能苍双闻莘党翟谭贡"
                    "劳逄姬申扶堵冉宰郦雍却璩桑桂濮牛寿通边扈燕冀郏浦尚农温别庄晏柴瞿阎充慕连茹习宦艾鱼容向古易慎戈廖庾终暨居衡步都耿满弘匡国文寇广禄阙东欧殳沃利蔚越夔"
                    "隆师巩厍聂晁勾敖融冷訾辛阚那简饶空曾毋沙乜养鞠须丰巢关蒯相查后荆红游竺权逯盖益桓公")
        _place_suffix = ("省", "市", "县", "区", "镇", "村", "路", "街", "基地", "机场", "港口", "岛", "山", "河", "湖")
        _org_suffix = ("公司", "集团", "研究院", "研究所", "大学", "学院", "中心", "基地",
                       "委员会", "管理局", "总局", "实验室")
        _cities = ("北京", "上海", "天津", "重庆", "成都", "西安", "沈阳", "广州", "深圳", "南京",
                   "杭州", "武汉", "哈尔滨", "长春", "大连", "青岛", "济南", "郑州", "长沙", "合肥",
                   "福州", "厦门", "昆明", "贵阳", "南宁", "兰州", "银川", "西宁", "乌鲁木齐", "拉萨",
                   "呼和浩特", "石家庄", "太原", "南昌", "海口", "三亚", "苏州", "无锡", "宁波", "温州")
        _bad_lead = set("来自由与和的在对于把被将于之其该此")
        _bad_inner = set("的了在是与和及或把被将对于之其该此一二三四五六七八九十百千万亿零几年月日个项条份名次")
        _not_name = {"计划", "任务", "完成", "标准", "立即", "建议", "意见", "方案", "系统", "数据", "项目",
                     "工作", "会议", "时间", "情况", "问题", "内容", "结果", "过程", "方法", "方式", "要求",
                     "规定", "制度", "管理", "技术", "设备", "信息", "服务", "平台", "中心", "过去", "来自",
                     "集团", "公司", "研究", "试验", "基地", "单位", "人员", "组织", "机构", "部门", "负责",
                     "联系", "电话", "统计", "处理", "检测", "分析", "评估", "控制", "完成", "指出"}
        _verbs = ("主持", "指出", "强调", "介绍", "提出", "要求", "负责", "表示", "认为", "确认", "约定",
                  "担任", "参加", "出席", "联系", "签发", "批示", "汇报", "说明", "宣布", "主讲")
        _span: list = []

        def _trim_lead(nm: str) -> str:
            while nm and nm[0] in _bad_lead:
                nm = nm[1:]
            return nm

        def _overlapped(a_: int, b_: int) -> bool:
            return any(x <= a_ < y or x < b_ <= y or (a_ <= x and b_ >= y) for x, y in _span)

        # 机构（先匹配，避免被地名/人名规则切碎）
        for m in re.finditer(r"[\u4e00-\u9fff]{2,12}?(?:" + "|".join(_org_suffix) + r")", text):
            raw = m.group(0)
            nm = _trim_lead(raw)
            if not (2 <= len(nm) <= 12) or any(ch in _bad_inner for ch in nm):
                continue
            entities.append({"entity": nm, "type": "机构"})
            _span.append((m.start() + (len(raw) - len(nm)), m.end()))
        # 地名：后缀规则 + 常用城市词典
        for m in re.finditer(r"[\u4e00-\u9fff]{2,8}?(?:" + "|".join(_place_suffix) + r")", text):
            raw = m.group(0)
            nm = _trim_lead(raw)
            if not (2 <= len(nm) <= 8) or any(ch in _bad_inner for ch in nm) or _overlapped(m.start(), m.end()):
                continue
            entities.append({"entity": nm, "type": "地名"})
            _span.append((m.start(), m.end()))
        for city in _cities:
            for m in re.finditer(re.escape(city), text):
                if not _overlapped(m.start(), m.end()):
                    entities.append({"entity": city, "type": "地名"})
                    _span.append((m.start(), m.end()))
        # 人名：姓氏 + 1~2 汉字，且其后紧跟动词或标点（避免普通词被误判）
        _sre = "|".join(sorted(set(_surname)))
        for m in re.finditer(r"([" + _sre + r"])([\u4e00-\u9fff]{1,2})", text):
            if _overlapped(m.start(), m.end()):
                continue
            cand = None
            for ln_ in (1, 2):
                if len(m.group(2)) < ln_:
                    continue
                nm = m.group(1) + m.group(2)[:ln_]
                tail = text[m.start(2) + ln_: m.start(2) + ln_ + 2]
                if nm in _not_name:
                    continue
                if any(tail.startswith(v) for v in _verbs):
                    cand = (nm, m.start(2) + ln_)
                    break
            if cand:
                entities.append({"entity": cand[0], "type": "人名"})
                _span.append((m.start(), cand[1]))
        # 编号 / 日期 / 数值（数值用非捕获组，保证匹配到完整数值而非仅单位）
        for etype, pat in (("日期", r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}"),
                           ("编号", r"[A-Z]{1,3}-?\d{3,}"),
                           ("数值", r"\d+(?:\.\d+)?\s?(?:km/h|m/s|dB|°|m²|g|小时|项|条|帧|份|个|名|次)?")):
            for mm in re.finditer(pat, text):
                if etype == "数值" and _overlapped(mm.start(), mm.end()):
                    continue
                val = mm.group(0).strip()
                if val:
                    entities.append({"entity": val, "type": etype})
                    if etype == "日期":
                        _span.append((mm.start(), mm.end()))
        # entityTypes 参数过滤（缺省返回全部类型）
        want = params.get("entityTypes") or []
        if isinstance(want, list) and want:
            entities = [e for e in entities if e["type"] in want]
        # 去重，保持出现顺序
        seen, uniq = set(), []
        for e in entities:
            k = (e["entity"], e["type"])
            if k not in seen:
                seen.add(k)
                uniq.append(e)
        result["entities"] = uniq[:200]
    elif algorithm == "sentiment_rule":
        pos_words = ["正常", "良好", "稳定", "成功", "合格", "good", "normal", "success"]
        neg_words = ["故障", "异常", "失败", "报警", "超限", "error", "fail", "alarm", "warning"]
        pos = sum(text.count(w) for w in pos_words)
        neg = sum(text.count(w) for w in neg_words)
        result["sentiment"] = {"positive_hits": pos, "negative_hits": neg,
                               "label": "偏正向" if pos > neg else "偏负向" if neg > pos else "中性"}
    elif algorithm == "spacy_ner":
        from .ner_service import extract_entities, model_available
        if not model_available():
            raise RuntimeError("Spacy NER 模型文件缺失")
        entities = extract_entities(text)
        result["entities"] = [{"entity": e["text"], "type": e["label"],
                               "start": e["start"], "end": e["end"]} for e in entities]
    elif algorithm == "gliner_ner":
        from .gliner_service import extract_entities
        gliner_ents = extract_entities(text, params.get("entityTypes") or [],
                                       threshold=float(params.get("threshold", 0.3)))
        result["entities"] = [{"entity": e["text"], "type": e["type"], "start": e["start"],
                               "end": e["end"], "confidence": e.get("confidence", 0.0)}
                              for e in gliner_ents][:200]
    else:  # text_class
        categories = {"飞行日志": ["飞行", "高度", "速度", "航向"],
                      "设备记录": ["传感器", "设备", "状态"],
                      "告警信息": ["告警", "报警", "异常", "故障"]}
        hits = {cat: sum(text.count(kw) for kw in kws) for cat, kws in categories.items()}
        best = max(hits, key=hits.get) if any(hits.values()) else "未分类"
        result["classification"] = {"category": best, "hits": hits}
    return result


LABEL_ALGOS = {"voice": label_voice, "radar": label_radar, "image": label_image,
               "video": label_video, "text": label_text}

# 内置规则（与原型 BUILTIN_LABEL_RULES 对齐）
BUILTIN_RULES = {
    "voice": [
        {"key": "silence_split", "name": "静音分割", "desc": "基于音量阈值自动分割语音片段",
         "params": {"threshold": -40, "minDuration": 200, "maxDuration": 10000}, "applicable": ["WAV", "MP3", "FLAC"]},
        {"key": "silero_vad", "name": "Silero 语音活动检测", "desc": "Silero VAD 模型语音段检测（ONNX，需模型文件）",
         "params": {"threshold": 0.5, "min_speech_duration": 0.25}, "applicable": ["WAV", "MP3", "FLAC"]},
        {"key": "whisper_transcribe", "name": "Whisper 语音转写", "desc": "faster-whisper 整文件转写，输出带时间戳与置信度的文本段（v3.2）",
         "params": {"language": "zh"}, "applicable": ["WAV", "MP3", "FLAC", "M4A"]},
        {"key": "fixed_slice", "name": "固定窗口切片", "desc": "按固定时长切片标注",
         "params": {"sliceDuration": 10, "overlap": 0.5}, "applicable": ["WAV", "MP3", "FLAC"]},
        {"key": "energy_gate", "name": "能量门限检测", "desc": "基于短时能量检测有效语音段",
         "params": {"energyThreshold": 0.1, "hangover": 5}, "applicable": ["WAV", "MP3"]},
        {"key": "pitch_track", "name": "基频跟踪", "desc": "基于基频变化检测语音边界",
         "params": {"pitchRange": [80, 400], "sensitivity": 0.8}, "applicable": ["WAV"]},
    ],
    "radar": [
        {"key": "cfar", "name": "CFAR目标检测", "desc": "恒虚警率检测算法",
         "params": {"guardCells": 4, "trainingCells": 16, "pfa": 1e-6}, "applicable": ["CSV", "PLY"]},
        {"key": "intensity_threshold", "name": "强度阈值", "desc": "基于RCS强度阈值检测目标点",
         "params": {"rcsThreshold": 15, "rangeMax": 500}, "applicable": ["CSV", "PLY"]},
        {"key": "clustering", "name": "点云聚类", "desc": "DBSCAN聚类算法分割目标",
         "params": {"eps": 2.5, "minPts": 5}, "applicable": ["CSV", "PLY"]},
        {"key": "doppler_filter", "name": "多普勒滤波", "desc": "基于速度信息过滤静止目标",
         "params": {"velocityMin": 0.5, "velocityMax": 100}, "applicable": ["CSV", "PLY"]},
    ],
    "image": [
        {"key": "coco_detect", "name": "语义目标检测（COCO）", "desc": "YOLOv8n 离线检测 80 类常见目标（人/车/飞机等），输出带置信度的标注框",
         "params": {"confThreshold": 0.4, "nmsThreshold": 0.45, "maxBoxes": 300}, "applicable": ["JPG", "JPEG", "PNG", "BMP", "TIFF"]},
        {"key": "edge_detect", "name": "边缘检测预框", "desc": "Sobel/Canny边缘检测生成候选框",
         "params": {"edgeThreshold": 50, "minArea": 100}, "applicable": ["JPG", "PNG", "TIFF"]},
        {"key": "color_cluster", "name": "色彩聚类分割", "desc": "K-Means色彩聚类自动分割区域",
         "params": {"k": 5, "maxIter": 100}, "applicable": ["JPG", "PNG"]},
        {"key": "template_match", "name": "模板匹配", "desc": "基于预定义模板匹配目标",
         "params": {"threshold": 0.8, "scaleRange": [0.5, 2]}, "applicable": ["JPG", "PNG", "TIFF"]},
        {"key": "contour_detect", "name": "轮廓检测", "desc": "基于轮廓提取生成标注框",
         "params": {"minArea": 200, "approxEpsilon": 0.02}, "applicable": ["JPG", "PNG"]},
        {"key": "sam3_detect", "name": "SAM 3 开放词汇分割", "desc": "文本概念提示分割任意目标，掩膜转外接框；需 GPU，评估项（v3.2）",
         "params": {"prompt": "", "confThreshold": 0.5}, "applicable": ["JPG", "JPEG", "PNG", "BMP", "TIFF"]},
    ],
    "video": [
        {"key": "coco_detect", "name": "语义目标检测（COCO）", "desc": "抽帧运行 YOLOv8n 检测 80 类目标，框回写到对应时间点",
         "params": {"confThreshold": 0.4, "nmsThreshold": 0.45, "frameInterval": 2, "maxFrames": 200}, "applicable": ["MP4", "AVI", "MOV", "MKV"]},
        {"key": "keyframe_extract", "name": "关键帧提取", "desc": "基于帧差法检测场景变化",
         "params": {"threshold": 8, "minInterval": 1}, "applicable": ["MP4", "AVI"]},
        {"key": "motion_detect", "name": "运动检测", "desc": "基于背景差分检测运动目标",
         "params": {"sensitivity": 5, "minArea": 500}, "applicable": ["MP4", "AVI"]},
        {"key": "object_track", "name": "目标跟踪", "desc": "基于光流的目标跟踪标注",
         "params": {"maxDistance": 50, "maxLost": 10}, "applicable": ["MP4"]},
        {"key": "scene_segment", "name": "场景分段", "desc": "基于颜色直方差分段",
         "params": {"histThreshold": 0.7, "minDuration": 2}, "applicable": ["MP4", "AVI"]},
        {"key": "sam2_video", "name": "SAM 2 视频分割", "desc": "文本或点提示的掩膜贯穿全片分割；二期，需 GPU（v3.2）",
         "params": {"prompt": "", "frameInterval": 2}, "applicable": ["MP4", "AVI", "MOV", "MKV"]},
    ],
    "text": [
        {"key": "keyword_extract", "name": "关键词提取", "desc": "基于TF-IDF提取关键词",
         "params": {"topK": 10, "minFreq": 2}, "applicable": ["TXT", "CSV", "JSON"]},
        {"key": "entity_recog", "name": "实体识别规则", "desc": "基于正则的实体识别",
         "params": {"entityTypes": ["人名", "地名", "机构"]}, "applicable": ["TXT", "CSV", "JSON"]},
        {"key": "sentiment_rule", "name": "情感规则标注", "desc": "基于情感词典的规则标注",
         "params": {"positiveDict": "pos.txt", "negativeDict": "neg.txt"}, "applicable": ["TXT", "CSV"]},
        {"key": "text_class", "name": "文本分类规则", "desc": "基于关键词匹配的文本分类",
         "params": {"categories": 5, "threshold": 0.6}, "applicable": ["TXT", "CSV", "JSON"]},
        {"key": "gliner_ner", "name": "GliNER 零样本实体识别", "desc": "任意实体类型名直接抽取，开放词汇无需训练；需模型（v3.2）",
         "params": {"entityTypes": ["人名", "地名", "机构"], "threshold": 0.3}, "applicable": ["TXT", "CSV", "JSON"]},
    ],
}


# Only advertise implemented, verifiable operations.
_DISABLED_ALGOS = {'template_match', 'sam3_detect', 'sam2_video', 'pitch_track',
                   'energy_gate', 'object_track', 'scene_segment', 'doppler_filter', 'cfar', 'clustering'}
BUILTIN_RULES = {k: [r for r in rules if r['key'] not in _DISABLED_ALGOS]
                 for k, rules in BUILTIN_RULES.items()}
for rules in BUILTIN_RULES.values():
    for rule in rules:
        if rule['key'] == 'keyword_extract': rule['desc'] = '按词频和词长提取关键词'
        if rule['key'] == 'keyframe_extract': rule['name'] = '画面变化时间点'
        if rule['key'] == 'motion_detect': rule['name'] = '帧差运动时间点'
        if rule['key'] == 'color_cluster': rule['name'] = '色彩区域候选框'

# ==================== 数据清洗 ====================
def clean_file(path: Path, rules: dict, dst_dir: Path) -> dict:
    """单文件清洗，返回 {removed_duplicates, removed_anomalies, removed_missing}。
    dst_dir 可为版本目录内的相对子目录（保持文件夹上传的目录结构）。"""
    stats = {"removed_duplicates": 0, "removed_anomalies": 0, "removed_missing": 0}
    suffix = path.suffix.lower()
    dst_dir.mkdir(parents=True, exist_ok=True)
    out = dst_dir / path.name
    if suffix in (".json", ".xml", ".csv"):
        # Structured records must not be deduplicated as physical lines.
        if suffix == '.json': json.loads(path.read_bytes())
        elif suffix == '.xml':
            import xml.etree.ElementTree as ET
            ET.parse(path)
        shutil.copy2(path, out)
    elif suffix == ".txt":
        from .label_transform import read_text
        text = read_text(path)
        lines = text.splitlines()
        before = len(lines)
        if rules.get("missing", True):  # 缺失值处理：去除空行
            lines = [l for l in lines if l.strip()]
            stats["removed_missing"] = before - len(lines)
        if rules.get("duplicate", True):  # 重复清理
            seen, deduped = set(), []
            for l in lines:
                if l not in seen:
                    seen.add(l)
                    deduped.append(l)
            stats["removed_duplicates"] = len(lines) - len(deduped)
            lines = deduped
        if rules.get("anomaly", True) and suffix == ".csv":  # 异常值检测（数值列 IQR）
            lines = _remove_csv_anomalies(lines, stats)
        out.write_text("\n".join(lines), encoding="utf-8")
    elif suffix in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
        from PIL import Image
        try:
            with Image.open(path) as im:
                im.verify()  # 损坏文件剔除
            shutil.copy2(path, out)
        except Exception:
            stats["removed_anomalies"] = 1
            return stats
    else:  # 音频/视频/雷达：按大小与可读性校验，原样保留有效文件
        if path.stat().st_size == 0:
            stats["removed_missing"] = 1
            return stats
        shutil.copy2(path, out)
    return stats


def _remove_csv_anomalies(lines: list, stats: dict) -> list:
    if len(lines) < 10:
        return lines
    try:
        header, rows = lines[0], lines[1:]
        cols = list(zip(*[r.split(",") for r in rows if "," in r]))
        numeric_cols = []
        for ci, col in enumerate(cols):
            try:
                vals = np.array([float(v) for v in col])
                numeric_cols.append((ci, vals))
            except ValueError:
                continue
        if not numeric_cols:
            return lines
        keep = np.ones(len(rows), dtype=bool)
        parsed = [r.split(",") for r in rows]
        for ci, vals in numeric_cols:
            q1, q3 = np.percentile(vals, [25, 75])
            iqr = q3 - q1
            lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
            col_vals = np.array([float(p[ci]) for p in parsed]) if len(parsed) == len(vals) else vals
            keep &= (col_vals >= lo) & (col_vals <= hi)
        removed = int((~keep).sum())
        stats["removed_anomalies"] += removed
        return [header] + [r for r, k in zip(rows, keep) if k]
    except Exception:
        return lines


# ==================== 数据转换 ====================
def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def transform_file(path: Path, transform_type: str, target_format: str,
                   params: dict, dst_dir: Path) -> Optional[Path]:
    suffix = path.suffix.lower()
    dst_dir.mkdir(parents=True, exist_ok=True)  # dst_dir 可为版本目录内的相对子目录
    if transform_type == "format":
        return _convert_format(path, suffix, target_format.lower(), dst_dir)
    if transform_type == "standardize":
        return _standardize(path, suffix, params, dst_dir)
    if transform_type == "feature":
        return _extract_feature(path, suffix, params, dst_dir)
    raise ValueError("暂不支持此转换类型")


def _convert_format(path: Path, suffix: str, target: str, dst_dir: Path) -> Optional[Path]:
    out = dst_dir / f"{path.stem}.{target}"
    audio_exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    if suffix in audio_exts and f".{target}" in audio_exts:
        subprocess.run([ffmpeg_exe(), "-y", "-i", str(path), str(out)],
                       capture_output=True, timeout=300, check=True)
        return out
    if suffix in image_exts and f".{target}" in image_exts:
        from PIL import Image
        with Image.open(path) as im:
            im.convert("RGB").save(out)
        return out
    if suffix == ".csv" and target == "ply":
        from .file_service import _load_radar_points
        pts = _load_radar_points(path, max_points=10_000_000)
        with open(out, "w", encoding="utf-8") as f:
            f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\n"
                    "property float x\nproperty float y\nproperty float z\n"
                    + ("property float intensity\n" if pts.shape[1] > 3 else "") + "end_header\n")
            for row in pts:
                f.write(" ".join(str(v) for v in row) + "\n")
        return out
    if suffix == ".ply" and target == "csv":
        from .file_service import _load_radar_points
        pts = _load_radar_points(path, max_points=10_000_000)
        with open(out, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "z"] + (["intensity"] if pts.shape[1] > 3 else []))
            writer.writerows(pts.tolist())
        return out
    raise ValueError(f"暂不支持 {suffix} → {target} 格式转换")


def _standardize(path: Path, suffix: str, params: dict, dst_dir: Path) -> Path:
    method = params.get("method", "zscore")
    if suffix in (".csv", ".txt", ".ply"):
        raise ValueError("暂不开放文本和点云数值标准化，避免改变标签坐标及字段含义")
    if suffix in (".csv", ".txt"):
        try:
            arr = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
            if method == "minmax":
                rng = np.ptp(arr, axis=0)
                rng[rng == 0] = 1
                arr = (arr - arr.min(axis=0)) / rng
            else:  # zscore
                std = arr.std(axis=0)
                std[std == 0] = 1
                arr = (arr - arr.mean(axis=0)) / std
            out = dst_dir / path.name
            header = ",".join(f"col_{i}" for i in range(arr.shape[1]))
            np.savetxt(out, arr, delimiter=",", header=header, comments="")
            return out
        except Exception:
            shutil.copy2(path, dst_dir / path.name)
            return dst_dir / path.name
    if suffix in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
        from PIL import Image
        size = int(params.get("size", 640))
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((size, size))
            out = dst_dir / path.name
            im.save(out)
            return out
    if suffix in (".wav", ".mp3", ".flac"):
        import soundfile as sf
        data, sr = sf.read(str(path), dtype="float32")
        peak = np.abs(data).max()
        target_peak = float(params.get("target_peak", 0.89))  # 约 -1dBFS 响度归一
        if peak > 0:
            data = data / peak * target_peak
        out = dst_dir / f"{path.stem}.wav"
        sf.write(str(out), data, sr)
        return out
    raise ValueError("此文件类型不支持标准化")


def _extract_feature(path: Path, suffix: str, params: dict, dst_dir: Path) -> Path:
    out = dst_dir / f"{path.stem}_features.json"
    features: dict = {"source": path.name}
    if suffix in (".wav", ".mp3", ".flac"):
        features.update(_mfcc(path, int(params.get("n_mfcc", 13))))
    elif suffix in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # HOG 简化：梯度方向直方图 9 bins
            gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
            mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
            hist, _ = np.histogram(ang[mag > mag.mean()], bins=9, range=(0, 360))
            features["hog"] = hist.tolist()
            features["hog_dim"] = 9
    elif suffix in (".csv", ".txt"):
        text = path.read_bytes().decode("utf-8", errors="ignore")
        words = re.findall(r"[A-Za-z]{2,}|[一-鿿]{2,}", text)
        freq = Counter(words)
        vocab = sorted(freq, key=freq.get, reverse=True)[:50]
        vec = [freq[w] for w in vocab]
        features["tfidf_vocab"] = vocab
        features["tfidf_vector"] = vec
    elif suffix == ".ply" or (suffix == ".csv" and params.get("radar")):
        from .file_service import _load_radar_points
        pts = _load_radar_points(path)
        if len(pts):
            if pts.shape[1] < 4: raise ValueError("强度检测需要真实 intensity 字段，不能用坐标代替")
            intensity = pts[:, 3]
            features["rcs_features"] = {
                "mean": round(float(intensity.mean()), 3), "std": round(float(intensity.std()), 3),
                "max": round(float(intensity.max()), 3), "point_count": int(len(pts))}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=2)
    return out


def _mfcc(path: Path, n_mfcc: int) -> dict:
    """MFCC：优先 librosa；不可用时用 soundfile+numpy 自实现（决策⑥）。"""
    try:
        import librosa
        y, sr = librosa.load(str(path), sr=None, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return {"mfcc_mean": np.round(mfcc.mean(axis=1), 4).tolist(),
                "mfcc_std": np.round(mfcc.std(axis=1), 4).tolist(),
                "n_mfcc": n_mfcc, "sample_rate": sr, "engine": "librosa"}
    except ImportError:
        import soundfile as sf
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        frame, hop = 1024, 512
        frames = [mono[i:i + frame] * np.hanning(frame)
                  for i in range(0, len(mono) - frame, hop)][:200]
        if not frames:
            return {"mfcc_mean": [], "engine": "numpy"}
        spec = np.abs(np.fft.rfft(np.array(frames), axis=1))
        log_spec = np.log(spec + 1e-8)
        # DCT 取前 n_mfcc 系数近似 MFCC
        coeffs = np.zeros((len(frames), n_mfcc))
        for k in range(n_mfcc):
            coeffs[:, k] = (log_spec * np.cos(np.pi * k * (np.arange(log_spec.shape[1]) + 0.5) / log_spec.shape[1])).sum(axis=1)
        return {"mfcc_mean": np.round(coeffs.mean(axis=0), 4).tolist(),
                "mfcc_std": np.round(coeffs.std(axis=0), 4).tolist(),
                "n_mfcc": n_mfcc, "sample_rate": sr, "engine": "numpy-dct"}


# ==================== 数据采样 ====================
def sample_split(files: list[Path], train_ratio: int, val_ratio: int, test_ratio: int,
                 method: str = "random") -> tuple[list, list, list]:
    if train_ratio + val_ratio + test_ratio != 100:
        raise ValueError("训练/验证/测试比例之和必须等于 100%")
    if method != "random": raise ValueError("当前仅支持随机划分")
    if any(r < 0 or r > 100 for r in (train_ratio,val_ratio,test_ratio)): raise ValueError("划分比例无效")
    files = sorted(files)
    if method == "time":  # 时间序列划分：按修改时间排序
        files.sort(key=lambda p: p.stat().st_mtime)
    elif method == "stratified":  # 分层采样：按扩展名分层后各层同比例划分
        by_ext: dict = {}
        for p in files:
            by_ext.setdefault(p.suffix.lower(), []).append(p)
        train, val, test = [], [], []
        for group in by_ext.values():
            random.Random(42).shuffle(group)
            n = len(group)
            n_train = round(n * train_ratio / 100)
            n_val = round(n * val_ratio / 100)
            train += group[:n_train]
            val += group[n_train:n_train + n_val]
            test += group[n_train + n_val:]
        return train, val, test
    else:
        random.Random(42).shuffle(files)
    n = len(files)
    n_train = round(n * train_ratio / 100)
    n_val = round(n * val_ratio / 100)
    return files[:n_train], files[n_train:n_train + n_val], files[n_train + n_val:]


# ==================== 任务执行引擎（后台线程 + 进度写库） ====================
def _run_in_thread(fn, task_id: int) -> None:
    t = threading.Thread(target=fn, args=(task_id,), daemon=True,
                         name=f"task-{task_id}")
    t.start()


def _finish(db, task, ok: bool, error: str = "") -> None:
    if ok and getattr(task, "output_version_id", None):
        version = db.get(DatasetVersion, task.output_version_id)
        if version and not data_files(Path(version.file_path)):
            ok, error = False, "处理结果为空，请调整规则后重试"
    task.status = "completed" if ok else "failed"
    task.completed_at = datetime.now()
    if hasattr(task, "progress"):
        task.progress = 100 if ok else task.progress
    if hasattr(task, "error_msg"):
        task.error_msg = error
    db.commit()
    if ok and getattr(task, "output_version_id", None):
        version = db.get(DatasetVersion, task.output_version_id)
        if version: vs.refresh_version_stats(db, version)


def start_auto_label(task_id: int) -> None:
    _run_in_thread(_exec_auto_label, task_id)


def _exec_auto_label(task_id: int) -> None:
    db = SessionLocal()
    try:
        task = db.query(AutoLabelTask).get(task_id)
        task.status = "running"
        db.commit()
        ds = db.query(Dataset).get(json.loads(task.source_ids)[0])  # source_ids=[dataset_id]
        input_v = db.query(DatasetVersion).get(task.input_version_id)
        output_v = vs.create_output_version(db, ds, input_version=input_v,
                                            task_type="auto_label", task_id=task.id,
                                            user_id=task.created_by)
        in_dir, out_dir = Path(input_v.file_path), Path(output_v.file_path)
        copy_all(in_dir, out_dir)
        params = json.loads(task.rules_config) if task.rules_config else {}
        algo = LABEL_ALGOS[task.data_type]
        files = data_files(in_dir)
        if task.algorithm not in {r["key"] for r in BUILTIN_RULES.get(task.data_type, [])}:
            raise ValueError("算法不存在或尚未开放")
        if len({p.with_suffix("") for p in files}) != len(files): raise ValueError("同目录同名不同后缀文件的标注名冲突")
        ok_count, fail_count, first_err = 0, 0, ""
        for i, p in enumerate(files):
            try:
                label = algo(p, task.algorithm, params)
                if isinstance(label, dict) and label.get("error"):
                    raise ValueError(label["error"])
                write_label(out_dir, p, label, src_dir=in_dir)
                ok_count += 1
            except Exception as e:
                fail_count += 1
                if not first_err:
                    first_err = str(e)
                old_label = out_dir / p.relative_to(in_dir).with_name(p.stem + "_label.json")
                old_label.unlink(missing_ok=True)
            task.progress = round((i + 1) / max(len(files), 1) * 100)
            db.commit()
        task.output_version_id = output_v.id
        task.result_path = str(out_dir)
        vs.refresh_version_stats(db, output_v)
        if fail_count or not files:
            _finish(db, task, False, f"成功 {ok_count} / 失败 {fail_count}：{first_err or '无可处理文件'}"[:500])
        else:
            _finish(db, task, True)
    except Exception as e:
        db.rollback()
        task = db.query(AutoLabelTask).get(task_id)
        if task:
            _finish(db, task, False, str(e)[:500])
    finally:
        db.close()


def start_clean(task_id: int) -> None:
    _run_in_thread(_exec_clean, task_id)


def _exec_clean(task_id: int) -> None:
    db = SessionLocal()
    try:
        task = db.query(CleanTask).get(task_id)
        task.status = "running"
        db.commit()
        ds = db.query(Dataset).get(json.loads(task.source_ids)[0])
        input_v = db.query(DatasetVersion).get(task.input_version_id)
        output_v = vs.create_output_version(db, ds, input_version=input_v,
                                            task_type="clean", task_id=task.id,
                                            user_id=task.created_by)
        in_dir, out_dir = Path(input_v.file_path), Path(output_v.file_path)
        rules = json.loads(task.rules) if task.rules else {}
        files = data_files(in_dir)
        before = {"files": len(files),
                  "size": sum(p.stat().st_size for p in files)}
        total = {"removed_duplicates": 0, "removed_anomalies": 0, "removed_missing": 0}
        seen_hashes = set()
        for i, p in enumerate(files):
            if rules.get("duplicate", True):
                h = hashlib.md5(p.read_bytes()).hexdigest()
                if h in seen_hashes:
                    total["removed_duplicates"] += 1
                    continue
                seen_hashes.add(h)
            stats = clean_file(p, rules, out_dir / p.relative_to(in_dir).parent)
            for k in total:
                total[k] += stats[k]
            task.progress = round((i + 1) / max(len(files), 1) * 100)
            db.commit()
        from .label_transform import carry_label
        for p in files:
            target = out_dir / p.relative_to(in_dir)
            if target.is_file(): carry_label(p, target, text_change=p.suffix.lower() == '.txt')
        after_files = data_files(out_dir)
        task.stats_before = json.dumps(before, ensure_ascii=False)
        task.stats_after = json.dumps({"files": len(after_files),
                                       "size": sum(p.stat().st_size for p in after_files),
                                       **total}, ensure_ascii=False)
        task.output_version_id = output_v.id
        task.result_path = str(out_dir)
        vs.refresh_version_stats(db, output_v)
        _finish(db, task, True)
    except Exception as e:
        db.rollback()
        task = db.query(CleanTask).get(task_id)
        if task:
            _finish(db, task, False, str(e)[:500])
    finally:
        db.close()


def start_integrate(task_id: int) -> None:
    _run_in_thread(_exec_integrate, task_id)


def _exec_integrate(task_id: int) -> None:
    db = SessionLocal()
    try:
        task = db.query(IntegrateTask).get(task_id)
        task.status = "running"
        db.commit()
        ds = db.query(Dataset).get(json.loads(task.source_ids)[0])
        input_v = db.query(DatasetVersion).get(task.input_version_id)  # 主输入版本为父
        output_v = vs.create_output_version(db, ds, input_version=input_v,
                                            task_type="integrate", task_id=task.id,
                                            user_id=task.created_by)
        out_dir = Path(output_v.file_path)
        other_versions = json.loads(task.detail).get("other_input_version_ids", []) if task.detail else []
        all_vid = [input_v.id] + other_versions
        manifest = {"strategy": task.strategy, "combo_type": task.combo_type,
                    "sources": [], "fused_at": datetime.now().isoformat()}
        for vid in all_vid:
            v = db.query(DatasetVersion).get(vid)
            if not v:
                continue
            src = Path(v.file_path)
            src_ds = db.query(Dataset).get(v.dataset_id)
            src_type = src_ds.data_type if src_ds else ""
            sub = out_dir / (f"source_{src_type}_ds{v.dataset_id}_V{v.version_no}"
                             if src_type and src_type != ds.data_type
                             else f"source_ds{v.dataset_id}_V{v.version_no}")
            copy_all(src, sub)
            files = list_files(src)
            manifest["sources"].append({
                "dataset_id": v.dataset_id, "dataset_name": src_ds.name if src_ds else "",
                "data_type": src_type,
                "version": f"V{v.version_no}", "stage": v.stage,
                "file_count": len(files),
                "files": [p.name for p in files[:200]]})
        # 融合策略落盘说明：特征级/决策级/原始数据级/语义级
        if task.strategy != "raw": raise ValueError("当前仅支持文件集合并")
        manifest["strategy_desc"] = {
            "feature": "特征级融合：各来源特征向量拼接为统一特征空间",
            "decision": "决策级融合：各来源独立推理结果投票/加权合并",
            "raw": "原始数据级融合：原始文件按统一命名与目录组织合并",
            "semantic": "语义级融合：基于统一语义标准对齐标签与术语后合并",
        }.get(task.strategy, task.strategy)
        with open(out_dir / "_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        task.output_version_id = output_v.id
        task.result_path = str(out_dir)
        vs.refresh_version_stats(db, output_v)
        _finish(db, task, True)
    except Exception as e:
        db.rollback()
        task = db.query(IntegrateTask).get(task_id)
        if task:
            _finish(db, task, False, str(e)[:500])
    finally:
        db.close()


def start_transform(task_id: int) -> None:
    _run_in_thread(_exec_transform, task_id)


def _exec_transform(task_id: int) -> None:
    db = SessionLocal()
    try:
        task = db.query(TransformTask).get(task_id)
        task.status = "running"
        db.commit()
        ds = db.query(Dataset).get(task.source_id)
        input_v = db.query(DatasetVersion).get(task.input_version_id)
        output_v = vs.create_output_version(db, ds, input_version=input_v,
                                            task_type="transform", task_id=task.id,
                                            user_id=task.created_by)
        in_dir, out_dir = Path(input_v.file_path), Path(output_v.file_path)
        params = json.loads(task.params) if task.params else {}
        files = data_files(in_dir)
        from .label_transform import carry_label
        if not files: raise ValueError('无可处理文件')
        targets = set()
        for i, p in enumerate(files):
            rel_dir = out_dir / p.relative_to(in_dir).parent
            expected = rel_dir / (p.stem + '.' + task.target_format.lower() if task.transform_type == 'format' else p.name)
            if expected.with_suffix('') in targets: raise ValueError('转换后文件名冲突')
            targets.add(expected.with_suffix(''))
            target = transform_file(p, task.transform_type, task.target_format, params, rel_dir)
            if not target or not target.is_file(): raise ValueError('转换未生成有效文件')
            scale = None
            if task.data_type == 'image':
                from PIL import Image
                with Image.open(p) as im: old_size = im.size
                with Image.open(target) as im: new_size = im.size
                if old_size != new_size: scale = (new_size[0]/old_size[0], new_size[1]/old_size[1])
            carry_label(p, target, image_scale=scale)
            task.progress = round((i + 1) / len(files) * 100)
            db.commit()
        task.output_version_id = output_v.id
        task.result_path = str(out_dir)
        vs.refresh_version_stats(db, output_v)
        _finish(db, task, True)
    except Exception as e:
        db.rollback()
        task = db.query(TransformTask).get(task_id)
        if task:
            _finish(db, task, False, str(e)[:500])
    finally:
        db.close()


def start_sample(task_id: int) -> None:
    _run_in_thread(_exec_sample, task_id)


def _exec_sample(task_id: int) -> None:
    db = SessionLocal()
    try:
        task = db.query(SampleDataset).get(task_id)
        task.status = "running"
        db.commit()
        ds = db.query(Dataset).get(task.source_id)
        input_v = db.query(DatasetVersion).get(task.input_version_id)
        output_v = vs.create_output_version(db, ds, input_version=input_v,
                                            task_type="sample", task_id=task.id,
                                            user_id=task.created_by)
        in_dir, out_dir = Path(input_v.file_path), Path(output_v.file_path)
        files = data_files(in_dir)
        train, val, test = sample_split(files, task.train_ratio, task.val_ratio,
                                        task.test_ratio)
        labels = {}
        for p in list_files(in_dir):
            if p.name.endswith("_label.json"):
                rel = str(p.relative_to(in_dir).with_suffix(""))
                labels[rel[:-len("_label")] if rel.endswith("_label") else rel] = p
        label_counts = {"train": 0, "val": 0, "test": 0}
        for group, name in ((train, "train"), (val, "val"), (test, "test")):
            sub = out_dir / name
            sub.mkdir(parents=True, exist_ok=True)
            for p in group:
                rel = p.relative_to(in_dir)
                tgt = sub / rel
                tgt.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, tgt)
                key = str(rel.with_suffix(""))
                if key in labels:  # 标注随样本复制
                    ltgt = sub / labels[key].relative_to(in_dir)
                    ltgt.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(labels[key], ltgt)
                    label_counts[name] += 1
        # 标注随行统计（前端任务列表可展示）
        task.train_count, task.val_count, task.test_count = len(train), len(val), len(test)
        task.train_path = str(out_dir / "train")
        task.val_path = str(out_dir / "val")
        task.test_path = str(out_dir / "test")
        task.output_version_id = output_v.id
        task.result_path = str(out_dir)
        vs.refresh_version_stats(db, output_v)
        _finish(db, task, True)
        # error_msg 字段复用为标注随行统计信息（采样成功时无错误）
        task.error_msg = (f"标注随行 train {label_counts['train']}/val {label_counts['val']}"
                          f"/test {label_counts['test']}（共 {len(labels)} 个标注文件）")
        db.commit()
    except Exception as e:
        db.rollback()
        task = db.query(SampleDataset).get(task_id)
        if task:
            _finish(db, task, False, str(e)[:500])
    finally:
        db.close()


def start_workflow(wf_id: int) -> None:
    _run_in_thread(_exec_workflow, wf_id)


def _exec_workflow(wf_id: int) -> None:
    """工作流 = 预处理任务节点的顺序执行链（决策⑳）：
    节点类型限定已有任务类型；上游输出版本自动作为下游输入；失败即中止。
    """
    db = SessionLocal()
    try:
        wf = db.query(Workflow).get(wf_id)
        wf.status = "running"
        db.commit()
        nodes = json.loads(wf.nodes) if wf.nodes else []
        ds = db.query(Dataset).get(wf.source_id) if wf.source_id else None
        if not ds:
            raise ValueError("工作流未关联数据集")
        current_vid = wf.input_version_id or ds.current_version_id
        for node in nodes:
            ntype = node.get("type")
            if ntype == "clean":
                task = CleanTask(name=f"{wf.name}-清洗", data_type=ds.data_type,
                                 source_ids=json.dumps([ds.id]),
                                 rules=json.dumps(node.get("params", {})),
                                 input_version_id=current_vid, created_by=wf.created_by)
                db.add(task); db.commit()
                _exec_clean_inline(task.id)
                db.expire_all()  # 内联执行用的是独立会话，先清缓存再读最新状态
                task = db.query(CleanTask).get(task.id)
            elif ntype == "auto_label":
                task = AutoLabelTask(name=f"{wf.name}-自动标注", data_type=ds.data_type,
                                     source_ids=json.dumps([ds.id]),
                                     algorithm=node.get("algorithm", ""),
                                     rules_config=json.dumps(node.get("params", {})),
                                     input_version_id=current_vid, created_by=wf.created_by)
                db.add(task); db.commit()
                _exec_auto_label_inline(task.id)
                db.expire_all()  # 内联执行用的是独立会话，先清缓存再读最新状态
                task = db.query(AutoLabelTask).get(task.id)
            elif ntype == "transform":
                task = TransformTask(name=f"{wf.name}-转换", data_type=ds.data_type,
                                     source_id=ds.id,
                                     transform_type=node.get("transform_type", "format"),
                                     target_format=node.get("target_format", ""),
                                     params=json.dumps(node.get("params", {})),
                                     input_version_id=current_vid, created_by=wf.created_by)
                db.add(task); db.commit()
                _exec_transform_inline(task.id)
                db.expire_all()  # 内联执行用的是独立会话，先清缓存再读最新状态
                task = db.query(TransformTask).get(task.id)
            elif ntype == "sample":
                p = node.get("params", {})
                task = SampleDataset(name=f"{wf.name}-采样", data_type=ds.data_type,
                                     source_id=ds.id,
                                     train_ratio=int(p.get("train_ratio", 70)),
                                     val_ratio=int(p.get("val_ratio", 10)),
                                     test_ratio=int(p.get("test_ratio", 20)),
                                     input_version_id=current_vid, created_by=wf.created_by)
                db.add(task); db.commit()
                _exec_sample_inline(task.id)
                db.expire_all()  # 内联执行用的是独立会话，先清缓存再读最新状态
                task = db.query(SampleDataset).get(task.id)
            else:
                raise ValueError(f"不支持的节点类型：{ntype}")
            if task.status != "completed":
                raise ValueError(f"节点「{task.name}」执行失败：{getattr(task, 'error_msg', '')}")
            current_vid = task.output_version_id  # 上游输出版本 → 下游输入
        wf.output_version_id = current_vid
        _finish(db, wf, True)
    except Exception as e:
        db.rollback()
        wf = db.query(Workflow).get(wf_id)
        if wf:
            _finish(db, wf, False, str(e)[:500])
    finally:
        db.close()


# 工作流内部同步执行（复用各任务执行体，去掉线程包装）
_exec_auto_label_inline = _exec_auto_label
_exec_clean_inline = _exec_clean
_exec_transform_inline = _exec_transform
_exec_sample_inline = _exec_sample


def _run_silero_vad(file_path: Path, rules_config: dict) -> list[dict]:
    """Silero VAD 语音活动检测。模型失败时降级到 silence_split。"""
    try:
        import soundfile as sf
        from .vad_service import detect_segments, model_available

        if not model_available():
            raise RuntimeError("Silero VAD 模型文件缺失")

        waveform, sample_rate = sf.read(str(file_path), dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if sample_rate != 16000:
            n = len(waveform)
            new_n = int(n * 16000 / sample_rate)
            waveform = np.interp(
                np.linspace(0, n - 1, new_n),
                np.arange(n),
                waveform
            ).astype(np.float32)
            sample_rate = 16000

        segments = detect_segments(
            waveform, sample_rate,
            threshold=rules_config.get("threshold", 0.5),
            min_speech_duration=rules_config.get("min_speech_duration", 0.25)
        )
        return [{"type": "segment", "start": round(s["start"], 3),
                 "end": round(s["end"], 3), "label": "语音段",
                 "confidence": round(s.get("confidence", 0), 3)} for s in segments]

    except Exception as e:
        print(f"[降级] Silero VAD 失败（{e}），回退到静音分割规则")
        return _silence_split(file_path, {"threshold": -40, "minDuration": 0.2, "maxDuration": 10})


def _run_spacy_ner(file_path: Path, rules_config: dict) -> list[dict]:
    """Spacy 中文 NER 实体识别。模型失败时降级到 entity_recog。"""
    try:
        from .ner_service import extract_entities, model_available

        if not model_available():
            raise RuntimeError("Spacy NER 模型文件缺失")

        text = file_path.read_text(encoding="utf-8")
        entities = extract_entities(text)
        return [{"type": "span", "start": e["start"], "end": e["end"],
                 "text": e["text"], "label": e["label"]} for e in entities]

    except Exception as e:
        print(f"[降级] Spacy NER 失败（{e}），回退到正则实体识别规则")
        return _entity_recog(file_path, {})
