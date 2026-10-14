"""文件存储与解析服务：目录结构、格式白名单、元数据提取、预览与可视化数据（决策⑨⑩⑲）。

目录结构（第4章）：
  {upload_dir}/{type}/{dataset_id}/data|labels
  {upload_dir}/processed/{dataset_id}/V{n}/
"""
from __future__ import annotations

import base64
import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import settings

# 上传格式白名单（决策⑲）
ALLOWED_EXTENSIONS = {
    "voice": {".wav", ".mp3", ".flac", ".m4a", ".ogg"},
    "radar": {".csv", ".ply"},
    "image": {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".json"},   # .json 用于 COCO 整表标注随文件夹上传自动导入
    "video": {".mp4", ".avi", ".mov", ".mkv", ".json"},
    "text": {".txt", ".csv", ".json", ".xml"},
}

CHUNK_SIZE = 8 * 1024 * 1024  # 分片大小 8MB


def dataset_dir(data_type: str, dataset_id: int, stage: str = "data") -> Path:
    d = settings.upload_dir / data_type / str(dataset_id) / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def temp_dir() -> Path:
    d = settings.upload_dir / "temp"
    d.mkdir(parents=True, exist_ok=True)
    return d


def processed_dir(dataset_id: int, version_no: int) -> Path:
    d = settings.upload_dir / "processed" / str(dataset_id) / f"V{version_no}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def allowed_file(data_type: str, filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS.get(data_type, set())


# ---------------- 元数据提取 ----------------
def extract_metadata(data_type: str, path: Path) -> dict:
    """解析单个文件元数据，失败时返回空 dict（不阻塞上传）。"""
    try:
        if data_type == "voice":
            return _voice_meta(path)
        if data_type == "radar":
            return _radar_meta(path)
        if data_type == "image":
            return _image_meta(path)
        if data_type == "video":
            return _video_meta(path)
        if data_type == "text":
            return _text_meta(path)
    except Exception:
        return {}
    return {}


def _voice_meta(path: Path) -> dict:
    import soundfile as sf
    info = sf.info(str(path))
    return {"duration_seconds": round(info.duration, 2), "sample_rate": info.samplerate}


def _radar_meta(path: Path) -> dict:
    if path.suffix.lower() == ".csv":
        count = 0
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for _ in reader:
                count += 1
        return {"point_count": count}
    if path.suffix.lower() == ".ply":
        with open(path, "rb") as f:
            head = f.read(4096).decode("ascii", errors="ignore")
        for line in head.splitlines():
            if line.startswith("element vertex"):
                return {"point_count": int(line.split()[-1])}
    return {}


def _image_meta(path: Path) -> dict:
    with Image.open(path) as im:
        return {"width": im.width, "height": im.height}


def _video_meta(path: Path) -> dict:
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        import subprocess
        out = subprocess.run(
            [exe, "-i", str(path)], capture_output=True, text=True, timeout=30).stderr
        meta: dict = {}
        import re
        m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", out)
        if m:
            meta["duration_seconds"] = round(int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)), 2)
        m = re.search(r"(\d{2,5})x(\d{2,5})", out)
        if m:
            meta["width"], meta["height"] = int(m.group(1)), int(m.group(2))
        m = re.search(r"([\d.]+) fps", out)
        if m:
            meta["fps"] = float(m.group(1))
        return meta
    except Exception:
        try:
            import cv2
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            meta = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "fps": round(fps, 2)}
            if fps:
                meta["duration_seconds"] = round(frames / fps, 2)
            cap.release()
            return meta
        except Exception:
            return {}


def _text_meta(path: Path) -> dict:
    raw = path.read_bytes()
    encoding = "UTF-8"
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            text = raw.decode(enc)
            encoding = {"utf-8": "UTF-8", "gbk": "GBK"}.get(enc, "UTF-8")
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="ignore")
    lines = text.splitlines()
    return {"line_count": len(lines), "word_count": len(text.split()),
            "encoding": encoding}


# ---------------- 预览数据（决策⑨ / 8.6） ----------------
def preview_data(data_type: str, path: Path) -> dict:
    if data_type == "voice":
        return _voice_preview(path)
    if data_type == "radar":
        return _radar_preview(path)
    if data_type == "image":
        return _image_preview(path)
    if data_type == "video":
        return _video_preview(path)
    if data_type == "text":
        return _text_preview(path)
    return {}


def _load_audio(path: Path):
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data.mean(axis=1), sr  # 转单声道


def _voice_preview(path: Path) -> dict:
    data, sr = _load_audio(path)
    duration = len(data) / sr if sr else 0
    # 降采样波形点（约2000点）：分段取 min/max 包络
    n = 2000
    if len(data) > n:
        seg = len(data) // n
        trimmed = data[: seg * n].reshape(n, seg)
        points = [[round(float(trimmed[i].min()), 4), round(float(trimmed[i].max()), 4)]
                  for i in range(n)]
    else:
        points = [[round(float(v), 4), round(float(v), 4)] for v in data]
    return {"waveform": points, "duration": round(duration, 2), "sample_rate": sr}


def _load_radar_points(path: Path, max_points: int = 20000) -> np.ndarray:
    """加载点云为 Nx3+ 数组（CSV: x,y,z[,intensity...]；PLY: ascii 顶点）。"""
    if path.suffix.lower() == ".csv":
        rows = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            step = 1
            for row in reader:
                try:
                    rows.append([float(x) for x in row[:4]])
                except (ValueError, IndexError):
                    continue
        pts = np.array(rows, dtype=np.float32) if rows else np.zeros((0, 4), dtype=np.float32)
    else:  # PLY
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()
        start = next((i for i, l in enumerate(lines) if l.strip() == "end_header"), 0) + 1
        pts_list = []
        for line in lines[start:]:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    pts_list.append([float(x) for x in parts[:4]])
                except ValueError:
                    continue
        pts = np.array(pts_list, dtype=np.float32) if pts_list else np.zeros((0, 4), dtype=np.float32)
    if len(pts) > max_points:  # 大文件降采样
        idx = np.random.default_rng(42).choice(len(pts), max_points, replace=False)
        pts = pts[idx]
    return pts


def _radar_preview(path: Path) -> dict:
    pts = _load_radar_points(path)
    points = [[round(float(x), 3) for x in row] for row in pts[:5000]]
    return {"points": points, "point_count": int(len(pts)),
            "columns": ["x", "y", "z", "intensity"][: pts.shape[1] if len(pts) else 3]}


def _image_preview(path: Path) -> dict:
    with Image.open(path) as im:
        w, h = im.size
        im.thumbnail((400, 400))
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode()
    return {"thumbnail": f"data:image/jpeg;base64,{b64}", "width": w, "height": h}


def _video_preview(path: Path) -> dict:
    meta = _video_meta(path)
    thumb = ""
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        ok, frame = cap.read()
        if ok:
            frame = cv2.resize(frame, (400, int(400 * frame.shape[0] / frame.shape[1])))
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            thumb = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
        cap.release()
    except Exception:
        pass
    return {"thumbnail": thumb, **meta}


def _text_preview(path: Path) -> dict:
    meta = _text_meta(path)
    from .label_transform import read_text
    text = read_text(path)
    return {"text": text[:200000], "lines": text.splitlines()[:100],
            "line_count": meta["line_count"], "encoding": meta["encoding"]}



# ---------------- 可视化数据（决策⑩ / 8.3） ----------------
def viz_data(data_type: str, path: Path) -> dict:
    if data_type == "voice":
        return _voice_viz(path)
    if data_type == "radar":
        return _radar_viz(path)
    if data_type == "image":
        return _image_viz(path)
    if data_type == "video":
        return _video_viz(path)
    if data_type == "text":
        return _text_viz(path)
    return {}


def _voice_viz(path: Path) -> dict:
    data, sr = _load_audio(path)
    duration = len(data) / sr if sr else 0
    # 波形（2000 点包络）
    wave = _voice_preview(path)["waveform"]
    # 频谱（FFT 幅度谱，降采样至 512 点）
    spectrum = []
    if len(data) > 0:
        sp = np.abs(np.fft.rfft(data * np.hanning(len(data))))
        freqs = np.fft.rfftfreq(len(data), 1 / sr)
        n = 512
        if len(sp) > n:
            idx = np.linspace(0, len(sp) - 1, n).astype(int)
            sp, freqs = sp[idx], freqs[idx]
        db = 20 * np.log10(sp + 1e-8)
        spectrum = [[round(float(f), 1), round(float(v), 2)] for f, v in zip(freqs, db)]
    # 语谱图（STFT，压缩为 ~120 帧 x 64 频bin）
    spectrogram = []
    if len(data) > 2048:
        from numpy.lib.stride_tricks import sliding_window_view
        win, hop = 1024, 512
        frames_n = min((len(data) - win) // hop + 1, 120)
        windows = sliding_window_view(data[: win + (frames_n - 1) * hop], win)[::hop]
        spec = np.abs(np.fft.rfft(windows * np.hanning(win), axis=1))
        spec_db = 20 * np.log10(spec + 1e-8)
        bins = np.linspace(0, spec_db.shape[1] - 1, 64).astype(int)
        spectrogram = np.round(spec_db[:, bins], 1).tolist()
    # 音高曲线（自相关粗估，每秒一点）
    pitch = []
    if sr and len(data) > sr // 2:
        frame = sr // 10
        for i in range(0, len(data) - frame, frame):
            seg = data[i:i + frame]
            if np.sqrt(np.mean(seg ** 2)) < 0.01:
                pitch.append([round(i / sr, 1), 0])
                continue
            corr = np.correlate(seg, seg, mode="full")[len(seg) - 1:]
            lo, hi = int(sr / 400), int(sr / 80)
            if hi < len(corr):
                f0 = sr / (np.argmax(corr[lo:hi]) + lo)
                pitch.append([round(i / sr, 1), round(float(f0), 1)])
    return {"duration": round(duration, 2), "sample_rate": sr,
            "waveform": wave, "spectrum": spectrum,
            "spectrogram": spectrogram, "pitch": pitch[:600]}


def _radar_viz(path: Path) -> dict:
    pts = _load_radar_points(path)
    if len(pts) == 0:
        return {"bev": [], "range_doppler": [], "distribution": [], "trajectory": []}
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    intensity = pts[:, 3] if pts.shape[1] > 3 else np.sqrt(x ** 2 + y ** 2 + z ** 2)
    bev = [[round(float(a), 2), round(float(b), 2), round(float(v), 2)]
           for a, b, v in zip(x[:5000], y[:5000], intensity[:5000])]
    # 距离-多普勒（用 z 当作速度维近似展示；真实多普勒需原始回波，点云场景以 z/强度代替）
    rng = np.sqrt(x ** 2 + y ** 2)
    rd = [[round(float(r), 2), round(float(zz), 2), round(float(v), 2)]
          for r, zz, v in zip(rng[:5000], z[:5000], intensity[:5000])]
    hist, edges = np.histogram(rng, bins=30)
    distribution = [{"range": f"{edges[i]:.0f}-{edges[i+1]:.0f}", "count": int(hist[i])}
                    for i in range(len(hist))]
    order = np.argsort(rng)[:200]
    trajectory = [[round(float(x[i]), 2), round(float(y[i]), 2)] for i in order]
    return {"bev": bev, "range_doppler": rd, "distribution": distribution,
            "trajectory": trajectory, "point_count": int(len(pts))}


def _image_viz(path: Path) -> dict:
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        small = im.resize((min(w, 512), max(1, int(h * min(w, 512) / w))))
        arr = np.asarray(small)
        # RGB 直方图
        hist = {}
        for i, ch in enumerate(("r", "g", "b")):
            counts, _ = np.histogram(arr[:, :, i], bins=32, range=(0, 256))
            hist[ch] = counts.tolist()
        # 色彩热力图（亮度 16x16 网格）
        gray = np.asarray(small.convert("L"))
        gh, gw = gray.shape
        bh, bw = max(gh // 16, 1), max(gw // 16, 1)
        heat = [[round(float(gray[r:r + bh, c:c + bw].mean()), 1)
                 for c in range(0, gw, bw)][:16] for r in range(0, gh, bh)][:16]
        # 缩略图
        buf = io.BytesIO()
        im.resize((min(w, 600), max(1, int(h * min(w, 600) / w)))).save(buf, "JPEG", quality=80)
        thumb = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    return {"histogram": hist, "heatmap": heat, "thumbnail": thumb,
            "width": w, "height": h, "resolution": f"{w}x{h}"}


def _video_viz(path: Path) -> dict:
    meta = _video_meta(path)
    frames_info, motion, scene_changes = [], [], []
    thumbs = []
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(total // 12, 1) if total else 30
        prev_gray = None
        idx = 0
        max_scan = min(total, 300) if total else 300
        while idx < max_scan:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0 and len(thumbs) < 12:
                small = cv2.resize(frame, (160, 90))
                _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])
                thumbs.append({"frame": idx, "time": round(idx / fps, 1),
                               "image": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()})
            gray = cv2.cvtColor(cv2.resize(frame, (64, 36)), cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = float(np.mean(np.abs(gray.astype(float) - prev_gray.astype(float))))
                motion.append([round(idx / fps, 2), round(diff, 2)])
                if diff > 25:
                    scene_changes.append(round(idx / fps, 2))
            prev_gray = gray
            idx += 1
        cap.release()
    except Exception:
        pass
    return {"meta": meta, "frames": thumbs, "motion": motion,
            "scene_changes": scene_changes}


def _text_viz(path: Path) -> dict:
    meta = _text_meta(path)
    text = path.read_bytes().decode("utf-8", errors="ignore")
    lines = text.splitlines()
    # 长度分布
    lengths = [len(l) for l in lines if l.strip()]
    hist_data = []
    if lengths:
        hist, edges = np.histogram(lengths, bins=min(20, max(len(set(lengths)), 5)))
        hist_data = [{"range": f"{edges[i]:.0f}-{edges[i+1]:.0f}", "count": int(hist[i])}
                     for i in range(len(hist))]
    # 词频 TOP20
    import re
    from collections import Counter
    words = re.findall(r"[A-Za-z]{2,}|[一-鿿]{2,}", text)
    top = Counter(words).most_common(20)
    return {"length_distribution": hist_data,
            "top_words": [{"word": w, "count": c} for w, c in top],
            "line_count": meta["line_count"], "word_count": meta["word_count"],
            "encoding": meta["encoding"]}


# ---------------- 打包下载 ----------------
def zip_directory_stream(source: Path):
    """流式打包目录为 zip（生成器，用于 StreamingResponse）。"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(source.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(source))
    buf.seek(0)
    yield from buf
