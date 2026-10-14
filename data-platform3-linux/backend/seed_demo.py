"""演示数据生成脚本：向运行中的服务灌入五类样例数据（第 12 章交付清单）。

用法：先启动服务（python -m app.main），再执行：
    python seed_demo.py [--base http://localhost:8000/api] [--user sysadmin] [--password <密码>]

说明：为避免离线环境额外引入 requests 及其传递依赖，本脚本仅使用标准库 urllib/http.cookiejar 实现
登录 Cookie 保持、JSON POST 与 multipart 文件分片上传。
"""
from __future__ import annotations

import argparse
import hashlib
import http.cookiejar
import io
import json
import time
import urllib.error
import urllib.request
from urllib.parse import urljoin

import numpy as np
import soundfile as sf


class ApiClient:
    """最小 API 客户端：Cookie 登录态 + JSON/multipart POST + GET。"""

    def __init__(self, base: str, timeout: int = 60):
        self.base = base.rstrip("/") + "/"
        self.timeout = timeout
        self.cookies = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.cookies)
        )

    def _request(self, method: str, path: str, data: bytes | None = None,
                 headers: dict | None = None) -> bytes:
        url = urljoin(self.base, path.lstrip("/"))
        req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
        try:
            with self.opener.open(req, timeout=self.timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code} {method} {path}: {body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"连接失败 {method} {path}: {e.reason}") from e

    def get_json(self, path: str) -> dict:
        raw = self._request("GET", path)
        return json.loads(raw.decode("utf-8") or "{}")

    def post_json(self, path: str, payload: dict) -> dict:
        raw = self._request(
            "POST", path,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        return json.loads(raw.decode("utf-8") or "{}")

    def post_multipart(self, path: str, fields: dict[str, str],
                       file_field: str, filename: str, raw: bytes) -> dict:
        boundary = "----DataPlatformSeedBoundary7d1a2f"
        body = io.BytesIO()
        for name, value in fields.items():
            body.write(f"--{boundary}\r\n".encode())
            body.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
            body.write(str(value).encode())
            body.write(b"\r\n")
        body.write(f"--{boundary}\r\n".encode())
        body.write(
            f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
            .encode()
        )
        body.write(b"Content-Type: application/octet-stream\r\n\r\n")
        body.write(raw)
        body.write(f"\r\n--{boundary}--\r\n".encode())
        resp = self._request(
            "POST", path, data=body.getvalue(),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        return json.loads(resp.decode("utf-8") or "{}")


def upload(client: ApiClient, dtype: str, rid: int, filename: str, raw: bytes) -> None:
    h = hashlib.md5(raw).hexdigest()
    r = client.post_json(f"/{dtype}/{rid}/upload/init", {
        "filename": filename, "size": len(raw), "chunk_total": 1, "file_hash": h,
    })
    fid = r["file_id"]
    client.post_multipart(f"/{dtype}/{rid}/upload/chunk",
                          {"file_id": str(fid), "chunk_index": "0"},
                          "chunk", filename, raw)
    client.post_json(f"/{dtype}/{rid}/upload/complete", {"file_id": fid})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000/api")
    ap.add_argument("--user", default="sysadmin")
    ap.add_argument("--password", required=True, help="登录密码（首次登录后已修改的密码）")
    args = ap.parse_args()

    client = ApiClient(args.base)
    client.post_json("/auth/login", {"username": args.user, "password": args.password})
    print("登录成功")

    # ---- 语音 ----
    r = client.post_json("/voice", {"name": "飞行通话录音", "format": "WAV",
                                    "project": "项目A", "scene": "机载场景", "secret_level": "秘密"})
    vid = r["id"]
    sr = 16000
    t = np.arange(sr * 3) / sr
    sig = (0.4 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)).astype("float32")
    buf = io.BytesIO()
    sf.write(buf, sig, sr, format="WAV")
    upload(client, "voice", vid, "flight_comm.wav", buf.getvalue())
    print(f"语音数据 #{vid} 已上传")

    # ---- 雷达 ----
    r = client.post_json("/radar", {"name": "机载雷达点云", "format": "CSV",
                                    "project": "项目B", "scene": "外场试验", "secret_level": "机密"})
    rid = r["id"]
    rng = np.random.default_rng(1)
    pts = np.concatenate([rng.normal(0, 50, (3000, 3)),
                          rng.normal(120, 15, (300, 3)),
                          rng.normal(-150, 20, (200, 3))])
    intensity = np.abs(pts[:, 2]) * 3 + rng.normal(20, 5, len(pts))
    csv_text = "x,y,z,intensity\n" + "\n".join(
        f"{p[0]:.2f},{p[1]:.2f},{p[2]:.2f},{i:.2f}" for p, i in zip(pts, intensity))
    upload(client, "radar", rid, "cloud.csv", csv_text.encode())
    print(f"雷达数据 #{rid} 已上传")

    # ---- 文本 ----
    r = client.post_json("/text", {"name": "飞行日志记录", "format": "TXT",
                                   "project": "项目A", "scene": "地面场景", "secret_level": "内部"})
    tid = r["id"]
    log_text = "\n".join(
        f"2026-07-{10 + i:02d} 08:{i:02d}:00 高度{8000 + i * 100}m 指示空速{450 + i}km/h 迎角{2 + i * 0.1:.1f}° 状态正常"
        for i in range(60))
    upload(client, "text", tid, "flight_log.txt", log_text.encode("utf-8"))
    print(f"文本数据 #{tid} 已上传")

    # ---- 数据集 + 自动标注 + 采样 + 质量评估 ----
    r = client.post_json("/datasets", {"name": "语音数据集A", "data_type": "voice",
                                       "source_data_ids": [vid]})
    dsid = r["id"]
    client.post_json("/auto-label", {"name": "静音分割标注", "data_type": "voice", "dataset_id": dsid,
                                     "algorithm": "silence_split", "rules_config": {}})
    time.sleep(2)
    client.post_json("/sample", {"name": "训练验证测试三划分", "data_type": "voice", "dataset_id": dsid,
                                 "train_ratio": 70, "val_ratio": 10, "test_ratio": 20})
    time.sleep(2)
    versions = client.get_json(f"/datasets/{dsid}/versions")["items"]
    if versions:
        q = client.post_json("/quality/evaluate", {"data_type": "voice", "data_id": dsid,
                                                   "version_id": versions[0]["id"]})
        print(f"数据集 #{dsid} 已创建，共 {len(versions)} 个版本，最新版本质量分 {q['overall_score']}")
    print("演示数据生成完成")


if __name__ == "__main__":
    main()
