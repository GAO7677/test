"""只读启动前检查：不导入 app、不连接数据库、不启动服务、不加载模型。"""
import ast
import importlib
import json
import sys
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]


def main():
    failures = []
    for path in (ROOT / "backend" / "app").rglob("*.py"):
        try:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            failures.append(f"语法：{path}: {exc}")
    for name in ("fastapi", "uvicorn", "sqlalchemy", "pymysql", "alembic",
                 "jose", "cryptography", "bcrypt", "multipart", "pydantic",
                 "PIL", "numpy", "onnxruntime", "cv2", "soundfile", "imageio_ffmpeg"):
        try:
            importlib.import_module(name)
        except Exception as exc:
            failures.append(f"依赖：{name}: {exc}")
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8-sig"))
    for key in ("upload_dir", "log_dir"):
        value = config[key]
        if PureWindowsPath(value).drive or not Path(value).is_absolute():
            failures.append(f"{key} 必须为 Linux 绝对路径")
        print(f"[配置] {key}: {value}")
    for value in ("frontend/dist/index.html", "backend/models/yolov8n.onnx",
                  config["models"]["vad_path"], config["models"]["whisper_path"]):
        path = ROOT / value
        if not path.exists():
            failures.append(f"缺少文件：{path}")
    if not config["database"].get("password"):
        print("[待配置] config.json database.password；启动前填写数据库账号与密码。")
    print("[范围] 此命令不验证 MySQL 连通性、建表权限、HTTP 页面或模型推理。")
    print("[可选] Whisper/spaCy/librosa 未安装；GLiNER ONNX 加载与 SAM 推理未接入。")
    for failure in failures:
        print(f"[FAIL] {failure}")
    print(f"[结果] 启动前静态/依赖检查：{'FAIL' if failures else 'PASS'}")
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
