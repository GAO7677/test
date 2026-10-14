"""PyInstaller 打包脚本（第 10 章）：

用法（Windows 目标机）：
    cd data-platform
    python -m venv .venv && .venv\\Scripts\\activate
    pip install -r backend/requirements.txt pyinstaller
    python build.py

产物：dist/DataPlatform/DataPlatform.exe（--onedir）
启动后自动：读取/生成 config.json → 初始化数据库与默认数据 → 启动服务并打开浏览器。
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST_STATIC = ROOT / "dist" / "static"


def main() -> None:
    # 自动构建前端并复制到 dist/static（v2.0：一键打包，无需手工 npm build）
    frontend = ROOT / "frontend"
    if not (frontend / "dist" / "index.html").exists():
        if (frontend / "node_modules").exists():
            print("[打包] 正在构建前端（npm run build）…")
            subprocess.run(["npm", "run", "build"], cwd=frontend, check=True, shell=True)
        else:
            print("[打包] 首次构建前端（npm install && npm run build）…")
            subprocess.run(["npm", "install"], cwd=frontend, check=True, shell=True)
            subprocess.run(["npm", "run", "build"], cwd=frontend, check=True, shell=True)
    if (frontend / "dist" / "index.html").exists():
        if DIST_STATIC.exists():
            shutil.rmtree(DIST_STATIC)
        shutil.copytree(frontend / "dist", DIST_STATIC)
    if not (DIST_STATIC / "index.html").exists():
        print("[错误] 未找到前端构建产物 dist/static/index.html")
        print("       请先执行：cd frontend && npm install && npm run build")
        sys.exit(1)

    spec_content = f"""# -*- mode: python ; coding: utf-8 -*-
# 自动生成：python build.py
a = Analysis(
    [r'{ROOT / "backend" / "app" / "main.py"}'],
    pathex=[r'{ROOT / "backend"}'],
    binaries=[],
    datas=[(r'{DIST_STATIC}', 'static'), (r'{ROOT / "backend" / "models"}', 'models')],
    hiddenimports=[
        'app.routers.auth', 'app.routers.users', 'app.routers.data',
        'app.routers.datasets', 'app.routers.tasks', 'app.routers.standards',
        'app.services.file_service', 'app.services.version_service',
        'app.services.preprocess', 'app.services.quality_service',
        'app.services.detector', 'app.services.vad_service', 'app.services.ner_service',
        'app.services.log_service', 'app.middleware', 'app.utils.common',
        'pymysql', 'soundfile', 'imageio_ffmpeg', 'cv2', 'PIL',
        'spacy', 'spacy.lang.zh', 'spacy.pipeline',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib'],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name='DataPlatform',
    console=True,          # 控制台窗口显示启动流程与实际端口
    icon=None,
)
coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False, upx=False,
    name='DataPlatform',
)
"""
    spec_path = ROOT / "DataPlatform.spec"
    spec_path.write_text(spec_content, encoding="utf-8")

    print("[打包] 正在执行 PyInstaller（--onedir --console）…")
    subprocess.run([sys.executable, "-m", "PyInstaller", "--clean", "--noconfirm",
                    "--collect-all", "spacy",
                    str(spec_path)], cwd=ROOT, check=True)
    print(f"[打包] 完成：{ROOT / 'dist' / 'DataPlatform' / 'DataPlatform.exe'}")


if __name__ == "__main__":
    main()
