#!/usr/bin/env bash
# 前台运行；数据库配置确认后再执行。
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$PROJECT_ROOT/backend/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "缺少虚拟环境，请先按 LINUX.md 安装 backend/requirements-linux.lock.txt" >&2
  exit 1
fi
# 当前已实现模型使用 CPU，隔离全部 GPU（包括物理 GPU 4）。
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/mnt/data/gaoya/agent-data/cache/data_platform3_linux/tmp
mkdir -p "$TMPDIR"
cd "$PROJECT_ROOT/backend"
if [[ "${1:-}" == "--check" ]]; then
  exec "$PYTHON" "$PROJECT_ROOT/scripts/check_linux.py"
fi
if [[ $# -ne 0 ]]; then
  echo "用法：bash start.sh [--check]" >&2
  exit 2
fi
exec "$PYTHON" -m app.main
