#!/usr/bin/env bash
set -euo pipefail
PLATFORM_MYSQL=/mnt/data/gaoya/agent-data/tools/mysql-platform
PLATFORM_OUTPUT=/mnt/data/gaoya/agent-data/outputs_v1/data_platform3_linux
if [ -f "$PLATFORM_OUTPUT/mysql/mysql.pid" ] && kill -0 "$(cat "$PLATFORM_OUTPUT/mysql/mysql.pid")" 2>/dev/null; then
  echo "MySQL instance is already running"
  exit 0
fi
export LD_LIBRARY_PATH="$PLATFORM_MYSQL/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$PLATFORM_MYSQL/bin/mysqld" --defaults-file="$PLATFORM_OUTPUT/mysql/my.cnf" --daemonize
