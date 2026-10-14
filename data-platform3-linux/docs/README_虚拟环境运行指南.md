# 数据中台 v3.2 —— 虚拟环境（WSL2 Ubuntu）运行指南

> 适用对象：已部署在 Linux 虚拟环境（WSL2 Ubuntu 26.04）中的数据中台 v3.2。
> 运行副本：`/home/yyq/data-platform3`（WSL 原生文件系统）
> Windows 侧转换工程：`D:\data-platform3`（代码改写一律在此，原工程 `D:\data-platform2` 保持原样）
> 服务形态：systemd 服务 `data-platform.service`（已配置开机自启）

---

## 1. 环境一览

| 项目 | 内容 |
|---|---|
| 虚拟环境 | WSL2 Ubuntu 26.04.1 LTS（用户 `yyq`） |
| 项目路径 | `/home/yyq/data-platform3` |
| Python | 3.14.4（虚拟环境 `backend/.venv`） |
| MySQL | 8.4.11（root / 123456，库 `data_platform`，已迁移 13 数据集 / 3 用户 / alembic 0003） |
| 服务端口 | 8000（被占用时自动顺延 8001/8002） |
| 访问地址 | http://localhost:8000（Windows 浏览器直接访问） |

---

## 2. 快速启动（推荐）

已在 Windows PowerShell / CMD 中配置 systemd 开机自启，**电脑或 WSL 重启后服务会自动恢复**。若访问打不开，执行下面这一条命令拉起：

```powershell
wsl -d Ubuntu -u root -- systemctl start data-platform
```

启动后验证：

```powershell
wsl -d Ubuntu -- bash -lc "curl -s http://127.0.0.1:8000/api/health"
# 期望输出：{"status":"ok","version":"3.2.0"}
```

---

## 3. 手动启动（标准流程）

打开 Ubuntu 终端（Windows 终端 → Ubuntu 标签，或执行 `wsl -d Ubuntu`），依次执行：

```bash
cd /home/yyq/data-platform3          # 1. 进入项目目录
source backend/.venv/bin/activate    # 2. 激活 Python 虚拟环境
cd backend
nohup python -m app.main > ../DataPlatform/Logs/server_stdout.log 2>&1 &   # 3. 启动项目（后端自动托管前端）
curl http://localhost:8000/api/health    # 4. 验证：{"status":"ok","version":"3.2.0"}
```

预期终端输出（参考）：

```
python 解释器: /home/yyq/data-platform3/backend/.venv/bin/python
python 版本  : Python 3.14.4
INFO:     Started server process [xxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

> 说明：手动方式适用于调试时前台观察输出；日常运行推荐第 2 节的 systemd 方式（后台托管、崩溃自动重启）。

---

## 4. 状态 / 停止 / 重启 / 日志

以下命令在 Windows PowerShell / CMD 中直接执行：

| 操作 | 命令 |
|---|---|
| 查看服务状态 | `wsl -d Ubuntu -- bash -lc "systemctl is-active data-platform"` |
| 启动 | `wsl -d Ubuntu -u root -- systemctl start data-platform` |
| 停止 | `wsl -d Ubuntu -u root -- systemctl stop data-platform` |
| 重启 | `wsl -d Ubuntu -u root -- systemctl restart data-platform` |
| 查看运行日志 | `wsl -d Ubuntu -- bash -lc "tail -50 /home/yyq/data-platform3/DataPlatform/Logs/server_stderr.log"` |
| 查看启动日志 | `wsl -d Ubuntu -- bash -lc "tail -50 /home/yyq/data-platform3/DataPlatform/Logs/server_stdout.log"` |

---

## 5. 访问与账号

- 浏览器访问：**http://localhost:8000**
- 默认账号：

| 用户名 | 密码 | 角色 |
|---|---|---|
| sysadmin | 123456 | 系统管理员（全部权限） |
| admin | 未知（迁移自原环境） | 管理员 |
| guest | admin | 访客 |

---

## 6. 常见问题

| 现象 | 处理 |
|---|---|
| 页面打不开、health 无响应 | 先 `systemctl is-active data-platform`；非 active 则 `systemctl start data-platform` |
| WSL 被 `wsl --shutdown` 关闭后服务未恢复 | 重新进入 WSL（`wsl -d Ubuntu`）等待几秒，再 `systemctl start data-platform`；若仍失败检查 MySQL：`wsl -d Ubuntu -- bash -lc "systemctl is-active mysql"` |
| 8000 端口被占用 | 服务会自动顺延 8001/8002，启动日志中会打印实际地址；也可 `systemctl stop data-platform` 后清理占用进程再启动 |
| 日志出现 `WSL ERROR: UtilAcceptVsock ... accept4 failed` | WSL 已知无害噪音，不影响服务 |
| 手动 nohup 启动与 systemd 同时运行 | 会端口冲突（后者顺延端口）。日常只使用 systemd 一种方式，手动调试完请 kill 对应 PID |
| Whisper / GliNER / SAM 显示"模型未就绪" | 正常：增强模型依赖（torch 系）未安装，与 Windows 原环境一致；YOLO 检测、Silero VAD 可用 |

---

## 7. 重要目录与文件

| 路径 | 说明 |
|---|---|
| `/home/yyq/data-platform3/config.json` | 运行配置（Linux 路径版：upload_dir/log_dir/gliner_path） |
| `/home/yyq/data-platform3/DataPlatform/Uploads` | 真实业务数据（processed 1–13 等，**禁止写入测试文件**） |
| `/home/yyq/data-platform3/DataPlatform/Logs` | 运行日志 |
| `/home/yyq/data-platform3/backend/.venv` | WSL 专用虚拟环境（勿删、勿拷回 Windows） |
| `/home/yyq/data-platform3/frontend/dist` | 前端构建产物（后端启动即托管） |
| `/etc/systemd/system/data-platform.service` | systemd 服务定义（User=yyq，Restart=on-failure） |
| `D:\data-platform3\db_backup\` | 部署记录：`data_platform_dump.sql`（Windows 库备份）、`fix_paths.sql`（路径迁移）、`verify_deploy.sh`（部署验证）等 |

---

## 8. 维护注意

- 需要改写代码时，一律在 `D:\data-platform3` 修改，然后同步到 `/home/yyq/data-platform3` 生效；`D:\data-platform2` 保持原样。
- 数据库备份：`wsl -d Ubuntu -- bash -lc "mysqldump -uroot -p123456 --databases data_platform > /home/yyq/data_platform_backup_$(date +%Y%m%d).sql"`，数据文件备份即整个 `DataPlatform/Uploads` 目录。
- 增强模型（Whisper/GliNER）如需启用：在虚拟环境内安装 `faster-whisper`、`gliner` 及其 torch 依赖（体积较大），模型目录已就位于 `models/`。
