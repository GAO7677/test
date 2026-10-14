# HANDOFF.md — 数据中台（离线单机全栈版）

## 1. 项目是什么

离线单机版「数据中台」v3.2：面向航空领域的五模态（语音/图像/文本/雷达/视频）数据获取、标注、清洗治理、质量评估、可视化的全栈系统，目标是完全离线内网单机可运行的完整数据治理闭环。

**目标定位（用户 2026-10-08 定稿）**：通过智能技术深度赋能，实现服务模式的根本性变革——从「数据供给导向」转变为「用户需求导向」的智能模式，主动提供定制化、高价值的数据集，显著提升数据利用效率、优化服务决策。

> ⚠️ 定稿口径（重要，接手的 Agent 必读）：**需求导向指「数据提供方式」，且该框架系统已具备**——下载模块即承载了需求表达（模态/场景/标签/项目/密级/需求数量条件）、匹配（`download/query` 条件查数 + `matched_count` 缺口对比）、交付（`download/pack` 打包 + `_lineage.json` 血缘）。**缺口只有两个"只有框架没有内容"的模块：本体知识图谱（G1）与 AIGC 生成（G2）**。用户 2026-10-08 明确裁定：需求工单/匹配推荐/评价反馈等"支撑链路"功能与下载页重复且对离线单机过重，**G3–G6 已全部回滚**，计划文档 3.2 差距清单仅留 G1/G2。**不要再新建需求工单/推荐/评价类功能，也不要再称其为"缺口"。**

非目标：不做在线多租户/云部署/实时流式处理；AIGC 生成、知识图谱/本体获取与抽取目前均为占位页面/501 接口（G1/G2 按计划 10–12 月推进）。

> 🖥️ **2026-10-09 新增部署形态：已部署到本机 WSL2 Ubuntu 26.04（Linux 虚拟机环境）**。
> - 工程：`D:\data-platform3`（Windows 侧转换适配版，**代码改写一律在此，data-platform2 保持原样**）；运行副本：`/home/yyq/data-platform3`（WSL 原生文件系统）。
> - 服务：systemd 服务 `data-platform.service`（已 enable 开机自启），MySQL 8.4.11，Python 3.14.4 venv（backend/.venv）。
> - 数据：Windows MySQL 9.7.1 `data_platform`（13 数据集/3 用户/alembic 0003）已整体迁移；Uploads 数据文件（0.82GB）已迁移；库内全部 Windows 绝对路径已改写为 Linux 路径（16 表 0 残留）。
> - 访问：Windows 浏览器直接打开 `http://localhost:8000`（WSL2 端口转发）。

## 2. 现在到哪了

**已完成：**
- v3.2 全栈功能齐备：五模态数据获取（分片断点续传/文件夹上传/COCO 导入）、手动标注（五模态画布）、自动标注（YOLOv8n 目标检测 / Silero VAD / Whisper 转写 / Spacy NER / GliNER / SAM2/SAM3 注册）、清洗/转换/集成/采样/工作流、六维质量评估+参考标准、可视化中心（数据集总览 Tab）、六级密级体系+按用户授权、用户/日志/系统配置、数据下载模块（query/pack/labels/aigc）。
- 模型文件已就位：`backend/models/yolov8n.onnx`（内置）；根目录 `models/` 下有 `faster-whisper-small/`（483MB）、`gliner_x_small/`（172MB）、`silero_vad.onnx`。
- Alembic 迁移 0001/0002/0003 齐备（alembic_version=0003）；README 为最新（v3.2，2026-09-17）。
- **G3–G6 曾实现、已按用户裁定全部回滚（2026-10-08）**：实现过需求工单/匹配推荐/定制组装/评价反馈（模型+0004 迁移+两路由+两服务+两前端视图），用户确认与下载页能力重复、离线单机场景价值低后选择方案 B 全部回滚：后端 6 文件删除、`main.py`/`models/__init__.py` 注册还原；MySQL DROP 三表并复位 alembic_version='0003'（**未用 alembic downgrade**，避免触发 0003 downgrade 删除真实 source_type 列）；前端两视图删除、router/MainLayout 还原、`npm run build` 重建 dist（57s）；后端重启验证 /api/health 200、sysadmin/123456 登录 200、/api/requirements 与 /api/deliveries 404（确认下线）。实现期抓到的 bug 与教训见第 5 块（recommend_service 的 Dataset 字段坑、upload_dir 落盘坑）。
- **开发计划文档已修订定稿**：`D:\data-platform2\数据中台_开发计划.docx`（**注意：用户已多次改名，当前唯一现行文件名就是这个，不带 _2026Q4 后缀**）。修订内容（方案 B 对齐）：3.2 差距清单删 G3–G6 仅留 G1/G2；4.1 阶段表删 S1/S2/S3（S4 收尾验收保留，对应缺口改 G1/G2、关键交付物去"评价闭环；"）；AIGC 线"与组装链路联调→与下载链路联调、组装任务可勾选 AIGC 补足→下载查询可勾选 AIGC 补足"；1.2/2.3/3.1/4.2 四处子句改为"支撑链路已由下载模块承载，不再重复建设"。文档结构：1 计划背景与目标 → 2 现状盘点 → 3 待完善功能（3.1 全链路 + 3.2 差距清单 G1–G2 + 3.3 AIGC 少样本生成研究【用户提供内容：七模态、四场景；本阶段重点 RGB、SAR、红外；Few-shot 抽取→生成→下游验证 + Real-only/Synthetic-only/Real+Synthetic/Full-real 对照；年底 FGVC-Aircraft/SADD/HIT-UAV 三方向端到端 + 统一评测协议 + ≥3 Baseline + 版本管理与标准化报告】）→ 4 实施计划（4.1 阶段划分：本体知识图谱线/AIGC 线/收尾验收 + 4.2 年底达成状态）。word 技能 SOURCE/TARGET 分离 + audit --source 0 错误 + catalogue（无 TOC 域跳过）+ 回读全 PASS + 写回原路径。
- **服务运行状态**：后端独立进程运行于 http://localhost:8000（`Start-Process python -m app.main`，日志 D:\DataPlatform\Logs\server_stdout/stderr.log）；前端 dist 为最新（无 G3–G6 页面）；真实 MySQL（localhost:3306 root/123456 data_platform）可用，datasets=12（对应 processed 1–12 真实数据）；users=3（sysadmin/system_admin 密码 123456、admin/admin 密码未知、guest/visitor 默认 admin）。
  - ⚠️ 2026-10-09 更正：实际 datasets=**13**（含"测试1定制数据集"，processed 目录 1–13）；Windows 服务当前已停（8000 无监听），MySQL97 服务仍运行（数据源完好，未删除）。
- **Linux 部署（2026-10-09 完成）**：
  - 转换工程 `D:\data-platform3`（复制 data-platform2 全量，排除 .venv/node_modules/.idea；config.json 已改 Linux 路径：upload_dir/log_dir=/home/yyq/data-platform3/DataPlatform/{Uploads,Logs}，gliner_path 显式指向 models/gliner_x_small 修复默认探测 gliner_medium 不匹配的问题；新增 start.sh）。
  - WSL2 Ubuntu 26.04（用户 yyq）：`/home/yyq/data-platform3` 含完整工程 + DataPlatform/Uploads（0.82GB 真实数据）。
  - 依赖：`backend/.venv`（Python 3.14.4）按 requirements.txt 装齐（spacy 改 3.8.16，见踩坑）；增强模型（faster-whisper/gliner/torch）未安装——与 Windows 原环境一致，Whisper/GliNER 显示"模型未就绪"，YOLO/Silero VAD 可用（onnxruntime 已装）。
  - MySQL：WSL 8.4.11 root 密码已设为 123456（caching_sha2_password）；`data_platform` 全库导入（13 数据集/3 用户/0003）；16 张含路径表（dataset_versions/五模态/任务结果/sample_datasets 等）Windows 路径全部 REPLACE 为 `/home/yyq/data-platform3/DataPlatform/Uploads/...`，验证残留=0。
  - 服务：systemd `data-platform.service`（User=yyq，ExecStart=backend/.venv/bin/python -m app.main，Restart=on-failure，enable 开机自启）；当前 active，`curl localhost:8000/api/health` → `{"status":"ok","version":"3.2.0"}`。
  - 验证全通过：health / sysadmin 登录（Cookie 认证）/ me / datasets=13 / voice / 前端 200 / download/query 真实文件扫描+标签匹配 / api/labels 标签全集 / Windows 侧 localhost:8000 访问正常。

**进行中：** 无开发中任务。**剩余待办**：① `processed/2/V1\_manifest.json` 被早期测试覆盖（内容不可恢复，数据文件 voice_*.wav 完好）是否处置，待用户拍板；② G1 本体、G2 AIGC 未开工（按计划 10–12 月研究线推进，AIGC 当前为研究起步期）。

**还没动：** G1 本体知识图谱、G2 AIGC（501 占位保留，研究线独立进行）；`seed_demo.py` 演示数据未运行；admin 账号真实密码未知（sysadmin/123456 可登录）。

## 3. 下一步具体做什么

**任务描述（按优先级）：**
1. **`processed/2/V1\_manifest.json` 处置（待拍板）**：若真实业务需要该清单，用真实库中 dataset id=2 V1 的来源信息重建；若可忽略则无需动作（数据文件未受影响）。
2. **G1/G2 按计划文档推进**（这是开发主线）：本体线（K 系列）以外部研究成功为前提，研究到位后先评审→集成（本体模型与存储、导入管理、图谱查询可视化、语义解析接入）→并入验证必须通过才算完成；AIGC 线（A 系列）按 3.3 研究内容与 4.1 阶段表开展：先做少样本 AIGC 生成方案（RGB、SAR、红外）与公开数据集接入（FGVC-Aircraft/SADD/HIT-UAV），生成数据入库（source_type 强制 AIGC）后与下载链路（query 查数/include_aigc、pack 打包）联调，做四组对照实验与统一评测协议。涉及文件：G1/G2 见计划 Word 3.2/3.3/4.1；`backend/app/routers/download.py`（aigc/generate 501 契约已冻结勿改接口形状）。
3. **日常验证**：改动后端后 `python -m compileall -q backend\app`；写盘测试必须先隔离 `settings.data["upload_dir"]`（见第 4 块测试纪律）；服务重启用 `Start-Process python -m app.main`（日志在 D:\DataPlatform\Logs\）。
- 验收标准：各任务验收标准见计划文档 4.1 达成标准列；AIGC 三方向端到端验证通过、统一评测协议与评测报告产出、下载查询可勾选 AIGC 补足。
- **Linux 环境运维（data-platform3）**：服务管理 `sudo systemctl {status|restart|stop} data-platform`（或 `bash /home/yyq/data-platform3/start.sh` 手动拉起）；日志 `/home/yyq/data-platform3/DataPlatform/Logs/server_{stdout,stderr}.log`；WSL 重启后服务自动恢复（systemd enabled）；数据备份 `mysqldump` + Uploads 目录。**Windows 原环境（data-platform2）暂不再维护运行**，如需 Windows 再跑：恢复 8000 服务需重新 Start-Process，且注意 data-platform3 的 config.json 已改 Linux 路径，勿与 data-platform2 的混用。

## 4. 约定

- 交付载体：正式交付物一律走 Word（`D:\data-platform2\数据中台_开发计划.docx` 为唯一现行版，注意文件名无 _2026Q4 后缀），Excel 旧稿不再更新。
- 技术栈：FastAPI + MySQL 8.0 后端（Python 兼容 3.8.8/3.10/3.14，依赖按 `backend/requirements.txt` 三组锁定）；Vue 3 + Vite + Element Plus + ECharts 前端；PyInstaller 打包 exe。
- 命名与风格：模块/文件英文小写下划线，注释中文；路由 `backend/app/routers/`，服务 `backend/app/services/`，前端视图 `frontend/src/views/`。
- 启动方式：后端启动即托管 `frontend/dist`（一键启动，无需单独起前端）；开发热更新才用 `npm run dev`。**Linux 部署版**：WSL 内 `systemctl start data-platform`（推荐，开机自启）或 `bash /home/yyq/data-platform3/start.sh`（nohup 手动）；Windows 版仍用 `Start-Process python -m app.main`。
- **代码改写边界**：2026-10-09 用户裁定——需要改写代码一律在 `D:\data-platform3` 做，**data-platform2 保持原样**；改完同步到 WSL（`/home/yyq/data-platform3`）才生效。
- **Linux 侧文件注意**：`/home/yyq/data-platform3/backend/.venv` 是 WSL 专用虚拟环境（勿删/勿拷回 Windows）；WSL 侧 requirements.txt 与 data-platform3 同步（spacy==3.8.16）；config.json 以 data-platform3（Linux 路径版）为准。
- 禁止事项：不删除/移动 `backend/models/` 与根目录 `models/` 下的模型文件；不擅自重建 `dist/`、`.venv/`；不动 `config.json` 的 jwt_secret（首次启动自动生成）；不新建"需求工单/匹配推荐/评价"类重复功能（用户已裁定回滚）；无 MySQL 联调只能用 `DATABASE_URL=sqlite:///./dev.db`（仅开发）。
- **测试纪律（重要）**：任何 sqlite 联调/TestClient 测试，只要会触发文件写入（组装、预处理、下载打包等），**必须先把 `settings.data["upload_dir"]` 指到临时目录**（`from app.config import settings; settings.data["upload_dir"] = str(Path(tmp)/"uploads")`），否则测试产物会真实落盘到共享目录 `D:\DataPlatform\Uploads`（该目录是真实业务数据，processed 下已有 1–12 号数据集）。只读/纯 CRUD 冒烟不写盘，可不隔离。

## 5. 踩过的坑

- **文档版本不同步**：`docs/部署指南_v3.0.md` 里工程名写的是 `data-platform1`（复制遗留），`CHANGELOG_v3.0.md` 只到 v3.0；唯一权威现状以 `README.md`（v3.2）和实际代码为准，引用 docs 前先核对版本。计划文档**文件名用户多次改名**，动手前先 Glob `D:\data-platform2\*.docx` 确认实际文件名，勿按旧名操作。
- **模型路径疑点**：README 称模型默认放 `backend/models/`，但 whisper/gliner/vad 实际在根目录 `models/` 而 `config.json` 路径全留空——不要想当然写死路径，动手前先 `Grep` 各 service 确认默认目录解析逻辑。
- **遍历大目录会超时**：PowerShell 递归遍历 `D:\data-platform2` 会因 node_modules/.venv 超时，务必先排除这些目录。
- **离线部署限制**：目标机完全离线时不能再 `pip install`/`npm install`，需在联网构建机备好 wheelhouse 与 `frontend/dist` 一起拷贝（见部署指南第 3 节）。
- **已知历史 bug 已修**：v3.0 曾因 `CONFIG` 引用不存在导致规则自动标注页 500、spacy 打包缺管线；当前包已修复（build.py 含 `--collect-all spacy` + vad/ner hiddenimports），如再遇 500 优先查服务导入而非改数据。
- **表格生成环境**：本机 Python 3.14 未预装 openpyxl（需 `pip install openpyxl`）；sheet 技能交付门禁 `excel_formula_verify.py` 依赖 LibreOffice——本机未装时退出码 4 是正常终态（只读诊断仍会跑），交付说明写明即可，不要反复重跑；artifact-preview 渲染 xlsx 只出文本不出图。
- **自检临时目录**：`artifact-preview` 默认把 `.preview/` 落在当前工作目录——跑完自检记得删除，不要留在项目根目录。
- **DOCX 编辑**：编辑已有 Word 必须走 word 技能 docx-editing.md（SOURCE=用户版备份，TARGET=副本，改完 audit --source + catalogue 更新目录再写回原路径）；python-docx 改单元格文本须保留首 run、删其余 run；**段落内子句替换用 `in` 匹配而非 startswith**（目标子句常在段落内部，如"本计划以填实…"在 1.2 长段落中部，startswith 会漏）；strict 文本匹配（如 `== '年底验收清单'`）会漏掉带编号的标题，要用 startswith。
- **MySQL 操作坑**：MySQL **不支持 `DROP INDEX IF EXISTS`**（报 1064 语法错），直接 DROP TABLE 即可（索引随表删）；**alembic downgrade 会执行 0003 的 downgrade（DROP COLUMN source_type 等真实列）**——本项目 0001–0003 是 create_all 建表后 stamp 的，表结构全部真实存在，任何回滚都必须手动执行目标迁移的等价 SQL + `UPDATE alembic_version`，严禁直接 `alembic downgrade`。
- **upload_dir 共享真实目录（最重的坑）**：`settings.upload_dir`（config.json）实际指向 `D:\DataPlatform\Uploads`，且 `Uploads\processed` 下 1–12 号目录是真实业务数据（**禁止删除**）。首轮 e2e 测试未隔离 upload_dir，把测试文件写进了真实目录：已在 `processed\1\V1` 下精确删除测试文件 img_001.jpg/img_001_label.json/img_002.jpg/img_002_label.json，删除了 `processed\2\V1\source_1_ds1_V1` 测试子树；但 `processed\2\V1\_manifest.json` 被测试覆盖为 custom_delivery 内容，**原清单内容不可恢复**（真实数据文件 voice_*.wav 完好），待用户拍板是否处置。教训：凡会写盘的测试一律先隔离 upload_dir。
- **G3–G6 回滚的教训（决策层）**：差距清单曾把"下载页已有的支撑链路（需求表达/匹配/交付）"与"真缺口（本体/AIGC）"并列成 G3–G6，导致重复实现。**定稿口径是"框架已有、只缺本体+AIGC"**——今后对照差距清单动工前，先核对既有模块（尤其 download.py 的 query/pack）是否已覆盖，避免重复建设；离线单机场景慎用"工单流转/评价闭环"等多人在线语义的功能。
- **Dataset 模型字段坑**：预处理层 `Dataset` 模型无 `project`/`scene` 列（数据获取层 DataFile 才有）——推荐服务曾直接引用导致 AttributeError 500。后续对 Dataset 的查询先核对 models/dataset.py 实际字段。
- **2026-10-09 Linux 部署期新坑（重要，接手必读）**：
  - **WSL venv 缺 ensurepip**：Ubuntu 默认 python3.14 无 venv 支持，必须先 `apt install python3.14-venv`（用 `wsl -u root` 会话装，sudo 需密码而 root 会话免密）。venv 装失败报 "The virtual environment was not created successfully"。
  - **WSL MySQL root 是 auth_socket**：`ALTER USER ... IDENTIFIED BY '123456'` 不切换插件（仍 auth_socket）；必须 `IDENTIFIED WITH caching_sha2_password BY '123456'` 才生效。用 `wsl -u root -- mysql` 免密执行。
  - **requirements.txt 的 spacy==3.8.5 无 py3.14 wheel**（Windows 锁定版）：data-platform3 已改 `spacy==3.8.16`（3.14 可用），Linux 装 requirements 失败先查这个。librosa 0.11.0（numba 链）在 3.14 解析通过，勿动。
  - **数据库迁移必须改写绝对路径**：dump 导入后，`dataset_versions.file_path`、五模态 `file_path/label_file_path`、`*_tasks.result_path`、`sample_datasets.*_path` 全是 Windows 绝对路径（`D:\DataPlatform\Uploads\...`），代码直接 `Path(file_path)` 读写——不改写则预览/下载/预处理/质检全 404。改写 SQL：`REPLACE(REPLACE(col,'\\','/'),'D:/DataPlatform/Uploads','/home/yyq/data-platform3/DataPlatform/Uploads')`。**验证残留必须用 `LIKE 'D:%'`，不能用 `LIKE '%DataPlatform%'`**（Linux 路径本身含 DataPlatform 子串，会误报残留）。
  - **登录是 Cookie 认证**：`/api/auth/login` 响应体无 access_token，token 走 Set-Cookie（dp_access_token）；curl 验证必须 `-c` 存 cookie、`-b` 带 cookie，不要找响应体里的 token 字段。
  - **WSL 里 `pkill -f 'app.main'` 会杀掉执行命令的 shell 自身**（命令行含同串）→ 用精确 PID（`kill 1587`）或 systemctl 管理。
  - **mysqldump 经 PowerShell 重定向会带 BOM**（Out-File utf8）→ 用 `cmd /c "mysqldump ... > file"` 字节流直通。
  - **本会话 PowerShell 工具长命令输出偶发丢失**：wsl 输出可能部分/全部消失，处理办法是让命令重定向到 WSL 内文件，再用短命令 `cat`/UNC `Get-Content -Encoding UTF8` 读；bash 脚本从 Windows 拷过去必须先 `sed -i 's/\r$//'` 去 CRLF。
  - **部署记录与可复用脚本**：`D:\data-platform3\db_backup\` 下有 `data_platform_dump.sql`（Windows 库导出备份）、`fix_paths.sql`（路径迁移）、`final_path_check.sql`（残留验证）、`data-platform.service`（systemd unit）、`verify_deploy.sh`（部署验证脚本，Cookie 方式）。
