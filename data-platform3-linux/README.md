> **GitHub 源码交付**：当前分支含源码与前端构建，不含模型权重、标注和真实配置。部署前请先读 [PACKAGE_README.md](PACKAGE_README.md)，完整改动见 [CHANGES_FROM_WINDOWS_TAR.md](CHANGES_FROM_WINDOWS_TAR.md)。

> 本目录为 Linux + 少样本生成适配副本，当前运行方法和验收结果见 [LINUX.md](LINUX.md)。下文保留原项目历史说明。
> 累计改动见 [CHANGELOG_LINUX.md](CHANGELOG_LINUX.md)；后续变更按日期追加。
> 逻辑审计见 [PLATFORM_LOGIC_AUDIT_20261013.md](PLATFORM_LOGIC_AUDIT_20261013.md)，待完善与验收清单见 [TODO_LOGIC.md](TODO_LOGIC.md)。

# 数据中台（离线单机全栈版）v3.2

## v3.2 变更摘要（2026-09-17）

1. **数据来源标签**：数据获取层五模态新增 source_type（真实采集/仿真/AIGC），模型 mixin 一处生效，列表筛选+表单选择，AIGC 数据彩色标签标识；配套 Alembic 0003 迁移。
2. **自动标注引擎升级**（对齐《五模态自动标注方案确认清单》）：语音新增 Whisper 转写（faster-whisper，词级时间戳+置信度）并修复 silero_vad 规则未注册的遗留 bug；文本新增 GliNER 零样本实体识别（任意类型名直接抽取）；图像/视频注册 SAM 3/SAM 2 规则（评估项/二期，需 GPU）；新模型统一 config.json models 节点装载，缺失走 503 机制。
3. **可视化中心改造**：删除 RGB 直方图/色彩热力图/频谱图/音高曲线四张无信息图；新增"数据集总览"Tab——标签类别分布、类别×文件覆盖热力图、标注源构成，新接口 `/{data_type}/{id}/label-stats`（与 preview 同口径寻址）。
4. **数据下载模块**（`routers/download.py`）：`/api/download/query` 按模态/场景/标签/项目/密级查询含指定标签的文件；`/api/download/pack` 跨数据集打包下载（附 _lineage.json）；`/api/labels` 标签全集；`/api/aigc/generate` AIGC 补足空接口（501 占位，二期接入点见接口 docstring）。
5. **导航扩展**：数据获取层新增知识图谱获取/本体获取，数据预处理层新增知识图谱抽取/本体抽取（均为占位页面）；**下线模板辅助标注**（HybridLabel 前后端与模板 CRUD 移除，Template 表保留以保历史任务）。

## v3.0 变更摘要（2026-09）

1. **语音自动标注增强（Silero VAD）**：新增基于 ONNX 的 Silero VAD 语音活动检测算法，替代传统能量门限，输出带置信度的语音段标注；模型文件缺失时自动降级到静音分割规则。
2. **文本自动标注增强（Spacy NER）**：新增基于 Spacy `zh_core_web_sm` 的中文命名实体识别算法，支持人名/地名/机构/时间/数值等实体类型；长文本自动分段处理（>10 万字符）；模型缺失时降级到正则实体识别。
3. **模型就绪状态联动前端**：`/auto-label/rules` 返回各规则 `model_ready` 与 `model_name`；前端自动标注页实时显示模型状态；模型未就绪时提交按钮禁用并提示用户放置模型文件。
4. **模型路径可配置**：`config.json` 新增 `models.vad_path` / `models.ner_path`，支持自定义模型位置（留空则使用默认 `backend/models/`）。
5. **打包适配**：PyInstaller `hiddenimports` 追加 `vad_service`、`ner_service`、`spacy` 相关模块；增加 `--collect-all spacy` 确保管线配置完整打包。

---

# 数据中台（离线单机全栈版）v2.2

## v2.2 变更摘要（2026-07）

1. **检测推理改双后端（修复部分机器 cv5 加载失败）**：onnxruntime 优先、cv2.dnn 兜底。背景：2026 年构建的 OpenCV 5.0 默认启用新 ONNX 解析器（ONNXImporter2），在部分 Windows 环境 parseFile 阶段即拒绝合法模型；onnxruntime 用独立解析器不受影响，3.8/3.10/3.14 均有 Windows x64 离线 wheel（1.19.2 / 1.23.2 / 1.28.0）。模型同步改为 opset 13 + onnxsim 简化（IR7），双后端检测结果一致。
2. **依赖修正**：numba（librosa 传递依赖）尚不支持 numpy 2.5，3.11+ 组 numpy 锁 2.4.6；requirements.txt 三组均加入 onnxruntime 并补全其传递依赖（coloredlogs/flatbuffers/protobuf/sympy 等，离线 wheel 已验证）。


## v2.0 变更摘要（2026-07）

1. **COCO 语义目标检测自动标注**：内置 YOLOv8n 检测模型（ONNX，12.8MB，onnxsim 简化图，兼容 OpenCV 4.8 与 5.0 的 DNN 解析器，随 exe 打包，完全离线），对图像/视频内容做真实目标识别——COCO 80 类（人、汽车、飞机、船等），输出带置信度的标注框；图像模态直接检测，视频模态按可配间隔抽帧检测并把框回写到对应时间点；参数：置信度阈值（默认 0.4）、NMS 阈值（0.45）、最大框数（300）、视频抽帧间隔（2 秒）。规则自动标注与工作流的自动标注节点均可选用"语义目标检测（COCO）"算法，产物与手动标注/COCO 导入格式一致，可直接预览叠加、进入数据集版本。
2. **推理引擎零新增依赖**：用 OpenCV DNN（opencv-python-headless 是既有依赖）加载 ONNX，不引入 onnxruntime——后者在 3.8 与 3.14 上的 wheel 支持断裂，cv2.dnn 三个版本通用（abi3 轮子）。
3. **Python 多版本适配（3.8.8 / 3.10 / 3.14）**：requirements.txt 按 python_version 环境标记分三组锁定版本（3.8 组=各库最后支持 3.8 的稳定版；3.10 组 numpy 锁 2.2.6——numpy 2.5 起不再提供 cp310 轮子；3.11+ 组覆盖 3.14），pip 安装时自动选用当前解释器对应组；三套依赖全部验证存在 Windows x64 离线 wheel。
4. **版本文件清单 label_count 兼容检测产物**：自动标注图像标签（boxes 结构）现在也计入标注条数（此前只数 annotations/items）。
5. **一键启动**：后端启动即托管前端页面（`frontend/dist` 存在时自动挂载，打包 exe 走 `_MEIPASS/static`），启动后端（或双击 exe）后浏览器打开即完整系统，无需单独 `npm run dev`；build.py 自动完成前端构建与复制。

## v1.5 变更摘要（2026-07）

1. **文件夹上传上限提升**：一次最多 5000 个文件 / 10GB（原 500 / 2GB）。
2. **修复手动标注工作台图片不显示**：版本文件列表用文件名（basename）拼接下载地址，文件夹上传的数据在子目录下导致 404；现统一以版本目录内相对路径寻址，同时图像模态等待图片加载完成后再建立画布。
3. **文件列表分页 + 文件名搜索**：数据获取层文件清单、手动标注工具文件列表均改为服务端分页（20/50/100/200 可选）+ 关键字模糊搜索，支撑 5000 文件级数据集。
4. **标注来源可追溯**：预览勾选"显示标注"时显示标注来源——优先"原始标注（随数据上传）"，无原始标注时自动回读"以本数据为来源的数据集"最新版本中的标注产物，并标注来源（如：数据集「XX」V3）。
5. **可视化中心文件切换**：可视化详情页支持上一个/下一个/可搜索下拉切换文件；图像页支持"显示标注"叠加（含标注来源提示）。
6. **COCO 整表标注导入**：上传含 images/annotations/categories 的 COCO JSON，按 file_name 匹配（支持带目录相对路径与纯文件名两种），自动拆分为每文件 `_label.json` 落盘；返回匹配/未匹配统计。文件夹上传时其中的 `instances*.json`/`annotations*.json` 会被自动识别导入（导入后不作为数据文件保留）；`iscrowd=1` 与 `ignore/others` 类标注按惯例跳过；annotations 的 category_id 越出 categories 表的不规范导出按 `cat_{id}` 保留。
7. **质量评估标准明细**：内置标准未声明可映射维度时，展开明细回退为全部六维得分（此前仅 GB/T 36344-2018 可展开）。
8. 数据集版本新增分页文件清单端点（含每文件标注条数 label_count）。
9. **适配 Python 3.8.8**：全部源码通过 3.8 语法校验（`X | None` 等写法经 `from __future__ import annotations` 延迟求值，移除 `missing_ok=` 等 3.9+ 用法）；requirements.txt 重新锁定为各库最后一个支持 3.8 的稳定版（FastAPI 0.103.2 / SQLAlchemy 2.0.23 / pydantic 2.5.3 / numpy 1.24.4 / opencv 4.8.1.78 等），全部 53 个依赖已验证存在 Windows x64 cp38（或 abi3）离线 wheel。

基于《数据中台全栈开发需求说明书 v1.3》实现并按 v1.4 变更集升级：FastAPI + MySQL 8.0 后端，Vue 3 + Element Plus 前端，PyInstaller 打包为 Windows 单机 exe。

## v1.4 变更摘要（2026-07）

1. **文件夹整体上传**：保留目录结构（相对路径入库），串行队列上传（默认上限 500 文件 / 2GB），断点续传按"文件 hash + 相对路径"匹配；项目字段改为真实项目库（可搜索、可新建）。
2. **下载升级**：按版本下载 + 内容筛选（数据/标注/清单任意组合打 ZIP）+ 版本内单文件下载 + 数据获取层单文件下载。
3. **密级体系**：六级密级 **公开 < 受控 < 内部 < 秘密 < 机密 < 绝密**，数据获取层与数据集全面启用。
4. **手动标注工具**：五种模态全部改为可视化画布标注——图像拖框、视频按时间点拖框（±0.5s 回放显示）、语音波形段选（Web Audio 解码）、文本划词（选区偏移计算）、雷达 2D BEV 框选（CSV/PLY 前端解析）；已有标注自动回显，提交写入新版本 `_label.json`（目录结构镜像）。
5. **按用户密级授权**：用户管理可为访客分别授予"可见密级"与"可操作密级"（可操作 ≤ 可见，强制校验）；管理员/系统管理员不受限；访客无操作授权时所有任务入口禁用（后端同步 403）。
6. **参考标准**：支持上传/下载标准原文（PDF/Word/图片，≤50MB）；自定义标准按"质量指标 + 比较符 + 阈值"规则评估；质量评估可勾选标准子集，每项标准单独出合规率与逐条规则结果。
7. **预览升级**：多文件切换（上一个/下一个/下拉搜索）+ 标注叠加显示开关（图像框、语音段、文本高亮、雷达 BEV 框）。
8. **修复**：结果回写 500（settings 未导入）；工作流全失败（会话缓存导致状态误判 + 算法硬编码语音规则）；采样标注随行键匹配。
9. **工作流按模态选算法**：自动标注节点算法随所选数据集模态动态加载，后端校验算法与模态匹配，采样比例和=100% 前后端双重校验。
10. **跨模态集成**：其余输入版本可跨数据集、跨模态选择（如图像+雷达）；产出数据集密级不得低于任一来源最高密级（防止高密级数据降级）；产物按来源模态子目录组织 + `_manifest.json` 记录血缘；新增"综合（mixed）"数据集类型。
11. **全部下拉支持关键字过滤**；采样任务完成后显示标注随行统计。
12. 清洗/转换任务保留文件夹上传的相对目录结构（不再扁平化）。

**运行环境：Python 3.8.8 / 3.10 / 3.14（依赖按 python_version 自动分组）、Node 18+（仅前端开发需要）、MySQL 8.0。**

## 一、目录结构

```
data-platform/
├── backend/                    # FastAPI 后端
│   ├── app/
│   │   ├── main.py             # 入口：启动流程 + uvicorn.Server 内嵌启动
│   │   ├── config.py           # config.json 读写（jwt_secret 自动生成写回 / D 盘回退）
│   │   ├── database.py         # 连接 + 自动建库(utf8mb4) + create_all + 种子数据
│   │   ├── security.py         # bcrypt(12) / JWT / 会话超时 / 并发会话 / 角色权限
│   │   ├── models/             # 全部 ORM 模型（用户/五类数据/数据集版本/任务/标准）
│   │   ├── routers/            # auth / users / data×5 / datasets / tasks / standards
│   │   ├── services/           # file_service / version_service / preprocess /
│   │   │                       # quality_service / log_service
│   │   └── utils/common.py     # 统一错误格式 / 分页 / 文件日志(30天滚动)
│   ├── alembic/                # 迁移（0001 基线；0002 密级授权/项目库/标准规则）
│   ├── requirements.txt        # 锁定版本
│   └── seed_demo.py            # 演示数据生成脚本
├── dist/static/                # 前端构建产物（Vite 打包后复制于此）
├── build.py                    # PyInstaller 打包脚本（生成 .spec 并执行）
└── README.md
```

前端源码见项目前端工程（Vue 3 + Vite + Pinia + Element Plus + ECharts，`npm run build` 产物复制到 `dist/static/`）。

## 二、开发环境运行

### 后端

```bash
cd backend
pip install -r requirements.txt
python -m app.main
```

启动流程：读取/生成 `config.json` → 清理 temp 过期临时文件 → 连接 MySQL（自动建库 `data_platform`，utf8mb4）→ `create_all` + `schema_version` 标记 → 初始化默认数据（默认账号/5 项参考标准/3 个标注模板/14 条航空术语）→ 端口 8000（占用自动顺延并提示）→ 自动打开浏览器。

> 无 MySQL 的联调环境可用 `DATABASE_URL` 覆盖，例如：
> `DATABASE_URL=sqlite:///./dev.db python -m app.main`（仅开发用，生产按需求书使用 MySQL 8.0）。

### 前端

**一键启动（推荐，v2.0）**：只要 `frontend/dist` 存在（执行过一次 `npm run build`），后端启动即同时托管页面——`python -m app.main` 后浏览器访问 `http://localhost:8000` 就是完整系统，无需单独启动前端。

```bash
cd frontend
npm install
npm run build      # 生成 frontend/dist（后端自动托管）；打包时 build.py 会自动执行本步并复制到 dist/static
npm run dev        # 可选：仅前端热更新开发时使用（Vite 代理 /api → localhost:8000）
```

### 默认账号（初始密码均为 `admin`，首次登录强制改密，不可跳过）

| 用户名 | 角色 | 权限概要 |
|---|---|---|
| sysadmin | 系统管理员 | 全部权限（用户/系统配置/日志/数据/预处理/标准） |
| admin | 管理员 | 数据与预处理 CRUD、查看用户与日志 |
| guest | 访客 | 默认仅查看公开级数据、无操作权限；可由系统管理员授予可见/可操作密级 |

## 三、Windows 打包部署

```bash
cd data-platform
python -m venv .venv && .venv\Scripts\activate
pip install -r backend/requirements.txt pyinstaller
python build.py          # 生成 dist/DataPlatform/DataPlatform.exe
```

目标机要求：Windows x64、MySQL 8.0（localhost:3306，账号密码在 `config.json` 中配置，默认 root/123456）。

双击 `DataPlatform.exe`：控制台打印启动流程与实际端口，自动打开浏览器。

## 四、config.json（程序同级目录，首次启动自动生成）

```json
{
  "database": { "host": "localhost", "port": 3306, "username": "root", "password": "123456", "dbname": "data_platform" },
  "upload_dir": "D:\\DataPlatform\\Uploads",
  "log_dir": "D:\\DataPlatform\\Logs",
  "jwt_secret": "<首次启动自动生成随机密钥并写回>",
  "server_port": 8000,
  "security": {
    "max_login_attempts": 5,
    "lockout_duration_minutes": 30,
    "session_timeout_minutes": 120,
    "allow_concurrent_sessions": true
  }
}
```

- D 盘不可用时自动回退到程序同级 `DataPlatform\` 并写回配置；
- 四项安全策略以数据库 `system_settings` 表为准（系统配置页修改即时生效）。

## 五、核心机制速查

| 机制 | 说明 |
|---|---|
| 认证 | 仅账号+密码登录（角色后端按账号决定）；bcrypt(12)；JWT Access 2h + Refresh 7d，HttpOnly Cookie |
| 账号锁定 | 连续失败 N 次锁定 M 分钟（可配），锁定期间拒绝登录并提示剩余时间 |
| 并发会话 | 关闭后新登录 token_version+1，旧会话立即 401 |
| 会话超时 | last_activity 超时未活动自动 401，前端自动回登录页 |
| 强制改密 | 初始/重置密码 must_change_password=1，弹窗无关闭按钮不可跳过 |
| 上传 | 8MB 分片、按文件 hash 断点续传、前端 1s 轮询进度、格式白名单；支持文件夹上传（保留目录结构，500 文件/2GB 上限） |
| 密级 | 六级：公开<受控<内部<秘密<机密<绝密；访客按用户授予可见/可操作密级（可操作≤可见），管理员不受限，列表/详情/下载/任务全链路强制校验 |
| 版本化 | 创建数据集自动生成 V1(draft,init)；任务成功自动创建输出版本并更新 current_version_id；draft→published 单向；published 只读不可删；draft 删除校验引用并清理目录 |
| 血缘 | parent_version_id 单父链；集成任务其余输入版本记录于任务 detail |
| 采样 | 训练/验证/测试三划分，比例和=100% 前后端双重校验 |
| 质量评估 | 六维度真实计算 + 参考标准合规；版本级评估回写 dataset_versions.quality_score |
| 操作日志 | 全部操作强制记录（DB + 文件 30 天滚动），不可删除，支持 CSV 导出 |

## 六、演示数据

服务启动后执行：

```bash
python backend/seed_demo.py --password <你的密码>
```

将创建语音/雷达/文本样例数据、数据集（含自动标注→采样版本链）与一次版本级质量评估。
