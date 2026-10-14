# Linux 数据平台与少样本生成

本目录是独立的 `data-platform3-linux` 副本，原 `data-platform3` 和 tar 保持不变。

## 启动与访问

前台启动平台（Ctrl+C 停止 Web 服务；已提交训练由独立 tmux 队列继续执行）：

```bash
bash /data/gaoya/test/Code_631/data-platform3-linux/start.sh
```

本机局域网入口：<http://10.176.42.45:8000/#/fewshot>。8000 占用时程序自动顺延，实际端口见控制台。初始化账号 `sysadmin`，初始密码 `admin`，首次正常登录要求改密。

启动前静态检查：

```bash
bash /data/gaoya/test/Code_631/data-platform3-linux/start.sh --check
```

## 使用流程

1. 打开「数据预处理层 → 少样本数据生成」。
2. 填写任务名、项目、场景、密级。按组上传图像，每组填写适用于所有图片的 caption 和生成数量。
3. 点击「开始训练与生成」。多组素材共同训练一个 FLUX.2 Klein Base 4B LoRA，按各组描述生成。
4. 全部生成图像自动登记为 AIGC 来源，并调用 YOLOv8n COCO 80 类目标检测。
5. 点击「进入人工修正」。自动标注结果是 V2，人工提交形成后续版本，可以使用原有数据集下载功能。
6. 已完成的正式任务可点击「复用 LoRA 再生成」，生成数量沿用原任务、种子更新；确认后提交，不重复训练。

限制：一次最多 100 张素材、20 组、单图 20MB、总计 500MB；一次最多生成 1000 张，每组最多 200 张。真实图像描述并非框标注；生成记录保存在任务目录和 `_manifest.json`，不会作为图片进入标注。自动框、零检测结果仍需人工核验；具体机型和自定义类别不在 COCO 模型能力范围内。

## 环境与数据库

- Web：`backend/.venv`，Python 3.10.12；Linux 依赖见 `requirements-linux.txt`，解析版本见 `requirements-linux.lock.txt`。
- 训练：复用 `/data/gaoya/agent-data/envs/rareplanes_flux2`，与 Web 环境隔离。
- 引擎：`integrations/fewshot`，固定 GitHub 提交 `a668a59c804835d6c60806872b6bfc707ad7c2ee`。上游训练和生成算法未修改。
- 底座：`/data/gaoya/agent-data/weights/beyondobjects_flux2/FLUX.2-klein-base-4B`。
- 数据库：独立 MySQL 8.0，监听本机 `127.0.0.1:3307`，业务库 `data_platform_linux`，验收库 `data_platform_linux_test`。
- MySQL 文件：`/data/gaoya/agent-data/outputs_v1/data_platform3_linux/mysql/`。
- 数据库账号已写入权限为 600 的 `config.json`，不在文档中记录密码。管理凭据位于 MySQL 目录的 `admin.json`（权限 600）。
- 新增 `fewshot_jobs` 表；新库启动自动建表。已有项目使用 Alembic 时新增迁移 `0004`，不要盲目重放历史迁移或还原旧备份。

MySQL 停止后可在另一终端以前台方式启动：

```bash
mysqld --defaults-file=/data/gaoya/agent-data/outputs_v1/data_platform3_linux/mysql/my.cnf
```

不导入旧 SQL：旧备份包含 Windows 路径，且包内没有原业务 Uploads 文件。

## 任务执行与存储

`config.json` 的 `fewshot` 节点配置引擎、解释器、底座和输出路径。训练使用物理 GPU 0 的 UUID；禁止物理 GPU 4。队列串行运行，目标卡已有超过 1 GiB 占用时继续排队。Web 与 COCO 检测使用 CPU。

正式训练强制 `tracking.enabled=true`，W&B 在线记录配置、进度、loss，不上传模型权重或 checkpoint；项目名 `data-platform-fewshot`。此次已验证账号认证，未执行 400 步正式训练。

训练在独立 tmux 会话 `dp-fewshot-<配置路径哈希>` 中执行。Web 重启不会终止训练；Web 启动发现未完成任务时唤醒队列。执行进程意外退出后，重新启动队列会将活动任务标为失败，保留产物。重试优先使用完成的 LoRA 或 checkpoint；没有可恢复 checkpoint 时使用新的 run 子目录，保留旧目录。生成或标注失败不要求从头训练。

运行数据：

```text
/data/gaoya/agent-data/outputs_v1/data_platform3_linux/
├── runtime/Uploads/、runtime/Logs/       # 平台业务数据
├── mysql/                              # 隔离 MySQL 实例
├── YYYYMMDD_fewshot_<uuid>/             # 每个用户任务，日期为北京时间
│   ├── support/                        # 分组图片和 caption
│   ├── config_001.json                 # 冻结配置（JSON 兼容 YAML）
│   ├── generation_plan.json
│   ├── worker.log
│   ├── generation_metadata.json
│   └── run_001/                        # FewShotLab 配置、日志、LoRA、生成批次
└── 20261013_integration_validation/     # 本次独立验收输出
```

Web 接口 `/api/fewshot/jobs` 提供任务列表、分组上传、详情；`/{id}/submit` 提交或重试；`/{id}/generate` 创建复用任务。输入使用服务器生成的文件名，校验格式、大小、像素数和重复图像；任务操作受角色、密级和创建人校验保护。导入完整批次前校验数量与图片 SHA-256，重复导入不会重复建数据集。

## 验收结果（2026-10-13）

- Linux 静态检查、基础依赖导入、pip check、前端生产构建通过。
- 独立 MySQL 集成检查：登录/权限、分组上传、重复提交、坏图/重复图拒绝、生成计划、入库完整性和幂等性、真实 CPU YOLO、人工版本、下载标签查询通过。
- 修复原平台标签查找中的 ID 混淆：图像获取记录 ID 与预处理数据集 ID 不再直接等同；回归检查确认不会读取无关数据集的标注。
- 真实 smoke：两组 8 张仓库示例图，共同训练 2 步，生成 2 张，再调用同一个 LoRA 额外生成 2 张；两批均完成入库和自动标注。
- 初次批次 2 张都有检测框；复用批次 1 张无检测结果，保留为空标注并提示人工检查。
- 初次批次完成 V3 人工修正与带标注 ZIP 导出。该短测试只验证流程，不代表生成质量。
- 网页检查涵盖桌面/390px 窄屏、增加分组、已完成任务预览、进入人工修正。截图和测试结果在验收输出目录。
- smoke 的 LoRA 保存/加载及复用验收结束后，停止测试队列，精确删除本次 smoke 权重；保留图片、配置、日志、元数据、验收记录和 ZIP，详见 `smoke_weight_cleanup.json`。没有删除输入底座或历史模型。

测试命令（仅专用测试库）：

```bash
DP_CONFIG_PATH=/data/gaoya/agent-data/outputs_v1/data_platform3_linux/20261013_integration_validation/config.json CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /data/gaoya/test/Code_631/data-platform3-linux/backend/.venv/bin/python /data/gaoya/test/Code_631/data-platform3-linux/backend/tests/check_fewshot.py
```

临时浏览器验收服务的前台命令（验收后关闭）：

```bash
DP_CONFIG_PATH=/data/gaoya/agent-data/outputs_v1/data_platform3_linux/20261013_integration_validation/config.json bash /data/gaoya/test/Code_631/data-platform3-linux/start.sh
```

环境说明：安装 MySQL 核心包时，系统此前未配置完成的内核包触发 NVIDIA DKMS 构建失败，apt 返回非零；MySQL 核心包已配置成功并通过实际数据库验收。此次未修复内核/驱动问题，也未重启。原前端依赖锁文件保留，npm 报告的既有依赖安全告警未在本功能中升级处理。


## 本地素材选择（2026-10-13）

少样本页面的每个素材组现在有“从本地素材库选择”入口。可分页预览、按文件名搜索、跨库选择并移除图片；也可与上传图片混用。每组仍需填写共用描述，多组共同训练一个 LoRA。

当前素材库白名单在 `backend/app/services/local_materials.py`，仅包括：

- `/data/gaoya/dataset/laion-high-resolution-data_v1_500`
- `/data/gaoya/dataset/laion1024_2500`
- `/data/gaoya/dataset/recoilme-aesthetic_photos_xs/images`

浏览和使用素材须有“内部”数据的查看及操作权限。接口不接受任意服务器路径，不递归扫描目录，拒绝符号链接、目录穿越和非图像文件。只读取白名单目录中的图像，不将同名 JSON 当作标注或自动填入共用描述。

选图本身不创建训练任务、不复制图片。提交任务时将选中素材复制到任务 support 和原始数据目录，记录素材库 ID、原始文件名、SHA-256；保留原数据。仍执行 100 张、单图 20 MB、总计 500 MB、4000 万像素和内容去重校验。无需新增 SQL 表。

验收：`backend/tests/check_local_materials.py` 在专用 MySQL 测试库验证权限、分页/搜索、缩略图、路径拒绝、本地与上传混用、本地单独创建、来源哈希和重复拒绝；浏览器验证跨页选图、390px 窄屏、实际创建草稿。未提交训练。截图保存在 `/data/gaoya/agent-data/outputs_v1/data_platform3_linux/20261013_local_materials/`。


## 按类别浏览与 caption（2026-10-13）

118 少样本素材选择器支持类别分组、类别筛选、文件名/描述搜索。图片右侧显示全部类别与源数据 caption；长描述可滚动查看，跨类别/跨数据集选择仍可保留。类别与 caption 仅用于素材浏览，分组训练描述由用户填写。

FGVC Aircraft 使用官方 train/val/test 的 variant 文本文件，覆盖 100 个机型和 10,000 张图；RarePlanes 从 COCO 角色标注关联图像，3 个角色可以在同一张图片中出现；LAION-SG 使用场景图 items.label 和 caption_ori，一张图片可对应多个对象类别；HPDv3 从图片配对的 path1/path2 匹配 prompt，当前库无法匹配 benchmark 类别，按“未分类”展示描述。无可匹配元数据的图片保留在“未分类”中。类别计数按图片去重，多标签数据的类别计数之和可能超过图片总数。

配置中的 local_materials.<id>.metadata 指定适配器 kind 及来源白名单 sources，local_material_index 指定只读浏览索引。索引位于远端 `/mnt/data/gaoya/agent-data/outputs_v1/data_platform3_linux/20261013_material_metadata/materials.sqlite`。元数据先流式解析，再生成 SQLite 索引；分页访问不会加载多 GB 原始 JSON。图片及源标注不修改，不把 dataset 类别自动转换为 COCO 检测标签。

索引只反映构建时的图片与元数据快照。管理员更新素材后，可在 118 上执行以下命令重建索引；构建成功后原子替换文件，无需重启 Web：

```bash
cd /mnt/data/gaoya/agent-data/apps/data-platform3-linux
backend/.venv/bin/python -m pip install -r backend/requirements-material-index.txt
backend/.venv/bin/python scripts/build_material_index.py --config config.json --output /mnt/data/gaoya/agent-data/outputs_v1/data_platform3_linux/20261013_material_metadata/materials.sqlite
```

只读接口验收脚本 `backend/tests/check_material_metadata.py` 要求专用测试库，验证真实类别计数、多标签筛选、caption 搜索、未知类别及无类别图片。浏览器验收截图在 `/data/gaoya/agent-data/outputs_v1/data_platform3_linux/20261013_material_metadata/`。


## 全部方法与模型选择（2026-10-13）

118 正式页面现已提供训练方法和基础模型选择，三种有效组合与上游 FewShotLab 的 a668a59c804835d6c60806872b6bfc707ad7c2ee 提交一致：BeyondObjects + FLUX.2 Klein Base 4B、BeyondObjects + SD2.1、ChimeraLoRA + SD2.1。Chimera 不支持 FLUX；输入无效组合会在创建前拒绝。训练文本框标为“训练文本提示词（caption）”。

BeyondObjects 多组共同训练一个 LoRA；Chimera 保留 JOINT 共享 A 和每张素材一个 B 头，生成按原生 Dirichlet 融合全部头，不伪装成普通单 LoRA。此入口没有提供 SB 所需输入 mask，使用仓库的无 mask 路径。Chimera 中间 checkpoint/resume 不支持，checkpoint_every 始终大于目标步数；失败时从新运行目录重新训练，完整多头权重可复用生成。BeyondObjects SD2.1 保留原生整轮约束，向上取整到完整轮次，界面显示实际步数；9 张素材、batch=1、accumulation=4 时，400 步目标会执行 402 步。

方法和模型存放在已有 FewshotJob.options JSON 中，无需新 SQL 列；历史缺少这些键的任务默认 BeyondObjects + FLUX.2。fewshot.backends 为每个 backend 配置受管理员控制的 python、base_model 和显存门槛；GPU UUID 只继承全局 fewshot.gpu，不接受客户端路径、解释器或 GPU 设置。两种 SD 方案复用 beyondobjects 环境和 chimeralora/sd21-base 权重，FLUX 复用 beyondobjects_flux2 环境和现有 FLUX 权重。没有修改现有训练环境的软件包。

本次用户任务 #1 的退出原因是加载 FLUX 文本编码器时的 CUDA OOM。CUDA_VISIBLE_DEVICES 只暴露物理 GPU 7，因此 PyTorch 日志的 GPU 0 是进程内编号。此前“启动时检查空闲显存”不能预留物理显存，其他 4 个生成进程在模型加载期间增加了占用。

当前共享门槛：FLUX 至少 24 GiB、BeyondObjects SD2.1 至少 18 GiB、Chimera SD2.1 至少 20 GiB。连续 3 次采样满足门槛（间隔 3 秒）后启动；每个 GPU 阶段及子进程初始化 CUDA 前再检查可用显存，变化不足则保持 queued。PyTorch 分配上限按最新可用显存扣除 4 GiB，并受冻结运行配置的较低上限约束。共享模式不能锁定其他项目的显存，因此仍可能出现运行中的 OOM；此时保留素材/进度并明确显示具体原因，等待用户重试。

平台集成层对 vendor 保持原样；仅修改 FewShotLab 的 core.device（平台环境变量启用的启动前显存检查）及 cli（保留子进程 failure.json 根因，避免 CalledProcessError 覆盖）。这些变化随训练源码快照记录，不影响其他项目的仓库副本。

验收：26 项 CPU 回归通过，包括模拟显存下降与子进程 OOM 根因保留；原始上游对 OOM 根因用例失败，平台修正后通过。独立 MySQL API 测试验证三种选项持久化、非法组合拒绝、历史默认和原生配置约束；三个环境均完成 9 张素材的本地 tokenizer/config 检查及 SD 原生训练模块导入。浏览器验证模型联动、多头提示、caption 标签、实际创建 Chimera 草稿和 390px 页面。提交请求被拦截，未启动 GPU 训练；测试 HTTP 服务已关闭。生产任务 #1 保持 failed，未自动重试，现有素材、日志及运行目录保留。

正式服务在本机 tmux 8000:0.0 前台运行：

```bash
ssh -t 118 'bash /mnt/data/gaoya/agent-data/apps/data-platform3-linux/start.sh'
```

验收截图和记录位于 `/data/gaoya/agent-data/outputs_v1/data_platform3_linux/20261013_all_methods/`。


## SSH118 数据集展示与功能验收（2026-10-13）

已在生产库登记 32 个“展示 ·”数据集，共 698 个有限样本：7 个图像集（368 张）、12 个视频集（104 段）、13 个文本集（226 份）。源文件约 362.4 MiB，原始数据保留；平台使用独立副本及版本目录。每个初始展示版本含 `_manifest.json`，记录原始路径、SHA-256、已有类别及 caption。Physics IQ 的两个 verified 目录中本次选中的视频内容相同，视为两个来源目录的镜像展示，不算独立样本证据。

数据来源包括 FGVC Aircraft（100 类，每类 2 张，原始框与机型）、RarePlanes 合成图与 7 类真实飞机裁剪、LAION-SG、HPDv3、Physics IQ 三个目录及切换帧、new_data0826、new_data0826_twoobj、0717pybullet、PhyCo Kubric、5 个 physv 受控物理视频目录。AVA 与 TAD66K 只发现已有分析 JSON/CSV，本次按文本接入，并在名称标为“已有分析结果（非原图）”；未将分析结果冒充原始照片。

少样本素材选择器保留 4 个完整图像库，新增 3 个有限展示素材库：RarePlanes 真实飞机裁剪 28 张/7 类、PhyCo 首帧 9 张/9 类（8 张有原始 caption）、Physics IQ 场景帧 12 张（未分类）。源类别不自动改写为 COCO 标签，未填写 caption 的组仍由用户输入训练文本。索引适配器新增 `platform_demo_manifest`，可以使用现有重建命令复现新增素材索引。

实际验收覆盖原始预览、可视化、质量评估、数据集版本、100 类统计及类别筛选下载；独立“验收 ·”数据集运行 CPU COCO 检测、标注确认写入、JPG→PNG、50/25/25 采样且标注随行、重复文件和空行清洗、文本关键词→清洗→采样工作流、视频关键帧检测、图像 9 维梯度直方图特征、80 个图文原始文件集成、结果回写与下载、版本发布和血缘。质量评分反映文件可读性、结构与标注配套，不代表检测准确率；当前“TF-IDF”与 HOG 功能分别为简化词频特征与梯度直方图。集成验证的是文件组织与来源清单，不代表模型层的跨模态融合。

发现并修复：版本数据量误计标注/清单文件；类别统计与下载漏读 `annotations` 及分类标签；下载清单误包含 `_manifest.json`；从原始数据创建版本未继承已有标注；软删除数据集缺少 `datetime` 导入。图像预览新增已有类别和 caption 文本展示。回归检查：`backend/tests/check_demo_imports.py`。导入脚本 `scripts/import_118_demos.py --apply` 按名称跳过已接入项目；验收脚本 `scripts/test_118_demos.py` 会拒绝重复提交已有报告目录的整批任务，复查先阅读报告及任务 IDs。

本次未新增 LoRA 训练任务；用户原有任务由独立 worker 管理。未发现适用的音频或雷达原始样本，因此没有伪造数据填充页面；SAM2/SAM3 和知识图谱/本体页面仍保留原有未实现范围。

运行服务仍在本机 tmux `8000:0.0` 前台，访问：

- 数据集：<http://10.176.53.118:8000/#/datasets>
- 素材选择：<http://10.176.53.118:8000/#/fewshot>
- 飞机 100 类及原始框：<http://10.176.53.118:8000/#/viz/image/3>
- 质量评估示例：<http://10.176.53.118:8000/#/quality?dataset_id=21>

前台启动命令：

```bash
ssh -t 118 'bash /mnt/data/gaoya/agent-data/apps/data-platform3-linux/start.sh'
```

接入清单、接口验收、浏览器截图与下载包：本机 `/data/gaoya/agent-data/outputs_v1/data_platform3_linux/20261013_platform_dataset_demo/`；118 上对应 `/mnt/data/gaoya/agent-data/outputs_v1/data_platform3_linux/20261013_platform_dataset_demo/`。脚本只在管理员指定路径复制有限样本，不扫描模型权重目录，不改动源数据。


## 数据下载 → 少样本补充 → 预处理版本下载（2026-10-13）

“数据下载”页查询后保存本次条件、需求数量、创建用户及候选图像 IDs。点击“AIGC 生成补足”只配置数量并带入少样本编辑页；用户可调整候选素材、分组、训练文本、方法、模型和数量，明确点击“开始训练与生成”后才提交任务。候选素材受现有上限约束（100 张、单图 20 MiB、总计 500 MiB），按文件哈希去重；支持与本地库素材、上传图像混用。单次最多生成 1000 张，分组共同训练沿用原模型行为。

生成任务保存 download_request_id，完成后显示在对应下载请求下方。自动标注完成即可下载，显示“待人工核验”。下载结果可选择具体版本；人工修正、自动标注、清洗、转换、采样、集成、工作流、质量评估入口自动携带 dataset_id、version_id 和返回下载请求的信息。预处理完成后返回下载页并刷新，可选择新的产物版本。任务未成功完成的版本禁止打包、发布或作为下游输入，避免下载部分产物。特征提取版本下载的是特征文件；进行图像标注时请选择仍包含图像的版本。

普通用户只可访问自己的少样本任务、训练素材、生成图像和派生版本；管理员可通过“查看所有用户”查看其他用户任务和下载请求。后端覆盖详情、预览、文件、打包、版本、质量报告和供数接口，继承私有来源的集成结果也保持隔离。已有共享素材与公开导入的 AIGC 数据保留原共享权限。人工标注只允许写入所选版本实际存在的数据文件，API 响应禁止账号切换后复用浏览器缓存。用户标识取自认证信息，客户端不能指定产物归属。

数据库新增 download_requests 表；fewshot_jobs 增加可为空的 download_request_id 及索引。启动时自动做幂等迁移，历史任务保留且不猜测其属于哪个下载请求。独立少样本任务的结果仍可通过既有数据集与按条件下载访问；从下载页提交的新补充任务才具有对应关联。复用任务继承原任务的下载请求关联。

验收：43 项隔离 CPU 接口检查、14 项浏览器检查、三个方法配置回归、3 项元数据/计数回归和历史表迁移检查均通过。生成批次使用明确标记的模拟批次验证接口串联，自动标注、导入、人工修正与版本下载执行真实代码；浏览器实际创建关联草稿，但拦截训练提交。未额外提交 GPU 训练，验收数据不进入生产库。正式任务 #1 保持 failed，#2 保持 completed，既有产物保留。升级时只停止无待处理任务的旧队列 worker；下次正式提交会启动新版 worker。

服务已更新至 118 的正式应用，仍在本机 tmux 8000:0.0 前台运行。访问：<http://10.176.53.118:8000/#/download>；前台启动命令：

```bash
ssh -t 118 'bash /mnt/data/gaoya/agent-data/apps/data-platform3-linux/start.sh'
```

代码快照、接口验收、浏览器截图和部署记录保存在本机 `/data/gaoya/agent-data/outputs_v1/data_platform3_linux/20261013_download_fewshot_integration/` 与 118 对应的 `/mnt/data/gaoya/agent-data/outputs_v1/data_platform3_linux/20261013_download_fewshot_integration/`。测试服务 8001 已关闭；它使用独立 SQLite、独立 JWT 和拒绝训练提交的入口，复现命令（前台）：

```bash
ssh -t 118 'DP_ACCEPTANCE_DIR=/mnt/data/gaoya/agent-data/outputs_v1/data_platform3_linux/20261013_download_fewshot_integration/acceptance /mnt/data/gaoya/agent-data/apps/data-platform3-linux/backend/.venv/bin/python /mnt/data/gaoya/agent-data/outputs_v1/data_platform3_linux/20261013_download_fewshot_integration/staging/scripts/serve_download_acceptance.py'
```


## 在线标注入口与累计改动记录（2026-10-13）

原始数据列表提供“在线标注”和“导入标注”。在线入口复用当前账号已有的单来源工作集及完成版本；没有时先展示复制数量/大小，用户确认后建立 V1，继承已有标注，再自动选择工作集和版本进入手动标注。保留原始文件，人工提交形成新版本。创建权限沿用管理员角色；通用工作集保留共享/密级策略，少样本私有结果仍隔离。复制失败回滚，未完成版本与缺失文件拒绝进入。

文本默认实体识别；已修复目标数据集不在前 100 条时的版本预加载。21 项本地及远端隔离接口检查、10 项浏览器检查、生产构建和正式接口只读预检通过；未新增 GPU 训练。临时 8001 验收服务已关闭，正式 8000 继续在本机 tmux 前台运行。

累计改动见 [CHANGELOG_LINUX.md](CHANGELOG_LINUX.md)，后续按日期追加。验收代码、记录、截图与部署基线在本机 `/data/gaoya/agent-data/outputs_v1/data_platform3_linux/20261013_online_annotation/` 及 118 对应目录。前台启动命令仍为：

```bash
ssh -t 118 'bash /mnt/data/gaoya/agent-data/apps/data-platform3-linux/start.sh'
```
