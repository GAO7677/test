# 数据中台：相对 Windows 原始 tar 的完整修改说明

更新日期：2026-10-14。本文描述清理后的当前有效实现；历史逐次记录见 [CHANGELOG_LINUX.md](CHANGELOG_LINUX.md)。

## 1. 对比基线、范围与数量

| 项目 | 位置或结论 |
| --- | --- |
| 原始基线 | `/data/gaoya/test/Code_631/data-platform3.tar`，直接读取归档成员逐文件比较；未改写原包 |
| 基线 SHA256 | `905eb245806eabd0f390d28e7dd0248380c4f06936ec184fdc3be4a46d4aadf1` |
| 本地当前代码 | `/data/gaoya/test/Code_631/data-platform3-linux` |
| SSH 118 运行代码 | `/mnt/data/gaoya/agent-data/apps/data-platform3-linux` |
| 首次核对 | 新增 90、修改 40、未变 61、删除 0，共 191 个纳入统计的文件 |
| 清理后功能版本 | 移除 2 个原包中不存在的一次性验收脚本后：新增 88、修改 40、未变 61、删除 0 |
| 含本报告的最终代码目录 | 本报告再新增 1 个 Markdown：新增 **89**、修改 **40**、未变 **61**、删除 **0**，共 **190** 个文件 |

统计包括源码、配置、依赖清单、脚本和文本说明；不包括 `.venv`、`node_modules`、缓存、前端 `dist`、构建目录、数据库备份、模型权重及 FewShotLab 示例素材。用户标注和训练输出位于项目外的运行数据目录，**不属于上述新增文件数**。因此该统计不是整个 tar 的所有二进制文件差异，也不是新增业务模块数量。

原来的 90 个新增文件按主要用途划分：平台业务代码 19、少样本集成源码/配置 29、视频模型集成 5、依赖及数据维护工具 7、测试/验收脚本 24、根目录维护文档 6。清理后测试/验收脚本为 22 个，其余类别不变，本报告另计。`scripts/check_linux.py` 虽归在检查脚本中，但被 `start.sh --check` 直接调用，不能当作临时产物删除。

当前技术框架沿用原包：Vue 前端、FastAPI 后端、MySQL、数据集版本和角色权限体系。没有重写登录系统，也没有改用 SQLite 作为正式数据库。

## 2. 主要功能变化总览

| 模块 | 相对原包的关键变化 | 主要入口/路径 |
| --- | --- | --- |
| Linux 运行 | 独立副本、可配置路径、CPU Web 环境、模型独立环境、前台启动与预检 | [start.sh](start.sh)、[backend/app/config.py](backend/app/config.py)、[LINUX.md](LINUX.md) |
| 少样本生成 | 分组上传/选择素材、训练与生成、模型/方法选择、失败重试、自动标注及结果回流 | [FewshotView.vue](frontend/src/views/FewshotView.vue)、[fewshot_service.py](backend/app/services/fewshot_service.py)、[integrations/fewshot](integrations/fewshot/README.md) |
| 本地素材 | 接入 118 现有数据，按类别或 caption 检索、预览和选取 | [local_materials.py](backend/app/services/local_materials.py)、[build_material_index.py](scripts/build_material_index.py) |
| 下载与 AIGC 补充 | 保存查询，带入候选素材，关联生成任务/结果，可选具体版本及继续预处理 | [DownloadView.vue](frontend/src/views/DownloadView.vue)、[download.py](backend/app/routers/download.py)、[download_service.py](backend/app/services/download_service.py) |
| 在线标注 | 区分“在线标注”和“导入标注”，从当前数据进入工作台，兼容多模态既有结果 | [data.py](backend/app/routers/data.py)、[ManualLabelView.vue](frontend/src/views/ManualLabelView.vue)、[annotationAdapter.js](frontend/src/annotationAdapter.js) |
| 视频标注 | 真视频播放、SAM mask、可改结果框/点、CoTracker 轨迹、后台自动处理、叠加预览与独立导出 | [video_annotation.py](backend/app/services/video_annotation.py)、[VideoAssist.vue](frontend/src/components/VideoAssist.vue)、[视频说明](integrations/video_annotation/README.md) |
| 数据处理正确性 | 来源同名文件隔离、标签同步变换、版本可用性、失败处理、算法与参数校验 | [preprocess.py](backend/app/services/preprocess.py)、[version_service.py](backend/app/services/version_service.py)、[label_transform.py](backend/app/services/label_transform.py) |
| 权限与交付 | 私有生成结果及派生版本隔离，管理员可查看任务，下载/服务共享访问检查 | [data_access.py](backend/app/services/data_access.py)、[datasets.py](backend/app/routers/datasets.py)、[service.py](backend/app/routers/service.py) |

## 3. Linux 适配与部署

- `backend/app/config.py` 支持 `DP_CONFIG_PATH`；相对路径按项目根目录解析，Linux 下阻止误用 Windows 盘符路径。保留原项目 Windows 兼容分支。
- 新增 Linux 基础依赖、锁定清单、可选模型依赖及素材索引依赖。Web 环境与 FLUX/SD/视频模型环境分开，Web 进程不占用 CUDA。
- `start.sh` 使用本项目虚拟环境，设置数据盘临时目录，前台启动后端；`--check` 执行 Linux 预检。`frontend/vite.config.js` 支持局域网监听。
- `backend/app/main.py` 注册少样本和视频接口、处理启动恢复，并对 API 返回设置私有且不缓存；Windows 自动开浏览器行为不套用到 Linux。
- 118 使用既有 MySQL 实例和独立模型环境。模型、数据集、日志、任务产物不随源码打包；实际机器配置不能直接用示例文件覆盖。

正式平台入口：`http://10.176.53.118:8000/`。在本机 tmux `8000` 窗口中使用的前台启动命令：

```bash
ssh -t 118 'bash /mnt/data/gaoya/agent-data/apps/data-platform3-linux/start.sh'
```

本次仅清理测试产物和整理文档，没有重启服务。

### 本地与 118 的差异

本轮清理前的 191 个文件中，189 个哈希一致，仅 `config.json`、`start.sh` 不同；前者是数据库/模型/素材等机器配置，后者主要是 `/data/...` 与 `/mnt/data/...` 临时目录差异。两端清理相同的两个验收入口并同步本文及累计记录后，应继续仅保留这两个已知环境差异。

118 另有 `scripts/start_mysql.sh`，用于该机器的 MySQL 实例管理，本地没有对应文件，不计入本地相对 tar 的新增数。它不是新增业务功能。

运行目录：本地 `/data/gaoya/agent-data/outputs_v1/data_platform3_linux/`；118 `/mnt/data/gaoya/agent-data/outputs_v1/data_platform3_linux/`。其中 `runtime/Uploads` 为正式上传、处理版本和标注任务数据，`runtime/Logs` 为运行日志，`mysql` 为数据库实例相关文件；不能作为普通测试目录清空。

## 4. 少样本训练、素材与生成结果

### 4.1 可选方法与执行流程

| 方法 | 底座模型 | 实现语义 |
| --- | --- | --- |
| BeyondObjects | FLUX.2 Klein Base 4B | 多组素材联合训练一个 LoRA，再按提示词生成 |
| BeyondObjects | Stable Diffusion 2.1 | 对应 SD2.1 训练/生成流程 |
| ChimeraLoRA | Stable Diffusion 2.1 | 保留共享 A、多个 B 头的原生结构，不等同于普通单 LoRA |

不是任意模型和方法都可组合；ChimeraLoRA 没有被包装为 FLUX 方案。训练代码来自所接入仓库及其上游，来源与修订说明见 [THIRD_PARTY.md](integrations/fewshot/THIRD_PARTY.md) 和 [vendor/sources.json](integrations/fewshot/vendor/sources.json)。新增 `vendor` 文件属于运行所需算法实现，不能按中间结果删除。

用户按组上传图像，也可选本地素材；每组填写共用的“训练文本提示词（caption）”。创建任务时保存素材、参数与来源快照。训练后生成图像，登记 AIGC 来源，全部进入现有 COCO 80 类自动目标框标注，再进入人工修正和版本下载。已完成的适配器可复用；任务保留失败原因及重试入口，不只显示外层 `CalledProcessError`。

`fewshot_worker.py` 提供独立 tmux 队列；`fewshot_service.py` 管平台任务、权限、模型选择、调用、产物登记与标注。`integrations/fewshot/fewshot/` 管训练/生成命令、参数、报告和实验记录。正式训练记录 W&B 配置、进度和指标，不上传权重。

118 任务按现有设置使用 GPU7，可与既有任务共享；启动前检查指定卡、空闲显存与预留量，并在执行时再次检查。显存门槛不是独占锁，也不能保证共享负载下永不 OOM。

### 4.2 已接素材与展示

118 配置的素材入口包括 `rareplanes`、`rareplanes_real`、`aircraft`、`laionsg`、`hpdv3`、`phyco_frames`、`physics_switches`。路径来自配置白名单；SQLite 素材索引用于检索和类别/caption 展示，不替代业务 MySQL。

`build_material_index.py` 建索引，`extend_118_materials.py` 扩展素材入口，`import_118_demos.py` 接入可展示的小型数据子集。历史接入验收记录包含 32 个演示数据集、698 个样本（图像 368、视频 104、文本 226）；这是当时导入规模，不是当前平台所有数据的实时总数。

原生类别/caption 用于素材组织，不能冒充已有目标框。Aircraft 的细粒度机型类别也不能由 COCO 的 `airplane` 框自动补齐。

## 5. 数据下载、AIGC 补充与用户隔离

1. 下载页面保存本次查询和命中的候选文件；“AIGC 补充”进入少样本页面并带入素材。
2. 用户确认候选素材，可增删分组、修改训练文本和数据；跳转本身不会启动训练。
3. 生成任务关联本次下载请求；结果回到对应请求下，显示任务状态和结果版本。
4. 自动标注后的结果允许直接下载，并提示待人工核验；用户按需进入人工修正、清洗、转换、采样等处理，再选择下载版本。

`DownloadRequest` 保存查询和候选来源；`FewshotJob.download_request_id` 关联补充任务。新增 `data_access.py` 将私有素材、生成结果、数据集及派生版本的归属检查复用于列表、预览、下载和任务接口；管理员可查看其他用户任务。原有共享数据规则与角色权限继续保留，不将所有原始数据改为个人私有。

版本未完成、失败或无有效产物时，不应成为可下载/发布/继续处理的完成版本。ZIP 改为数据盘临时文件，避免整个包常驻内存；下载使用与数据内容匹配的标签，并纳入视频独立导出文件。

## 6. 在线标注及视频工作台的最终逻辑

### 6.1 原始数据与多模态入口

原数据条目的“在线标注”进入对应工作台；“导入标注”才要求上传标注文件。`annotation-context` 建立或复用该数据的处理上下文，校验访问权限，失败时回滚不完整副本。并未放开原有首次创建个人工作集的所有角色权限。

`annotationAdapter.js` 兼容图像框、视频逐帧结果、语音 segments、文本实体、雷达 detections，并保留未修改标注及来源元数据；未知几何格式不盲目覆盖。文本保留编码/换行语义并处理 Unicode 与浏览器字符偏移差异；语音支持播放核对与转写文本编辑；雷达框回显按已有 BEV 坐标转换。

### 6.2 视频：准备、确认、后台处理

当前流程已替代早期“每确认一个物体立即自动提交”的方案：

1. 选择“逐帧手动标注”或“首帧手动标注＋后续自动标注”。
2. 多选最终需要的文件：目标框、分割 mask、点轨迹。
3. 一个类别下拉栏，首项为自定义类别；类别保存到个人标签库。界面不要求用户通过编号区分物体，内部仍保留跟踪身份。
4. 初次画框是给 SAM 的提示。SAM 生成 mask，以其外接框作为默认结果框，并保留原始 `prompt_xyxy`。
5. 自动处理前可修改结果框、添加/擦除 mask、调整轨迹点。独立修改框不重新调用 SAM，不裁剪 mask，也不清空点；改 mask 时，默认框随之更新，手动修正过的框保留。
6. 轨迹点默认从完整 mask 采样，每个物体分别计算，默认上限 8、最多 32。保留上方“调整采样点”：点击点删除、点击画面添加。旧的“重新采样”“清空后手动选点”按钮已删除；选择轨迹但点数为零时禁止确认。
7. “确认标注”只加入人工准备清单。逐帧模式保存并预览；首帧模式确认全部物体后统一点击“开始后台自动标注”。有未确认草稿时禁止提交。
8. 后台接受后冻结输入，允许关闭页面或刷新，在历史任务中查看进度、取消、预览和下载。编辑完成结果后，可显式“重新自动标注”；失败/取消任务可“修改并重试”恢复原输入。

每条已确认结果提供预览、修改、删除；修改使用副本，取消不污染原结果。生成中、失败及完成状态明确显示；过期 SAM 返回和任务轮询不会覆盖新草稿。首帧自动模式的整体“继续修正”保持模式并回到首帧；直接修改后续帧仍是单帧人工修正，不隐式重跑整段。

### 6.3 框、mask、轨迹的来源与预览

| 产物 | 当前来源和语义 |
| --- | --- |
| 首帧目标框 | 默认 SAM mask 外接框；若用户独立修改，则精确保留最终确认框 |
| 后续目标框 | SAM 跟踪 mask 的外接框；是指定物体的跟踪结果，不自动发现新物体 |
| mask | SAM 2.1 分割，人工帧可添加/擦除；按列编码的 COCO 未压缩 RLE，保留孔洞 |
| 点轨迹 | CoTracker3 跟踪用户确认的点，保存坐标与可见性；内部区分 `track_id` 和 `point_id` |
| 预览 | 原视频播放，按已生成类型多选叠加框、mask、轨迹；全不选时仅播放原视频 |
| 导出 | 按选择提供 `_box.json`、`_mask.json`、`_trajectory.json`；完整 `_label.json` 留存编辑和溯源信息 |

预览开关只影响显示，不修改下载内容或生成设置；单物体预览仅显示该物体，完整版本下载包含全部物体。自动输出标为待人工核验。当前预览是在浏览器叠加，不额外生成渲染 MP4。

视频模型通过平台子进程统一管理，复用 `AutoLabelTask` 和 `DatasetVersion`，未部署 CVAT，也未新增常驻 GPU 模型服务。SAM 2.1 Hiera Tiny 和 CoTracker3 使用 118 的独立环境及 GPU7；视频任务串行排队、显存不足拒绝启动。旧单类型接口保留兼容，`selected_detection.py` 不是当前组合流程计算后续框的默认方式。

## 7. 原有模块的逻辑修复与功能收敛

| 问题 | 当前改动 | 主要路径 |
| --- | --- | --- |
| 多来源同名文件互相覆盖 | 按来源保留独立相对路径，继承对应标签，复制失败回滚 | `routers/datasets.py`、`services/preprocess.py` |
| 失败版本成为默认输入 | 成功且非空后切换当前版本；列表、下载和下游共享可用性检查 | `services/version_service.py`、`routers/tasks.py`、`VersionSelect.vue` |
| 并发版本目录冲突 | 数据集行锁分配版本号、目录独占，禁止删除已占用目标目录 | `services/version_service.py` |
| 标签和变换后数据错位 | 图像框随缩放；文本清洗映射实体偏移；不支持的几何格式拒绝 | `services/label_transform.py`、`services/preprocess.py` |
| 清洗破坏结构/孤儿标签 | 区分结构化文本，不逐行去重破坏 JSON；只为保留数据附标签 | `services/preprocess.py` |
| 错误输出被算作完成 | 损坏、部分失败、空输出及不支持的转换不能伪装成功 | `services/preprocess.py`、`services/label_schema.py` |
| 下载搭配错误历史标签 | 校验成功版本和数据内容；修正原数据 ID 与处理数据集 ID 混用 | `services/label_locator.py`、`services/download_service.py` |
| 类别检索漏掉嵌套结果 | 递归提取类别、文本分类与真实标注，不以摘要替代类别 | `services/label_schema.py`、`routers/download.py` |
| 假算法/不合法参数可提交 | 模态与能力白名单、参数范围、模型依赖检查，单任务与工作流共享校验 | `routers/tasks.py`、`services/preprocess.py` |
| 模型目录存在就宣称可用 | Whisper/GLiNER 增加 Python 依赖检查；旧占位模型入口收起 | `whisper_service.py`、`gliner_service.py` |
| 页面功能名称超过实现 | 暂收起未实现的复杂融合、特征变换、时间/分层/K折采样等，保留实际可执行操作 | `IntegrateView.vue`、`TransformView.vue`、`SampleView.vue`、`WorkflowView.vue` |
| 雷达图表物理含义错误 | 不把 z 坐标叫多普勒，不把距离排序叫真实轨迹；CSV/PLY 保留强度字段 | `VizDetailView.vue`、`services/preprocess.py` |
| 质量分容易被理解为标注精度 | 改称内容有效性；按完整路径和合法标签计算配套率 | `QualityPanel.vue`、`services/quality_service.py` |
| 人工实体导致图谱供数错误 | 兼容人工实体，保证边端点存在；不宣称真实关系推理 | `routers/service.py` |
| 术语密级与访问不一致 | 术语检索/检查按密级过滤，数据服务使用相同访问约束 | `routers/standards.py`、`routers/service.py` |
| 只加载前 100 个工作集 | 分页加载，版本请求限制并发；跨数据集/版本清理编辑上下文 | `composables.js`、`ManualLabelView.vue` |
| Web 重启遗留运行中状态 | 中断的普通预处理/视频任务明确失败，可恢复视频输入重试 | `backend/app/main.py`、`services/video_annotation.py` |

表内后端简写均相对 `backend/app/`，前端组件/视图的完整路径见第 11 节。历史审计中的“待完善”不能直接当成现在全部仍未修复，也不能因一部分修复就认为整个待办已完成。

## 8. 数据库变化与原有能力边界

| 对象 | 变化 | 对应文件 |
| --- | --- | --- |
| `fewshot_jobs` | 新增少样本任务、输入分组/选项、执行状态、来源和结果关联 | [models/fewshot.py](backend/app/models/fewshot.py)、[0004_fewshot.py](backend/alembic/versions/0004_fewshot.py) |
| `download_requests` | 新增下载查询、命中文件、补充数量及用户归属 | [models/download_request.py](backend/app/models/download_request.py) |
| `annotation_labels` | 新增个人类别库；`user_id + name` 唯一 | [models/user.py](backend/app/models/user.py) |
| `fewshot_jobs.download_request_id` | 存量表补列、创建关联索引 | [database.py](backend/app/database.py) |
| 模型注册 | 注册新增表以参加既有建表流程 | [models/__init__.py](backend/app/models/__init__.py) |

新表沿用 `create_all`，少样本另有 Alembic 迁移；不是所有后续表都新增了独立 SQL/Alembic 文件。部署时需核实模型注册和存量列迁移，不能认为只运行 `0004` 就包含所有后续调整。业务 schema 版本号仍沿用原值，没有据此宣称已建立完整的新迁移体系。

原包已有的登录、安全策略、用户管理、数据模型和版本模型大体沿用。`routers/auth.py`、`routers/users.py`、`security.py`、`models/data.py`、`models/dataset.py` 及前端用户/登录页面在本次字节比较中未改变。历史账号重置属于数据库操作，不是新增登录代码；本文不记录凭据。

## 9. 验证证据与仍未实现的范围

以下是既有验收记录，本次文档/清理没有重跑训练或 GPU 模型：

| 历史证据目录（相对项目输出根） | 已验证范围 |
| --- | --- |
| `20261013_integration_validation` | FLUX 短 smoke 训练、生成、复用、自动标注、人工修正和下载闭环；不是正式训练精度评估 |
| `20261013_all_methods` | 方法/模型选择、配置及 CPU 合约检查；不是三套方法全部完成正式训练 |
| `20261013_online_annotation` | 原数据在线标注入口、权限和页面交互 |
| `20261013_logic_fixes` | 数据/标签变换、版本、失败处理、接口和浏览器回归 |
| `20261014_video_background` | 组合后台流程、叠加预览；118 GPU7 两帧真实 SAM/CoTracker 流程验收 |
| `20261014_video_point_controls` | 精简选点、取消恢复、确认/保存后的重新自动标注入口，共 18 项浏览器检查 |
| `20261014_full_change_audit` | 本轮 tar 哈希、逐文件清单、两端差异、清理清单及正式数据前后校验 |

保留检查脚本是为了后续 Windows/Linux 修改可回归，并非页面运行产物。验收脚本及其固定测试路径可能需要重新准备隔离数据，不能在正式库上直接执行。

当前主要限制：

- 视频自动处理面向固定帧率，单次最多 300 帧、10 个物体、每物体 32 点；不表示长视频全部已处理。尚无 VFR/PTS 精确支持、自动发现中途新物体、自动转码、MOT 专用导出。
- 视频/普通预处理后台仍依赖 Web 进程：关闭页面可继续，重启 Web 会中断；明确失败和恢复输入重试不等于计算断点续跑。少样本独立队列的生命周期不同。
- 仍需完善来源分组防训练/测试泄漏、逐文件审核覆盖、查询补足是否达标的统计、原始数据聚合口径，以及通用 polygon/mask 几何转换。
- 质量有效性分数不等于有真值的标注准确率；自动生成数量不等于细粒度类别需求达标；标准代理分数不等于已执行全部标准条款。
- 未完成通用特征/决策/语义融合、真实关系抽取、标准/外部回写全闭环、所有模型加载与格式的完整验收、生产高并发和大包压力测试。
- `config.json.example` 未覆盖 118 所有实际素材索引和视频模型配置；旧 `LINUX.md`、审计/待办中的日期性记录需结合当前源码及最新视频说明阅读。
- 用户提出的 Windows↔Linux 一键生成副本脚本尚未实现；原 `start.bat` 和 Windows 原目录未自动获得全部 Linux 版新功能。不能把“保留 Windows 兼容代码”写成已完成双平台同步交付。

## 10. 本轮清理范围

按用户确认，仅清理两端测试/验收标注和中间产物，保留正式用户标注及数据集自带原始标签。

| 类型 | 本地删除数 | 118 删除数 |
| --- | ---: | ---: |
| 隔离验收标注副本 | 154 | 56 |
| 隔离验收 SQLite 数据库 | 19 | 11 |
| 验收导出副本/下载包 | 14 | 2 |
| 临时发布/候选代码包 | 23 | 23 |
| 一次性验收启动脚本 | 2 | 2 |
| 合计 | **212** | **94** |

总计释放 52,932,334 字节（约 50.5 MiB）。删除的代码目录文件为 `backend/tests/start_fewshot_smoke.py`、`scripts/serve_download_acceptance.py`；它们原本就是 Linux 版新增，所以相对 Windows tar 的“删除文件数”仍为 0。

正式库、正式任务目录、素材索引/目录、原始数据未清理。118 的 1,264 个正式标注/结果文件哈希，以及 13 张数据/任务相关表的内容摘要，清理前后均一致。没有删除正式用户自定义类别。配置、日志、指标、Markdown、验收结论、`source_before`/`production_before` 回退备份继续保留；历史说明中出现的测试下载包路径可能已按本节清理。

清理使用明确目录清单、删除前哈希、正式数据路径排除和活动进程检查，不对所有 `label` 文件进行全盘删除。两端审计目录的 `cleanup_plan.json` 保存逐文件路径/大小/哈希/原因，`cleanup_result.json` 保存实际结果；移除的两个小脚本另在审计目录保留回退包。

## 11. 完整新增/修改文件清单

路径相对项目根目录，可点击查看。下表覆盖清理后相对原 tar 的全部 89 个新增文件与 40 个修改文件，包括本文；61 个未改变文件不逐个重复列出。“新增”表示原 tar 中没有，并不表示该文件全部为原创算法。

### 平台根目录与部署

| 路径 | 状态 | 职责/改动 |
| --- | --- | --- |
| [AGENTS.md](AGENTS.md) | 新增 | 规定累计记录、同步及凭据保护要求 |
| [CHANGELOG_LINUX.md](CHANGELOG_LINUX.md) | 新增 | 累计功能、修复、验收与部署历史 |
| [LINUX.md](LINUX.md) | 新增 | Linux 环境、启动、数据和模型部署说明 |
| [PLATFORM_LOGIC_AUDIT_20261013.md](PLATFORM_LOGIC_AUDIT_20261013.md) | 新增 | 逐模块逻辑审计与问题证据 |
| [README.md](README.md) | 修改 | 增加 Linux 副本说明和文档入口 |
| [TODO_LOGIC.md](TODO_LOGIC.md) | 新增 | 修复验收目标及尚未完成范围 |
| [VIDEO_ANNOTATION_RESEARCH_20261013.md](VIDEO_ANNOTATION_RESEARCH_20261013.md) | 新增 | 视频标注工具与模型调研记录 |
| [config.json](config.json) | 修改 | 本地机器路径与少样本运行配置；不记录敏感值 |
| [config.json.example](config.json.example) | 修改 | Linux 可用配置示例及少样本配置骨架 |
| [start.sh](start.sh) | 修改 | Linux 虚拟环境、CPU Web、临时目录、前台启动及预检 |
| [CHANGES_FROM_WINDOWS_TAR.md](CHANGES_FROM_WINDOWS_TAR.md) | 新增 | 本次基线对比、完整路径清单与清理说明 |

### 后端模型、接口与服务

| 路径 | 状态 | 职责/改动 |
| --- | --- | --- |
| [backend/alembic/versions/0004_fewshot.py](backend/alembic/versions/0004_fewshot.py) | 新增 | 少样本任务表迁移 |
| [backend/app/config.py](backend/app/config.py) | 修改 | 配置覆盖、跨平台路径解析、模型/少样本配置 |
| [backend/app/database.py](backend/app/database.py) | 修改 | 补齐下载请求关联列及索引 |
| [backend/app/fewshot_worker.py](backend/app/fewshot_worker.py) | 新增 | 独立少样本队列执行入口 |
| [backend/app/main.py](backend/app/main.py) | 修改 | 注册新接口、启动恢复、缓存策略与 Linux 启动行为 |
| [backend/app/models/__init__.py](backend/app/models/__init__.py) | 修改 | 注册新增数据库模型 |
| [backend/app/models/download_request.py](backend/app/models/download_request.py) | 新增 | 下载查询及候选素材记录 |
| [backend/app/models/fewshot.py](backend/app/models/fewshot.py) | 新增 | 少样本任务、来源、结果及下载关联模型 |
| [backend/app/models/user.py](backend/app/models/user.py) | 修改 | 新增个人可复用标注类别表 |
| [backend/app/routers/data.py](backend/app/routers/data.py) | 修改 | 原数据在线标注上下文、原文件预览和访问检查 |
| [backend/app/routers/datasets.py](backend/app/routers/datasets.py) | 修改 | 来源隔离、版本可用性、标签及版本下载 |
| [backend/app/routers/download.py](backend/app/routers/download.py) | 修改 | 查询保存、AIGC 候选素材、版本化下载 |
| [backend/app/routers/fewshot.py](backend/app/routers/fewshot.py) | 新增 | 少样本任务、模型选项、本地素材和结果接口 |
| [backend/app/routers/service.py](backend/app/routers/service.py) | 修改 | 供数权限、图谱实体适配及磁盘打包 |
| [backend/app/routers/standards.py](backend/app/routers/standards.py) | 修改 | 术语密级过滤和语义相关校验 |
| [backend/app/routers/tasks.py](backend/app/routers/tasks.py) | 修改 | 任务能力/参数校验、工作流、版本及人工标注提交 |
| [backend/app/routers/video_annotation.py](backend/app/routers/video_annotation.py) | 新增 | 视频输入校验、任务提交/状态/取消/输入恢复及权限 |
| [backend/app/services/data_access.py](backend/app/services/data_access.py) | 新增 | 原始/私有/派生数据共用的用户访问检查 |
| [backend/app/services/download_service.py](backend/app/services/download_service.py) | 新增 | 下载请求、候选素材及关联结果组织 |
| [backend/app/services/fewshot_service.py](backend/app/services/fewshot_service.py) | 新增 | 少样本编排、GPU 准入、来源快照、结果登记与自动标注 |
| [backend/app/services/file_service.py](backend/app/services/file_service.py) | 修改 | 文本预览保留规范化内容与更完整的字符范围 |
| [backend/app/services/gliner_service.py](backend/app/services/gliner_service.py) | 修改 | GLiNER 可用性增加 Python 包依赖检查 |
| [backend/app/services/label_locator.py](backend/app/services/label_locator.py) | 修改 | 原始/版本标识修正及成功且内容匹配的标签定位 |
| [backend/app/services/label_schema.py](backend/app/services/label_schema.py) | 新增 | 标签有效性、嵌套类别和标注信息提取 |
| [backend/app/services/label_transform.py](backend/app/services/label_transform.py) | 新增 | 文本编码/偏移映射、图像框几何变换和标签保护 |
| [backend/app/services/local_materials.py](backend/app/services/local_materials.py) | 新增 | 配置白名单素材、索引、类别/caption、分页及图像读取 |
| [backend/app/services/preprocess.py](backend/app/services/preprocess.py) | 修改 | 模态处理、标签变换、参数白名单、失败与产物检查 |
| [backend/app/services/quality_service.py](backend/app/services/quality_service.py) | 修改 | 内容有效性、完整路径标签配套率与非法标签排除 |
| [backend/app/services/version_service.py](backend/app/services/version_service.py) | 修改 | 版本分配锁、目录保护、完成后切换及可用性检查 |
| [backend/app/services/video_annotation.py](backend/app/services/video_annotation.py) | 新增 | 视频模型子进程、队列、任务结果及版本合并 |
| [backend/app/services/video_exports.py](backend/app/services/video_exports.py) | 新增 | 按输出类型整理框、mask、轨迹下载文件 |
| [backend/app/services/whisper_service.py](backend/app/services/whisper_service.py) | 修改 | Whisper 可用性增加 Python 包依赖检查 |

### 后端依赖与回归检查

| 路径 | 状态 | 职责/改动 |
| --- | --- | --- |
| [backend/requirements-linux-optional.txt](backend/requirements-linux-optional.txt) | 新增 | 独立可选模型功能依赖 |
| [backend/requirements-linux.lock.txt](backend/requirements-linux.lock.txt) | 新增 | Linux 基础环境依赖锁定版本 |
| [backend/requirements-linux.txt](backend/requirements-linux.txt) | 新增 | Linux CPU Web 基础依赖 |
| [backend/requirements-material-index.txt](backend/requirements-material-index.txt) | 新增 | 大规模素材元数据索引依赖 |
| [backend/tests/check_demo_imports.py](backend/tests/check_demo_imports.py) | 新增 | 演示数据导入检查 |
| [backend/tests/check_download_fewshot.py](backend/tests/check_download_fewshot.py) | 新增 | 下载查询、少样本关联和权限回归 |
| [backend/tests/check_fewshot.py](backend/tests/check_fewshot.py) | 新增 | 少样本任务基础接口回归 |
| [backend/tests/check_fewshot_engines.py](backend/tests/check_fewshot_engines.py) | 新增 | 方法/模型组合与引擎配置检查 |
| [backend/tests/check_gpu_admission.py](backend/tests/check_gpu_admission.py) | 新增 | 共享 GPU 显存准入边界检查 |
| [backend/tests/check_local_materials.py](backend/tests/check_local_materials.py) | 新增 | 本地素材白名单及选择接口检查 |
| [backend/tests/check_material_metadata.py](backend/tests/check_material_metadata.py) | 新增 | 类别与 caption 元数据检查 |

### 前端页面与组件

| 路径 | 状态 | 职责/改动 |
| --- | --- | --- |
| [frontend/src/annotationAdapter.js](frontend/src/annotationAdapter.js) | 新增 | 多模态标注读写、帧/物体身份及未修改信息保留 |
| [frontend/src/components/PreviewPane.vue](frontend/src/components/PreviewPane.vue) | 修改 | 视频使用实际播放器，文本等预览适配 |
| [frontend/src/components/TaskTable.vue](frontend/src/components/TaskTable.vue) | 修改 | 显示取消状态 |
| [frontend/src/components/VersionSelect.vue](frontend/src/components/VersionSelect.vue) | 修改 | 不可用版本禁选与状态提示 |
| [frontend/src/components/VideoAssist.vue](frontend/src/components/VideoAssist.vue) | 新增 | 自动任务提交、状态轮询、取消及过期返回保护 |
| [frontend/src/components/VideoPreview.vue](frontend/src/components/VideoPreview.vue) | 新增 | 统一原视频播放和错误提示组件 |
| [frontend/src/composables.js](frontend/src/composables.js) | 修改 | 数据集分页、版本并发限制和下载来源上下文 |
| [frontend/src/router/index.js](frontend/src/router/index.js) | 修改 | 少样本页面路由 |
| [frontend/src/videoMask.js](frontend/src/videoMask.js) | 新增 | mask RLE、画笔/擦除、采样点及筛选 |
| [frontend/src/views/AutoLabelView.vue](frontend/src/views/AutoLabelView.vue) | 修改 | 能力就绪提示及下载上下文 |
| [frontend/src/views/CleanView.vue](frontend/src/views/CleanView.vue) | 修改 | 清洗操作及下载来源上下文衔接 |
| [frontend/src/views/DataTypeView.vue](frontend/src/views/DataTypeView.vue) | 修改 | 原数据在线标注/导入标注分离及预览入口 |
| [frontend/src/views/DownloadView.vue](frontend/src/views/DownloadView.vue) | 修改 | 保存查询、AIGC 补充结果、版本和后续预处理入口 |
| [frontend/src/views/FewshotView.vue](frontend/src/views/FewshotView.vue) | 新增 | 分组素材、训练文本、模型方法、任务及结果界面 |
| [frontend/src/views/IntegrateView.vue](frontend/src/views/IntegrateView.vue) | 修改 | 收敛为实际文件集合并、保持返回下载上下文 |
| [frontend/src/views/MainLayout.vue](frontend/src/views/MainLayout.vue) | 修改 | 少样本导航、窄屏布局及下载返回入口 |
| [frontend/src/views/ManualLabelView.vue](frontend/src/views/ManualLabelView.vue) | 修改 | 多模态人工工作台、视频准备/确认/后台/修正/叠加预览 |
| [frontend/src/views/QualityPanel.vue](frontend/src/views/QualityPanel.vue) | 修改 | 有效性命名与质量入口的来源上下文 |
| [frontend/src/views/SampleView.vue](frontend/src/views/SampleView.vue) | 修改 | 收起未实现采样方式并保留上下文 |
| [frontend/src/views/TransformView.vue](frontend/src/views/TransformView.vue) | 修改 | 实际支持的转换方式、格式限制及清晰命名 |
| [frontend/src/views/VizDetailView.vue](frontend/src/views/VizDetailView.vue) | 修改 | 原视频播放及雷达图表物理含义修正 |
| [frontend/src/views/WorkflowView.vue](frontend/src/views/WorkflowView.vue) | 修改 | 工作流能力检查提示与来源上下文 |
| [frontend/vite.config.js](frontend/vite.config.js) | 修改 | 局域网访问监听设置 |

### 少样本模型集成

| 路径 | 状态 | 职责/改动 |
| --- | --- | --- |
| [integrations/fewshot/.gitignore](integrations/fewshot/.gitignore) | 新增 | 少样本子项目生成物忽略规则 |
| [integrations/fewshot/README.md](integrations/fewshot/README.md) | 新增 | 独立少样本工具使用说明 |
| [integrations/fewshot/THIRD_PARTY.md](integrations/fewshot/THIRD_PARTY.md) | 新增 | 算法来源、修订和上游说明 |
| [integrations/fewshot/configs/chimera.yaml](integrations/fewshot/configs/chimera.yaml) | 新增 | ChimeraLoRA SD2.1 配置 |
| [integrations/fewshot/configs/flux2.yaml](integrations/fewshot/configs/flux2.yaml) | 新增 | BeyondObjects FLUX.2 配置 |
| [integrations/fewshot/configs/rareplanes_sd21_example.yaml](integrations/fewshot/configs/rareplanes_sd21_example.yaml) | 新增 | RarePlanes 的 SD2.1 使用配置示例 |
| [integrations/fewshot/configs/sd21.yaml](integrations/fewshot/configs/sd21.yaml) | 新增 | BeyondObjects SD2.1 配置 |
| [integrations/fewshot/fewshot/__init__.py](integrations/fewshot/fewshot/__init__.py) | 新增 | 少样本工具包入口 |
| [integrations/fewshot/fewshot/cli.py](integrations/fewshot/fewshot/cli.py) | 新增 | 训练/生成 CLI 与平台调用适配 |
| [integrations/fewshot/fewshot/core.py](integrations/fewshot/fewshot/core.py) | 新增 | 参数、路径、运行准备及内存约束 |
| [integrations/fewshot/fewshot/generation.py](integrations/fewshot/fewshot/generation.py) | 新增 | 生成执行与输出元数据 |
| [integrations/fewshot/fewshot/report.py](integrations/fewshot/fewshot/report.py) | 新增 | 运行报告整理 |
| [integrations/fewshot/fewshot/tracking.py](integrations/fewshot/fewshot/tracking.py) | 新增 | 实验日志/W&B 与路径管理 |
| [integrations/fewshot/fewshot/training.py](integrations/fewshot/fewshot/training.py) | 新增 | 三类训练方案的命令与流程组织 |
| [integrations/fewshot/requirements/flux2.txt](integrations/fewshot/requirements/flux2.txt) | 新增 | FLUX.2 模型环境依赖 |
| [integrations/fewshot/requirements/sd21.txt](integrations/fewshot/requirements/sd21.txt) | 新增 | SD2.1 模型环境依赖 |
| [integrations/fewshot/run.py](integrations/fewshot/run.py) | 新增 | 少样本独立执行入口 |
| [integrations/fewshot/scripts/package.py](integrations/fewshot/scripts/package.py) | 新增 | 少样本工具打包 |
| [integrations/fewshot/scripts/verify_sources.py](integrations/fewshot/scripts/verify_sources.py) | 新增 | 上游来源核验 |
| [integrations/fewshot/tests/test_contracts.py](integrations/fewshot/tests/test_contracts.py) | 新增 | 配置和命令合约回归 |
| [integrations/fewshot/tests/test_generation_metadata.py](integrations/fewshot/tests/test_generation_metadata.py) | 新增 | 生成元数据回归 |
| [integrations/fewshot/tests/test_output_directory.py](integrations/fewshot/tests/test_output_directory.py) | 新增 | 输出目录行为回归 |
| [integrations/fewshot/tests/test_platform_launch_failure.py](integrations/fewshot/tests/test_platform_launch_failure.py) | 新增 | 启动失败信息回归 |
| [integrations/fewshot/tests/test_platform_memory_guard.py](integrations/fewshot/tests/test_platform_memory_guard.py) | 新增 | 内存约束回归 |
| [integrations/fewshot/tests/test_tracking_paths.py](integrations/fewshot/tests/test_tracking_paths.py) | 新增 | 实验记录路径回归 |
| [integrations/fewshot/vendor/beyondobjects/config.py](integrations/fewshot/vendor/beyondobjects/config.py) | 新增 | 接入的 BeyondObjects 配置实现 |
| [integrations/fewshot/vendor/beyondobjects/finetune_t2i.py](integrations/fewshot/vendor/beyondobjects/finetune_t2i.py) | 新增 | 接入的 BeyondObjects SD 训练实现 |
| [integrations/fewshot/vendor/beyondobjects/util.py](integrations/fewshot/vendor/beyondobjects/util.py) | 新增 | BeyondObjects 辅助实现 |
| [integrations/fewshot/vendor/chimera/main_chimera.py](integrations/fewshot/vendor/chimera/main_chimera.py) | 新增 | 接入的 ChimeraLoRA 原生多头训练实现 |
| [integrations/fewshot/vendor/chimera/util.py](integrations/fewshot/vendor/chimera/util.py) | 新增 | ChimeraLoRA 辅助实现 |
| [integrations/fewshot/vendor/flux2/diffusers_revision.json](integrations/fewshot/vendor/flux2/diffusers_revision.json) | 新增 | FLUX 训练器依赖来源修订 |
| [integrations/fewshot/vendor/flux2/model_files.json](integrations/fewshot/vendor/flux2/model_files.json) | 新增 | FLUX 模型文件清单 |
| [integrations/fewshot/vendor/flux2/model_revision.json](integrations/fewshot/vendor/flux2/model_revision.json) | 新增 | FLUX 模型修订信息 |
| [integrations/fewshot/vendor/flux2/official_train.py](integrations/fewshot/vendor/flux2/official_train.py) | 新增 | 接入的 FLUX 官方训练器实现 |
| [integrations/fewshot/vendor/sources.json](integrations/fewshot/vendor/sources.json) | 新增 | 外部源码来源清单 |

### 视频模型集成

| 路径 | 状态 | 职责/改动 |
| --- | --- | --- |
| [integrations/video_annotation/README.md](integrations/video_annotation/README.md) | 新增 | 当前视频工作台语义、模型、导出和边界说明 |
| [integrations/video_annotation/mask_codec.py](integrations/video_annotation/mask_codec.py) | 新增 | 模型侧 mask RLE 编解码 |
| [integrations/video_annotation/point_tracking.py](integrations/video_annotation/point_tracking.py) | 新增 | CoTracker 点轨迹推理 |
| [integrations/video_annotation/run.py](integrations/video_annotation/run.py) | 新增 | 视频模型执行入口及组合流程 |
| [integrations/video_annotation/selected_detection.py](integrations/video_annotation/selected_detection.py) | 新增 | 旧单类型目标检测接口辅助实现 |

### 检查与数据维护脚本

| 路径 | 状态 | 职责/改动 |
| --- | --- | --- |
| [scripts/build_material_index.py](scripts/build_material_index.py) | 新增 | 构建素材类别/caption 检索索引 |
| [scripts/check_annotation_adapter.mjs](scripts/check_annotation_adapter.mjs) | 新增 | 前端标注格式适配回归 |
| [scripts/check_annotation_outputs.py](scripts/check_annotation_outputs.py) | 新增 | 标注类型导出检查 |
| [scripts/check_annotation_types.py](scripts/check_annotation_types.py) | 新增 | 视频多类型标注接口检查 |
| [scripts/check_linux.py](scripts/check_linux.py) | 新增 | 正式启动脚本调用的 Linux 预检 |
| [scripts/check_logic_fixes.py](scripts/check_logic_fixes.py) | 新增 | 跨模块逻辑修复回归 |
| [scripts/check_online_annotation.py](scripts/check_online_annotation.py) | 新增 | 在线标注接口、权限及版本回归 |
| [scripts/check_video_annotation.py](scripts/check_video_annotation.py) | 新增 | 视频模型接口、取消及结果检查 |
| [scripts/check_video_workflow.py](scripts/check_video_workflow.py) | 新增 | 多物体准备、后台提交和结果恢复回归 |
| [scripts/extend_118_materials.py](scripts/extend_118_materials.py) | 新增 | 扩展 118 素材白名单配置 |
| [scripts/import_118_demos.py](scripts/import_118_demos.py) | 新增 | 将 118 现有数据接为可展示子集 |
| [scripts/test_118_demos.py](scripts/test_118_demos.py) | 新增 | 接入数据的页面/预处理接口验收 |

## 12. 后续维护方式

每次后续修改仍在 `CHANGELOG_LINUX.md` 末尾追加行为、路径、验证和限制；当最终功能、文件结构、数据库或部署方式变化时，同步更新本文相应章节和统计日期。历史验收方案与当前功能明确区分。

以原 tar 为基线复核时，沿用第 1 节排除范围，逐文件比较哈希，不用解压时间或历史记录代替实际差异；本地和 118 的机器配置分别维护。新算法源文件、正式测试工具与临时验收产物分别说明，避免只用“新增文件数”判断平台复杂度。
