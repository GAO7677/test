# 数据中台待完善清单

来源：[2026-10-13 逻辑审计](PLATFORM_LOGIC_AUDIT_20261013.md)。本清单保留原审计验收目标；已验证的修复于2026-10-13部署到118；未勾选项按下方表格保留剩余范围。P0 优先阻止数据或标签错误；P1 使宣称能力与实际行为一致并补齐闭环；P2 改善交付与使用体验。

每项修复后填写日期、涉及文件、验证结果，并同步追加 `CHANGELOG_LINUX.md`；只验证任务成功不算完成，必须满足本项验收要求。

## P0

- [x] **L01 多来源同名文件覆盖**

  现状：新建数据集复制到同一相对路径；两个来源的 same.txt 只剩后一个，数据和标签均可被覆盖。

  建议：按 source_id 保留独立目录或稳定 sample_id；冲突显式拒绝，复制与入库原子提交。

  验收：两个来源同名文件均保留且来源可追溯；失败不留可用的半成品版本。

  证据：R17；datasets.py:create_dataset

- [ ] **L02 几何变化没有同步标签**

  现状：图像标准化缩小尺寸，旧 bbox 原样复制；雷达坐标标准化同样需要更新坐标标签。

  建议：转换返回几何映射，统一变换 bbox/polygon/mask/点云框；不支持时使旧标签失效并要求重标。

  验收：1600×800→640×320 时 [1000,200,1500,600]→[400,80,600,240]，框全部在数据域内。

  证据：R04；preprocess.py:_standardize/_exec_transform

- [x] **L03 清洗破坏文本、结构和标签对应**

  现状：删除空行/重复行后原样复制字符 span；对合法 JSON 按行去重会破坏结构；删除数据文件后仍复制其标注。

  建议：按模态/结构清洗；文本返回 offset_map，JSON/CSV 按记录处理，标签仅跟随保留数据；无法映射时转待重标。

  验收：清洗后 text[start:end] 与标注文本一致；JSON/XML 可解析；无孤儿标签。

  证据：R05/R06；preprocess.py:clean_file/_exec_clean

- [x] **L04 自动标注无法完整进入人工修正**

  现状：人工读取仅兼容 items/annotations/boxes，遗漏语音 segments、文本 entities、雷达 detections；视频嵌套 boxes 不变成可绘制框，保存可能丢掉原结果。

  建议：建立统一标注 schema 和各算法适配器，保留 caption、分类、置信度、来源、帧信息；迁移前禁止覆盖无法解析的标签。

  验收：图像、视频、语音、文本、雷达的自动结果均可回显/修正/保存，无未编辑标签或元数据丢失。

  证据：U01–U04；ManualLabelView.vue:openWorkbench/normAnn

- [x] **L05 人工工作台跨数据集缓存污染**

  现状：labelsMap 和 changed 只按相对文件名缓存；切换数据集/版本后旧 same.txt 标签及待提交项继续存在。

  建议：缓存键使用 dataset_id/version_id/sample_id；切换时提示未保存改动并隔离缓存，提交校验当前上下文。

  验收：两个数据集同名文件互不污染；切换版本不提交上一版本的标签。

  证据：U05；ManualLabelView.vue:onDatasetChange/loadFiles/onSubmit

- [x] **L06 失败或未执行被标为完成**

  现状：损坏图返回 error 字典仍计成功；部分失败无状态；转换异常/不支持时原样复制仍 completed；pipeline 可生成空版本。

  建议：区分 completed/partial_failed/failed，统计每文件成功/失败/未处理及实际算法；未知操作拒绝，空产物拒绝交付；错误记录与标签分开。

  验收：损坏文件不成为有效负样本；未转换不宣称转换成功；下游不接收错误标签或空产物。

  证据：R07/R10/R11；preprocess.py:_exec_auto_label/_exec_transform

- [ ] **L07 普通下载的数据与标签没有固定版本绑定**

  现状：原始文件下载会配下游清洗/缩放版本的标签；定位器还可读失败版本。原始 sidecar 优先时又可能隐藏下游人工修正。

  建议：普通下载显式选择 raw 或 dataset_version；数据和标签取同一版本并固化 hash/版本 ID，定位器仅取可交付版本，禁止跨内容版本补标签。

  验收：原始文本与清洗文本分别配自己的 span；失败版本标签不能被检索/打包；同一快照重复下载一致。

  证据：R21/R22；label_locator.py:find_label_with_path；download.py:download_pack

## P1

- [ ] **L08 视频覆盖范围缺少契约**

  现状：COCO 默认每2秒、最多200个采样帧；零检测帧不记录。帧差算法最多2000帧；可视化最多300帧，长视频无充分截断提示。

  建议：显式选择首帧/抽帧/逐帧；保存 selected_frames、空检测、处理区间、总帧数、覆盖率和截断原因，可分片继续。

  验收：800秒视频不得在398秒处静默结束；每个约定帧都有已处理/空检测/失败状态。

  证据：R01/R03；preprocess.py:label_video；file_service.py:_video_viz

- [ ] **L09 视频人工标注不是逐帧/跟踪工具**

  现状：人工框只存 time，显示容差±0.5秒；没有 frame_index、PTS、track_id、跨帧传播。命中框时也没有按当前帧过滤，播放中未逐帧刷新。

  建议：按任务提供帧步进、帧精确定位与时间轴；当前帧选择/绘制一致；跟踪另存 track_id 和遮挡/可见性，传播须明确模型。

  验收：25fps 相邻帧可分别编辑；当前帧不会选中另一帧隐藏框；轨迹保持同一身份。

  证据：源码确认；ManualLabelView.vue:onBoxStart/onBoxEnd/redraw

- [ ] **L10 采样方式没有真正执行**

  现状：页面四种 method 被路由丢弃，执行始终 random；辅助分层函数按扩展名，time 按文件mtime，K折未实现。

  建议：持久化并传入 method；按任务类别分层、按采集时间/视频序列做时间划分；未实现K折先禁用。

  验收：random/time/stratified 产生各自应有结果；K折输出K组索引与fold_id；不以mtime冒充采集时间。

  证据：R12；tasks.py:create_sample；preprocess.py:_exec_sample/sample_split

- [ ] **L11 划分缺少来源分组和防泄漏**

  现状：文件级随机划分无法约束同一视频帧、同一场景/对象、镜像或同一生成任务；标准化/特征参数可能先在全量数据拟合。

  建议：使用 origin_group_id 做分组划分；冻结真实测试/验证集后拟合 scaler/vocab 和生成补充，默认仅补训练集。

  验收：同一来源组不跨集合；哈希重复不跨集合；测试集不参与LoRA素材选择或统计量拟合。

  证据：源码确认；preprocess.py:sample_split；fewshot 来源/生成清单

- [ ] **L12 版本未声明产物类型**

  现状：图像特征JSON仍属于image数据集，后续标注/质量按图片处理；mixed只按数据集类型分支。

  建议：版本增加 artifact_type/file_modalities、schema与支持操作；每个节点检查输入/输出能力，mixed按文件或子集分派。

  验收：特征版本不能进入图像框标注；视频抽帧明确产生image版本并保留视频来源；mixed可按子模态处理。

  证据：源码确认；preprocess.py:_extract_feature；tasks.py；version模型

- [ ] **L13 算法、参数与数据类型校验不足**

  现状：自动接口接受假算法和与数据集不一致的data_type，假图像算法返回固定中心框；固定切片overlap等缺少范围校验。

  建议：后端由数据集和版本推导类型，注册算法白名单/参数schema/格式适配；非法参数提交前拒绝。

  验收：假算法、图像集配文本算法、overlap≥1、sliceDuration≤0 均返回4xx且不建任务。

  证据：R08/R09；tasks.py:create_auto_label；preprocess.py:label_voice/label_image

- [ ] **L14 模型就绪只检测目录**

  现状：118上Whisper/GliNER规则显示ready，但对应Python依赖缺失；SAM2/SAM3即使放权重也仍未实现推理。

  建议：分开 implemented/dependency_ready/weights_ready/device_ready/usable；模型入口做轻量加载或CPU预检，工作流与单任务一致。

  验收：缺依赖时页面不可提交并指出原因；占位算法不因目录存在而显示可运行。

  证据：runtime_inventory.json；whisper_service.py/gliner_service.py/sam2_service.py/sam3_service.py

- [ ] **L15 算法名称与真实能力不符**

  现状：模板匹配返回中心框；视频跟踪与运动检测共享帧差，未做光流或身份跟踪；场景分段也不是完整分段；基频跟踪只返回全段；人工选择分类/语义分割类型仍只有画框工具，未产生分类表单或像素掩膜。

  建议：暂改名为演示/候选规则或禁用；实现真实算法后再恢复名称；清理未生效参数，公布精确输出定义。

  验收：每个公开算法可证明执行其声明的操作；跟踪含track_id，分段含起止时间，模板有外部模板输入。

  证据：R08及源码确认；preprocess.py:label_image/label_video/label_voice/BUILTIN_RULES

- [ ] **L16 语音转写与人工修正未形成闭环**

  现状：Whisper分支使用未定义file_path；自动segments未回显；工作台只有段选/类别，没有有效的转写文本修正与播放校核。

  建议：修复路径，统一音频解码/VAD/ASR和原始时间映射；添加按段播放、文本编辑；明确段级或词级时间戳。

  验收：WAV/MP3/M4A按已声明范围跑通；自动转写可听取、改文本、保存；不声称word-level而实际segment-level。

  证据：R15/U02；preprocess.py:label_voice；ManualLabelView.vue；whisper_service.py

- [ ] **L17 雷达字段、物理含义与特征不一致**

  现状：多普勒用z坐标近似速度，RCS缺失时用坐标范数；轨迹按距离排序；CSV雷达走词频特征；CSV→PLY丢intensity。

  建议：强制字段/单位/坐标系/时间定义；真实velocity/RCS与几何z分开；无回波数据时不宣称真实距离-多普勒；格式转换保留列。

  验收：CSV/PLY转换往返保留必要字段；radar特征不含tfidf；轨迹有timestamp/track_id且单位明确。

  证据：R13/R14；preprocess.py:label_radar/_extract_feature/_convert_format；file_service.py:_radar_viz

- [ ] **L18 文本编码、换行与结构没有统一**

  现状：元数据可标GBK，但处理常用UTF-8忽略错误；人工读原始CRLF，预览splitlines后以LF重组；实体规则还常无偏移。

  建议：入库保留原文件并建立规范化文本视图、encoding/offset_unit/line-ending映射；JSON/XML/CSV选择字段或记录，不盲扫原始格式。

  验收：GBK/CRLF/emoji与重复实体测试通过；同一span在自动、人工、预览、导出指向同一内容。

  证据：U06；file_service.py:_text_meta/_text_preview；preprocess.py:label_text

- [x] **L19 标签检索漏掉真实类别**

  现状：视频只提取“n个目标”的帧摘要，遗漏嵌套飞机等类别；自动文本classification.category及sentiment不被统一提取。

  建议：按schema递归提取类别ID/名称；分清文件级分类、帧级检测、实体类别、关键词和caption。

  验收：飞机视频按airplane命中；文本分类按飞行日志命中；关键词不伪装为实体类别。

  证据：R02/R16；label_schema.py:label_names；download.py:_labels_of

- [ ] **L20 质量分不代表标签质量**

  现状：accuracy实为前10文件可读性；仅stem匹配导致跨目录覆盖率虚高，错误JSON也算标注；视频没有帧覆盖和人工审核指标。

  建议：改名为内容有效性，按sample_id/相对路径算合法标签率、帧覆盖、审核覆盖；真实准确率需带ground truth的抽检集。

  验收：2张同名不同目录图仅1张有效标签时覆盖50%；error标签不算有效；空框与合法负样本区分。

  证据：R07/R18；quality_service.py:_score_files/_accuracy

- [x] **L21 版本完成状态与当前版本指针混在一起**

  现状：普通任务产物写完前即更新current；失败不回退。init在复制前提交，ready只依赖部分task状态；普通下拉不提示处理失败。

  建议：版本增加building/ready/failed，完成校验后再切换current；失败保留诊断但不成为默认版本；所有输入/交付共享状态校验。

  验收：失败任务后仍能默认进入上个ready版本；复制未完不能被预览评估为完整版本或作为下游输入。

  证据：源码确认；version_service.py:create_initial_version/create_output_version；VersionSelect.vue

- [ ] **L22 普通预处理任务缺少持久执行与恢复**

  现状：自动标注/清洗/转换/采样/工作流是Web中的daemon线程，重启会丢执行，缺恢复、取消和统一资源队列；少样本的tmux队列另有耐久性。

  建议：普通任务迁入持久worker，记录逐文件进度、重试策略、cancel和启动恢复；限制CPU/IO并发，不干扰正式训练。

  验收：重启Web后任务可恢复或明确失败；无永久running；重复重试不产生重复版本/文件。

  证据：源码确认；preprocess.py:_run_in_thread；main.py:recover_fewshot_queue

- [ ] **L23 并发版本号和目录没有独占保护**

  现状：next_version_no为max+1；多个任务可争用同一版本；copy_version会清理计算出的目标目录，缺少目录归属保护。

  建议：锁定数据集行分配版本号；独占staging目录，原子rename，禁止清理非本任务目录。

  验收：同一数据集并发2个任务各得独立版本；冲突不删除已存在/正在写入的产物。

  证据：源码风险，未做生产并发压测；version_service.py:next_version_no/copy_version

- [ ] **L24 工作流只有顺序连接，缺语义预检**

  现状：公开为线性链是合理的，但feature→image标注、未就绪模型、自由填写无效目标格式、负采样比例等缺完整预检；edges存储但不参与执行。

  建议：保存/执行前逐节点验证输入产物、算法和参数；执行前再检版本；明确只支持线性链，非空edges拒绝或后续实现DAG。

  验收：非法链不写出中间误导结果；上游partial不继续；重跑不存在/失败输入时明确拒绝。

  证据：源码确认；tasks.py:create_workflow/run_workflow；preprocess.py:_exec_workflow

- [ ] **L25 集成页面宣称的融合没有执行**

  现状：raw/feature/decision/semantic全部copy_all到子目录，只把策略描述写入manifest；没有拼接、投票或语义对齐。

  建议：第一阶段仅开放“文件集合并”；其余标未实现。后续先定义样本关联键、时间同步、特征schema和权重，再逐项实现。

  验收：选择特征/决策/语义融合时产生真实融合结果与可核对参数，否则提交前拒绝。

  证据：源码确认；preprocess.py:_exec_integrate；IntegrateView.vue

- [x] **L26 图谱供数与标注结构不兼容**

  现状：服务API的graph仅读dict.entities；手动标注数组引发500，mentions边的实体target没有实体节点；四个图谱/本体页面仍占位。

  建议：图谱服务复用统一标签适配；生成实体节点/稳定ID与合法边；页面继续明确规划中，真实关系抽取需另立可验收范围。

  验收：自动NER与手动span都可供图；每条边端点存在；mentions不宣称知识关系推理。

  证据：R19/R20；service.py:_scene_graph；四个ModulePlaceholder页面

- [ ] **L27 AIGC完成数量与需求达标没有区分**

  现状：图片数量完成不等于命中查询标签；COCO80不能给FGVC100机型标签。待人工核验主要是文案，缺审核状态和审核覆盖；零框不应自动当失败或已确认负样本。

  建议：保留允许直接下载的决定；增加逐文件review_status、生成/自动标注/符合查询的不同计数；映射COCO大类与原生分类。

  验收：Boeing机型需求不会因生成airplane框就计为机型补足；审核后文案/指标更新；8张空框可分别确认或退回。

  证据：源码与runtime库存；fewshot_service.py:label_batch；download请求/页面

- [ ] **L28 角色入口与密级元数据边界不统一**

  现状：获操作权限的访客能做预处理/少样本，但普通在线标注首次建副本仍只允许管理员；语义术语带密级却未按view_level过滤。

  建议：明确共享原始素材与个人工作集政策后统一入口权限；术语列表/检查/图谱供数按密级过滤；管理员可查看他人保持。

  验收：同一角色前后端可用入口一致；低密级账号看不到高密级术语；少样本私有来源及派生版本继续隔离。

  证据：源码确认；data.py:annotation-context；security.py；R23；standards.py:list_terms/semantic_check；service.py:_scene_graph

## P2

- [ ] **L29 原始数据状态与仪表盘口径偏粗**

  现状：一次上传单个label即可把整个原始集标为100%已标注；原始元数据由首个文件覆盖，不能代表混合分辨率/总时长；清洗统计还混入sidecar计数。

  建议：按文件聚合标注/审核状态；分辨率分布、总时长和数据文件计数独立；原始和版本指标明确命名。

  验收：100文件仅标1个显示1%；多尺寸图不显示为单一尺寸；清洗前后文件数同口径。

  证据：源码确认；data.py:upload_label/_refresh_aggregates；preprocess.py:_exec_clean；dashboard

- [ ] **L30 导出没有任务专属schema和样本清单**

  现状：主要下载平台JSON+ZIP，无明确图像COCO/YOLO、视频逐帧/轨迹、语音转写或实体任务导出适配；原始打包血缘不含选定标签版本/hash。

  建议：先规范平台内部schema，再按确认的训练消费端提供导出；每包加sample_id、split、schema、类别映射、来源hash和review状态。

  验收：下游加载器不猜坐标或时间单位；数据、标签、split能一一对应；native格式的版本/语义说明随包。

  证据：源码确认；download.py:_zip_files/download_pack；datasets.py:_zip_version；service.py:_zip_supply

- [ ] **L31 标准、语义、外部回写仍是独立工具**

  现状：自定义规则确有计算，但无规则的标准“合规”只是代理分数；术语替换返回字符串而非新版本；外部回写保存附件但未注册可处理数据集/版本。

  建议：将标准指标对照与人工合规审查区分；术语变换形成带offset映射版本；回写按结果类型验证并选择导入到对应版本。

  验收：无法把通用分数当标准原文已完整执行；术语变换可追溯且不破坏span；回写结果能按明确类型进入后续处理。

  证据：源码确认；quality_service.py:evaluate_quality；standards.py:semantic_replace/create_writeback

- [ ] **L32 大数据规模和资源使用缺少约束**

  现状：ZIP先整体BytesIO缓冲，普通下载20GB上限仍可耗尽内存；全量标签扫描/版本列举重复IO，前100工作集限制与长文件读取仍存在。

  建议：打包改流式或数据盘临时包；文件/标签索引，任务和前端分页；限制并发与单次处理预算，超限显式提示。

  验收：大包下载内存不随包体积线性增长；大目录页面不全盘扫描；任务显示预算与实际覆盖。

  证据：源码确认；download.py:_zip_files；datasets.py:_zip_version；composables.js；pp.list_files


## 2026-10-13 修复范围与剩余工作（已部署）

本轮以纠错和收起无效功能为主，不增加跟踪引擎、复杂融合或新的任务系统。原清单勾选暂保留未完成，以免把“已阻止错误路径”混写为全部建议实现。

| 项目 | 本地实现 | 尚未完成的原建议 |
|---|---|---|
| L01、L03、L06 | 来源隔离、复制回滚、清洗标签映射、失败/空产物不可交付 | 生产健康与构建检查已通过 |
| L02 | 图像框随缩放；未知几何格式拒绝；数值坐标标准化暂关闭 | polygon/mask通用映射 |
| L04、L05 | 五模态适配、元数据保留、上下文缓存清理 | 现有格式的缓存及四模态页面验收通过；新格式适配另行处理 |
| L07 | 内容一致且成功版本才可补标签 | 固定下载快照和显式原始/处理版本选择的全面验收 |
| L08、L09 | 空帧、截断信息、帧率步进、当前帧命中过滤 | VFR/PTS精确定位、轨迹、全片标注与传播 |
| L10、L12、L15、L25 | 暂不开放不真实的算法、采样、融合和特征版本入口 | 真正的分层/时间/K折、特征产物、融合引擎 |
| L13、L14、L24 | 能力白名单、主要参数及依赖检查、工作流共享校验 | 各模型真实加载验收与更细格式约束 |
| L16、L18 | Whisper路径、转写文本编辑、GBK/CRLF及UTF16适配 | ASR实际模型、结构化文本字段选择、历史偏移迁移 |
| L17 | CSV/PLY强度列保留；误导规则收起、图表改名 | 雷达字段单位/真实物理信号处理 |
| L19、L20、L26 | 类别递归检索、有效配套率、人工实体图谱 | 真值准确率、真实关系抽取 |
| L21、L22、L23 | 成功才切默认版本、重启中断失败、版本行锁和目录独占 | 耐久worker、恢复/取消、生产MySQL并发压力测试 |
| L28 | 术语密级过滤 | 首次个人工作集创建权限政策保持原状，未擅自扩权 |
| L32 | 三条ZIP路径改用磁盘临时文件 | 大目录索引/分页、CPU/IO预算与大包实测 |

L11、L27、L29、L30、L31仍保留后续需求：来源分组防泄漏、审核与查询达标统计、原始聚合口径、专用导出、标准/回写闭环。训练、生成和现有用户隔离行为保留。验证记录见 `CHANGELOG_LINUX.md` 与逻辑修复实验目录。

### 最终验收补充

2026-10-13：L01/L03/L04/L05/L06/L19/L21/L26按现有格式与页面闭环完成；未知标注格式会拒绝覆盖。25项浏览器检查、9项适配检查、两端24项逻辑和21项接口检查通过。L32的前100数据集选择限制已修复，版本请求限8路并发。其余未勾选项继续以表中剩余范围为准。

### 2026-10-13 视频工作台补充

已增加实际原视频播放、工作台SAM2.1框/mask跟踪、目标ID、mask画笔修正及版本下载。视频相关项仅关闭本轮实际验收范围；关键帧插值、VFR兼容、完整审核状态和MOT等专用导出仍未实现，不能由本次模型跟踪推断全部视频待办已完成。
