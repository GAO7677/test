# 视频标注工具调研（2026-10-13）

目标：视频目标框标注，沿用 COCO 80 类；接入现有数据中台，保留用户隔离与版本回流，界面保持简洁。本文是选型调查，未安装或接入下列工具。

## 1. 当前问题与能力边界

- 调研时 `PreviewPane.vue` 视频分支只显示首帧缩略图，解释了“视频预览还是图像”。本轮已改为视频播放，13项验收通过并同步至118；预览修复与标注工具选型分别处理。
- `ManualLabelView.vue` 是自建 Vue + Canvas 工具：已有视频播放、前后帧定位，在选中帧画框；画框逻辑沿用图像标注。没有完整轨迹 ID、关键帧插值、遮挡/出画管理或 AI 目标跟踪。
- 自动 COCO 视频标注是抽帧独立检测。某帧出现同类框，不代表已关联为同一目标；未检测帧不能视为已标注。

应区分三件事：**播放预览**展示原视频；**关键帧插值**根据人工框计算中间帧框；**AI 跟踪**根据图像内容传播目标位置。插值与跟踪仍需人工检查，均不是“只标首帧就自动保证全片正确”。[CVAT 轨迹定义](https://docs.cvat.ai/docs/getting_started/vocabulary/)

## 2. 最新稳定版本

核验方法：访问官方 GitHub `releases/latest` 的实际重定向，读取发布页 `datetime`；以下日期为 UTC。开发分支版本、SDK 版本和 SaaS 工具界面代号不混用。

| 工具 | 本次核实的稳定版本 | 发布日期（UTC） | 官方证据 |
| --- | --- | --- | --- |
| CVAT Community | **v2.78.0** | 2026-10-06 18:19:32 | [release](https://github.com/cvat-ai/cvat/releases/tag/v2.78.0) |
| Label Studio Community | **1.23.2** | 2026-09-29 14:47:15 | [release](https://github.com/HumanSignal/label-studio/releases/tag/1.23.2) |
| X-AnyLabeling | **v4.1.1** | 2026-10-12 17:29:20 | [release](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v4.1.1) |
| Supervisely | 官方当前文档为 **Video Annotation Tool 3.0**；未确认整个平台最新稳定发布编号/日期 | 未确认 | [视频工具文档](https://docs.supervisely.com/labeling/labeling-toolbox/videos-3.0) |

Supervisely 的“3.0”是视频工具名称，不能表述成平台最新版本。部署日期若晚于本次调研，应再查稳定 release，并固定版本而不是跟随 `develop` 或浮动 `latest`。

## 3. 功能与接入成本

| 工具 | 视频框与跨帧对象 | 关键帧与 AI | 部署、权限与适配判断 |
| --- | --- | --- | --- |
| **CVAT** | 原生 Track 模式；同一对象跨帧组成轨迹，可合并轨迹并标记出画 | 人工关键帧之间插值；AI 跟踪另需部署模型 | 浏览器、自托管；有个人空间、角色、API。最贴近本平台的视频框任务。 |
| **Label Studio** | `VideoRectangle` 输出对象 ID 和关键帧序列 | 支持插值，导出默认仅关键帧，可显式导出插值帧；AI 需另接 ML backend | 浏览器、自托管；多模态强，但 Community 所有用户可见所有项目，不可直接沿用一个共享实例满足本平台隔离。 |
| **X-AnyLabeling** | 提供多目标跟踪，模型输出 `track_id` 写入 `group_id` | 官方支持 ByteTrack、BoT-SORT、TrackTrack、SAM2/3-Video；未核实 CVAT 式人工关键帧插值工作台，不作承诺 | Qt 桌面工具；适合个人离线标注，接入现有多用户 Web 页需额外架构，优先级较低。 |
| **Supervisely** | 视频对象 ID、时间线、轨迹视图 | 插值、自动跟踪、修改后重新跟踪，支持 MixFormer 等应用 | 浏览器；有 API/SDK；本地私有化属 Enterprise，功能齐全但当前需求下成本和范围较大。 |

能力证据：[CVAT Track 模式](https://docs.cvat.ai/docs/annotation/manual-annotation/modes/track-mode-basics/)、[Label Studio 视频模板与输出格式](https://labelstud.io/templates/video_object_detector)、[X-AnyLabeling v4.1.1 README](https://github.com/CVHub520/X-AnyLabeling/blob/v4.1.1/README.md)、[X-AnyLabeling 固定版本跟踪代码](https://github.com/CVHub520/X-AnyLabeling/blob/v4.1.1/anylabeling/services/auto_labeling/__base__/yolo.py)、[Supervisely 视频工具](https://docs.supervisely.com/labeling/labeling-toolbox/videos-3.0)。

### 开源与付费边界

- **CVAT Community：MIT。** 关键帧插值属于基础轨迹标注。Community 的模型集成方式为 Nuclio，公开示例包含 TransT；官方额外 SAM 2 Tracker 和 native AI Agents 的可用范围包括 Enterprise/Online，不能宣传为 CE 装好即有的免费能力。[许可证](https://github.com/cvat-ai/cvat/blob/v2.78.0/LICENSE)、[官方 AI 能力矩阵](https://docs.cvat.ai/docs/annotation/auto-annotation/ai-models/)
- **Label Studio Community：Apache-2.0。** 官方明确 CE 用户能看见全部项目；细粒度项目权限属于付费版本。即使平台页面隐藏入口，也无法防止用户直接访问标注服务的其他项目，必须另做有效隔离或选择支持权限的版本。[许可证](https://github.com/HumanSignal/label-studio/blob/1.23.2/LICENSE)、[CE 用户说明](https://labelstud.io/guide/signup)、[版本比较](https://labelstud.io/guide/label_studio_compare)
- **X-AnyLabeling：GPL-3.0。** 主体是桌面应用；其推理 Server 不等同于完整多用户 Web 标注站点。[固定版本说明](https://github.com/CVHub520/X-AnyLabeling/blob/v4.1.1/README.md)、[Server 使用方式](https://github.com/CVHub520/X-AnyLabeling-Server/blob/main/docs/source/en/get_started.md)
- **Supervisely：免费 Community 不等于可免费私有化的整套开源平台。** 官方定价页将 self-hosted、离线环境和 SSO 列在 Enterprise，API/SDK 在各方案可用。[官方方案](https://supervisely.com/pricing/)

## 4. 建议的最小方案（尚未实施）

建议选 **CVAT Community v2.78.0** 承担视频标注；平台保留数据管理与下载，标注工作台使用 CVAT 原生页面。依据是轨迹能力、浏览器访问、自托管和角色管理均贴合当前需求，避免自行重写视频编辑器。[Community](https://docs.cvat.ai/docs/products/community/)、[权限](https://docs.cvat.ai/docs/account_management/user-roles/)、[API](https://docs.cvat.ai/docs/api_sdk/api/)

1. **先修预览。** 视频显示带播放/暂停/拖动的播放器；首帧仅作封面。媒体访问沿用用户和数据权限，浏览器不支持编码时明确提示，不默认批量转码。
2. **入口只保留必要动作。** 当前视频/版本 → “视频标注” → CVAT 对应任务；保存后“同步标注” → 平台新版本 → 预览/下载。首期可用新标签页，避免先做复杂嵌入。
3. **首期人工关键帧 + 插值即可。** 开放 COCO 80 类框、轨迹 ID、关键帧、遮挡/出画；人工检查中间帧。原有抽帧检测结果只能作为候选框，不能自动冒充已建立轨迹。
4. **身份按用户映射。** 平台用户映射独立 CVAT 用户，任务由对应用户持有或明确分配；管理员单独授权查看全部。平台导入/同步接口再次验证用户、原始文件和版本，不能给所有浏览器共用管理员令牌。CVAT 个人空间将资源关联到个人账户；角色策略仍需用两个普通用户及管理员实际验收。[个人空间](https://docs.cvat.ai/docs/account_management/organization/)、[角色](https://docs.cvat.ai/docs/account_management/user-roles/)
5. **结果保留轨迹语义。** 原始 CVAT 结果和平台映射均保留 `track_id`、帧号/时间、关键帧、出画/遮挡；对检测训练再导出逐帧框。下载明确“人工关键帧/插值结果、待核验”，不丢掉轨迹后仍称视频完整标注。
6. **AI 跟踪后置。** 真有省人工需求时再单独评估模型、GPU 7 显存与标注质量；人工关键帧插值阶段不需要推理模型，也不应为了“最新”先加入多个模型服务。

最低验收：可播放并定位原视频；同一对象跨帧 ID 一致；关键帧修改影响插值帧；出画不继续留框；保存、重开、同步、下载一致；用户 A 无法通过改 URL/API 读取用户 B 的任务或媒体；管理员按授权可查看；长视频/可变帧率验证帧号与时间对应。Label Studio 官方也提示变帧率可能产生定位差异，说明此项不能只靠浏览器播放验收。[帧率说明](https://labelstud.io/templates/video_object_detector)

## 5. 已确认与仍待验证

已确认：三个开源项目最新 release 及 UTC 日期；官方说明的视频轨迹/插值能力；CE 权限和主要付费边界。GitHub API 出口遇到限流，版本改由官方 release 网页及其完整时间属性核实。

未执行：安装、模型推理、实际长视频性能、接口回写和跨用户验收；未确认 Supervisely 整个平台最新版本，也未确认 X-AnyLabeling 的人工关键帧插值界面。CVAT 最小接入会新增独立服务和身份/结果映射，需要用户确认选型后再实施。

## 2026-10-13 后续决策

用户明确取消CVAT部署，要求视频工作台内选择跟踪和mask，并沿用平台统一管理。现已按该方向实现、验收；上面的CVAT选型保留为历史调研，不再作为当前执行方案。当前方案见`integrations/video_annotation/README.md`，发布状态见`CHANGELOG_LINUX.md`。
