# 视频 Context 加噪：训练、推理与 Demo

新增：[首步 clean → 重建／自由演化 · 全程条件角色：代码、配置及10组验证](scripts_mytrain/inference/single_case/CONDITION_ROLE.md)。

本仓库整理 **Wan、Cosmos3-Nano、Cosmos-Predict2.5** 的训练代码来源、历史 clean / noisy 推理，以及 context 与 future 同等级加噪的推理对照。当前提供 `v2v_gap_038` 样本的 **9 个历史推理视频及10个首步 clean 条件角色对照视频**，并附对应代码、配置和验证记录。

同一 backbone 的三个推理版本共用同一份 **N75 训练权重**。`clean` 指推理时使用干净 context，不代表使用了 clean 训练权重；`equal_sigma` 是推理对照，没有重新训练。

## 快速导航

| 内容 | 入口 |
|---|---|
| 单视频 + prompt 推理入口及9组验证 | [single_case/README.md](scripts_mytrain/inference/single_case/README.md) |
| 训练入口、源码版本证据与启动命令 | [training/README.md](training/README.md) |
| 六个历史推理入口、依赖与复现命令 | [historical/README.md](scripts_mytrain/inference/historical/README.md) |
| Context / future 同等级加噪实验 | [equal_sigma/README.md](scripts_mytrain/inference/equal_sigma/README.md) |
| 9 个视频与来源说明 | [demos/gap038/README.md](demos/gap038/README.md) |
| 支持联动播放的对比页 | [comparison.html](demos/gap038/comparison.html)，需通过下方 HTTP 服务打开 |

## Backbone 与训练方案

Wan（PhysRVG DiT 起点）、Cosmos3-Nano（MoT）和 Cosmos-Predict2.5（DiT）是三个 backbone。**A/C/N75 表示训练方案，clean/noisy/equal_sigma 表示推理设置**：同一 N75 权重可以进行 clean 推理。

| 训练组 | Context 输入噪声等级 | Future 监督 |
|---|---|---|
| C | 干净，sigma_c=0 | Future-only Flow Matching MSE |
| A | 与 future 相同，sigma_c=sigma_f | 同上 |
| N75 | sigma_c=Q(0.75*sigma_f)，Q 为训练网格最近邻 | 同上 |

这些变体还涉及时间嵌入和条件标记：Wan 使用逐 token timestep；Nano C 的 context 不加去噪时间嵌入、A/N75 作为 noisy token；Predict2.5 C 的 context mask=1、A/N75 mask=0。它们不是单独改变噪声强度的消融。

完整公式、三模型对照、loss、训练代码和证据范围见 [训练方法 README](training/README.md)。项目记录中 Wan/Nano 有 A/C/N75，Predict2.5 有 C/N75，A 仅代码支持；仓库训练证据主要归档 N75，不包含所有训练组的完整资产或权重。

## 视频对比

| Backbone | 权重 | Clean | 原 noisy | 同等级 noisy |
|---|---|---|---|---|
| Wan | N75 step1000 | [视频](demos/gap038/wan_clean.mp4) | [视频](demos/gap038/wan_noisy.mp4) | [视频](demos/gap038/wan_equal_sigma.mp4) |
| Cosmos3-Nano | N75 step1500 | [视频](demos/gap038/nano_clean.mp4) | [视频](demos/gap038/nano_noisy.mp4) | [视频](demos/gap038/nano_equal_sigma.mp4) |
| Cosmos-Predict2.5 | N75 step1500 | [视频](demos/gap038/predict25_clean.mp4) | [视频](demos/gap038/predict25_noisy.mp4) | [视频](demos/gap038/predict25_equal_sigma.mp4) |

视频保留实际推理输出的原始 MP4 字节，未重新编码。每行内的对照保持相同输入、权重和随机种子。Wan 使用旧 Test70，Nano 使用 Test70_v2，Predict2.5 使用旧 Test70 triplet，**跨行输入并不完全相同**，应优先比较同一行。

### 本地可视化

在仓库根目录以前台方式启动：

```bash
python3 -m http.server 8845 --bind 0.0.0.0 --directory demos/gap038
```

打开 [http://localhost:8845/comparison.html](http://localhost:8845/comparison.html)。页面支持每行联动播放、全部播放/暂停、调速和查看逐步 sigma / timestep。

原实验服务器的服务地址为 [http://10.176.42.45:8845/comparison.html](http://10.176.42.45:8845/comparison.html)，需要网络可达且服务仍在运行。该服务器的前台重启命令是：

```bash
python3 -m http.server 8845 --bind 0.0.0.0 --directory /data/gaoya/agent-data/outputs/github-context-inference-20261010-efc_zrlo/demos/gap038
```

## 单 case 推理

新增三个直接处理单个视频的入口：[wan.py](scripts_mytrain/inference/single_case/wan.py)、[nano.py](scripts_mytrain/inference/single_case/nano.py)、[predict25.py](scripts_mytrain/inference/single_case/predict25.py)。每个入口支持 `--video`、`--prompt`、`--adapter`、`--mode clean|noisy|equal_sigma`、`--output`，不要求历史 `result.json` 或批量case清单。默认gap038配置及依赖路径位于 `single_case/presets/`。

```bash
/data/gaoya/agent-data/envs/cosmos_nano_acn75/bin/python scripts_mytrain/inference/single_case/nano.py \
  --mode noisy --gpu 3 --output /data/gaoya/agent-data/outputs/single_case_demo/nano
```

本次9组全部运行：Wan/Nano六组对已有demo逐字节通过；Predict2.5三组仍有像素差异，不能宣称全部严格复现。Predict2.5 clean本次输出与最初历史视频一致，但与已上传复跑demo不同；报告区分这两套参考。

原模型工程、权重和部分历史输入缓存仍是外部依赖。新视频/新prompt自动去掉默认旧case缓存。生成后可用 `--verify-against` 比较参考MP4，非逐字节一致返回退出码2；[verify_matrix.py](scripts_mytrain/inference/single_case/verify_matrix.py) 可顺序跑完9组。详细参数和运行命令见 [单case说明](scripts_mytrain/inference/single_case/README.md)，实测结果见 [verification.json](scripts_mytrain/inference/single_case/verification.json)。

## 推理代码与加噪方式

| Backbone | Clean 入口 | Noisy 入口 | 本次 noisy demo 实际核心 |
|---|---|---|---|
| Wan | [wan_clean.py](scripts_mytrain/inference/historical/wan_clean.py) | [wan_noisy.py](scripts_mytrain/inference/historical/wan_noisy.py) | `ContextNoise`、`sample` |
| Cosmos3-Nano | [nano_clean.py](scripts_mytrain/inference/historical/nano_clean.py) | [nano_noisy.py](scripts_mytrain/inference/historical/nano_noisy.py) | [frozen/nano/v2/hooks.py](scripts_mytrain/inference/historical/frozen/nano/v2/hooks.py) 中的 `NanoContext` |
| Cosmos-Predict2.5 | [predict25_clean.py](scripts_mytrain/inference/historical/predict25_clean.py) | [predict25_noisy.py](scripts_mytrain/inference/historical/predict25_noisy.py) | `install_predict` 中的 `denoise` |

入口由 [reproduce.py](scripts_mytrain/inference/historical/reproduce.py) 根据历史记录恢复配置并选择冻结实现，不能仅复制一个入口文件运行。

对原 noisy demo，每次模型调用前从干净 context latent `c` 重建输入：

```text
sigma_c = nearest_training_sigma(0.75 * sigma_f)
context_input = (1 - sigma_c) * c + sigma_c * epsilon_c
```

`epsilon_c` 在整段推理中固定，变化的是 `sigma_c`；future 沿原生 scheduler 迭代去噪。Wan 从完整初始噪声取 context 切片；Nano 和 Predict2.5 使用独立 FP32 context 噪声，seed 为 20261007。

同等级实验使用独立入口 [equal_sigma/run.py](scripts_mytrain/inference/equal_sigma/run.py)：

```text
sigma_c = sigma_f
timestep_c = timestep_f
context_input = (1 - sigma_c) * c + sigma_c * epsilon_c
```

此处直接使用当前 future sigma / timestep，不做训练网格量化；context 噪声来源、条件 mask、CFG 和原生采样器保持原 noisy 实现。它不是每步重新抽噪声，也不包含 RePaint、回跳或提前 n 步变体。

| Backbone | 本次采样记录 | 原 noisy 的 context 设置 |
|---|---|---|
| Wan | 40 步，seed42，CFG 关闭 | 保留原 mask，替换输入和 token timestep；scheduler 只更新 future，再拼回 clean context |
| Nano | 40 步，seed42，CFG=6 | context 改为 noisy token role；scheduler 更新后恢复 clean context |
| Predict2.5 | 配置35步，实际36个 timestep / 72次 CFG 调用，seed0，CFG=7 | 条件输入 mask 置零；直接调用 net，再用解析 context 速度保留原生轨迹 |

同等级实验逐步记录：[Wan](demos/gap038/wan_equal_sigma_trace.json) · [Nano](demos/gap038/nano_equal_sigma_trace.json) · [Predict2.5](demos/gap038/predict25_equal_sigma_trace.json)。记录中的 `step` 从0开始，描述模型调用时的等级，不包含最后一次 scheduler 更新后的终点。

## 训练代码来源

| Backbone | 入口与训练核心 | 来源核查 |
|---|---|---|
| Wan | [RL-LoRA raw 入口](training/wan/workspace/scripts_mytrain/train/train_full_sa_pybullet_rl_lora_raw.py) → [训练核心](training/wan/workspace/scripts_mytrain/train/train_full_sa_pybullet_aux.py) | 训练记录确认入口和 argv；未找到训练当日完整源码快照，所附源码为对应工作区当前版本 |
| Nano | [train.py](training/nano/source/train.py)、[common.py](training/nano/source/common.py) | 训练运行目录快照，13个文件全部与原始哈希记录一致 |
| Predict2.5 | [官方入口](training/predict25/native/scripts/train.py)、[N75 配置](training/predict25/projects/generic_acn75/n75_mb4_config.py)、[model.py](training/predict25/projects/generic_acn75/model.py) | 自定义代码来自运行目录快照；所附原生核心与训练记录的 Git commit 一致 |

完整配置、启动命令和版本证据见 [训练说明](training/README.md) 与 [训练源文件清单](training/source_manifest.json)。此次整理没有重跑训练；源码归档不等于从头训练复现已通过。

## 运行条件与验证范围

仓库包含代码、配置、验证记录和 demo 视频，**不包含权重、数据集、训练缓存或完整 backbone 工程**。配置保留原实验绝对路径；运行推理/训练需要原模型工程、对应 Python 环境和外部资产。查看视频只需要 Python HTTP 服务和浏览器。

在原始 PhysRVG 工作区可先做历史入口预检查，不占 GPU、不生成输出：

```bash
python3 scripts_mytrain/inference/historical/nano_noisy.py --check-only
```

推理复跑必须指定全新或空的 `/data/gaoya/...` 输出目录。三个环境的具体命令见 [历史推理说明](scripts_mytrain/inference/historical/README.md) 和 [同等级实验说明](scripts_mytrain/inference/equal_sigma/README.md)。

已完成的验证：

- **历史 Wan / Nano**：默认 clean / noisy 各一例，共4个视频与历史视频逐字节一致。
- **历史 Predict2.5**：输入与完整 context / timestep 审计一致，但 clean / noisy 视频仍有像素差异；原始未拆分脚本对照也有差异，具体原因尚未确定，不能标为严格复现成功。
- **同等级实验**：三个 backbone 的每次调用均通过 sigma / timestep 严格相等检查，输入与初始噪声配对检查通过；这证明对照设置按代码执行，不代表生成质量一定更好。
- **视频发布**：9个视频通过解码与本地 HTTP 文件校验；上传视频哈希与本地生成文件一致。

证据：[历史复现报告](scripts_mytrain/inference/historical/verification.json) · [原6个视频清单](demos/gap038/manifest.json) · [同等级实验报告](demos/gap038/equal_sigma_manifest.json)。

## 许可

保留各源文件既有版权和许可声明；上游模型、工程与依赖遵循各自许可。Predict2.5 原生代码许可见 [LICENSE](training/predict25/native/LICENSE)。

[新增：首步 clean · 全程条件角色的10个视频对照](demos/condition-role-clean-first/README.md)。
