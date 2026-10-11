# Backbone、A/C/N75 训练方法与代码

本文先统一说明三个 backbone 的 A/C/N75 训练机制；下方来源证据和训练命令主要对应已上传的 gap038 demo 所用 **N75 权重**。同一 backbone 的 clean、原 noisy、equal_sigma 三个推理版本共用同一份训练权重；推理 clean 不表示它来自 C/clean 训练，equal_sigma 也没有重新训练。

| Backbone | Demo checkpoint | 实际训练入口 | 训练核心 | 来源证据 |
|---|---|---|---|---|
| Wan | N75 step1000 | [train_full_sa_pybullet_rl_lora_raw.py](wan/workspace/scripts_mytrain/train/train_full_sa_pybullet_rl_lora_raw.py) | [train_full_sa_pybullet_aux.py](wan/workspace/scripts_mytrain/train/train_full_sa_pybullet_aux.py) | [训练启动记录](wan/records/launch.json)、[resolved config](wan/records/resolved_config.json) |
| Cosmos3-Nano | N75 step1500 | [train.py](nano/source/train.py) | [common.py](nano/source/common.py)：context_pair / noise_input / future_loss | 运行目录 `source/` 快照；13 个文件全部与[当时的哈希清单](nano/source_hashes.json)一致 |
| Cosmos-Predict2.5 | N75 step1500 | [官方 scripts/train.py](predict25/native/scripts/train.py) + [n75_mb4_config.py](predict25/projects/generic_acn75/n75_mb4_config.py) | [model.py](predict25/projects/generic_acn75/model.py)、[mechanism.py](predict25/projects/generic_acn75/mechanism.py) | 运行目录 `code_snapshot/`；[启动记录](predict25/records/launch.json)及[checkpoint metadata](predict25/records/step001500_metadata.json) |

## 三个 backbone 与 A / C / N75 的含义

**Backbone 是模型结构；A、C、N75 是训练方案；clean、noisy、equal_sigma 是推理设置。三者不能混为一谈。** 例如，N75 权重可以使用 clean 推理；“首步 clean”也不会把 A 权重变成 C 权重。

| Backbone | 本项目模型与训练起点 | Context 表示 |
|---|---|---|
| Wan2.2-TI2V-5B / PhysRVG | Wan 架构的 DiT，本轮 LoRA 从 PhysRVG DiT 权重开始，不是直接从原生 Wan DiT 开始 | 与 future 合并成视频 token；无独立条件 mask 输入通道 |
| Cosmos3-Nano | MoT：Reasoner + Generator，基于 Nano 模型进行 LoRA | 本项目视频 context 进入 Generator 分支；通过条件／noisy 索引控制时间嵌入 |
| Cosmos-Predict2.5 | 基于官方 post-trained Video2World 模型的 DiT，使用官方训练循环加自定义 RF forward | 与 future 合并；额外条件 mask 作为网络输入通道 |

以下描述本项目自定义训练，不等同于各官方工程的全部训练配置。

### 加噪公式与时间条件

令干净 latent 为 `z = concat(c0, f0)`，随机高斯噪声为 `epsilon = concat(epsilon_c, epsilon_f)`。在本实验49帧配置中，前8帧提供2个 context latent，后11个 latent 为 future。以下使用标准化 RF sigma 表示噪声等级：

```text
future_input  = (1 - sigma_f) * f0 + sigma_f * epsilon_f
context_input = (1 - sigma_c) * c0 + sigma_c * epsilon_c

C:   sigma_c = 0
A:   sigma_c = sigma_f
N75: sigma_c = Q(0.75 * sigma_f)
```

`Q(s)` 是在该训练配置的配对 sigma 网格中选择与 s 最近的等级。N75 的实际 sigma 因此通常不严格等于 `0.75*sigma_f`。C 和 A 走精确端点分支，不需要近邻量化。这里的“同噪声等级”指 sigma 相同，不表示 context 与 future 使用逐元素相同的一份噪声；代码从整段随机噪声张量取各自的时间切片。训练每个 batch 重新采样，不能套用推理中整条轨迹固定 context 噪声的描述。

| Backbone | C：干净 context | A：同等级加噪 | N75：部分加噪 |
|---|---|---|---|
| Wan | context timestep=0，使用 E(0) | `match_noise` 下 timestep 与 future 相同 | 使用 scheduler 中与 sigma_c 配对的 timestep |
| Nano | `condition_frame_indexes=[0,1]`；context 不添加去噪时间嵌入 | 条件索引为空，context 纳入 noisy 索引，添加 E(t_f) | 同 A 的 noisy 角色，context 添加 E(t_c) |
| Predict2.5 | context timestep=0，context mask=1、future mask=0 | context/future timestep 相同，mask 全0 | context timestep=`sigma_c * num_train_timesteps`，mask 全0 |

Nano 的时间参数同样由标准化 sigma 乘训练时间尺度生成，再按 noisy token 位置打包。Wan 使用 scheduler 配对的时间标签，不应跨模型直接比较 timestep 数值。**E(0) 不等于不添加时间嵌入。** Wan 时间通过 AdaLN 调制，Nano 在 noisy token 输入特征上加时间嵌入，Predict2.5 使用逐帧时间调制和独立 mask 通道。

Wan 代码另支持 `force_zero` 诊断分支，不能将其混入上表 `match_noise` 的定义；历史权重以各次运行配置为准。Predict2.5 自定义训练直接调用 `self.net`，避免原生 `denoise()` 将加噪 context 覆盖回干净条件。其训练 C 的 timestep=0 也不意味着原生 clean 推理必须等于0：原生推理受 `conditional_frame_timestep` 控制，已有推理记录使用0.1。

### Loss 与可训练参数

三个 backbone 的上述自定义训练采用 future-only Flow Matching MSE：

```text
target_f = epsilon_f - f0
loss = mean((predicted_velocity_f - target_f)^2)
```

只对后11个 future latent 的元素计算均值，context 不直接计入 loss。Nano C 的输出位置已排除 context；A/N75 则从完整预测中切除 context 后计算损失。Wan 和 Predict2.5 也限制在 future 范围。Context 仍通过联合注意力影响 future，future loss 可以反传到 context 特征和共享参数。

已归档的 N75 训练配置均使用 LoRA rank32 / alpha64、effective batch8；具体注入模块和 batch 构成按 backbone 配置查看，不能仅凭相同 rank 认为可训练参数量相同。Nano 默认只训练 Generator 注意力和 FFN 的 LoRA，原参数冻结；[Nano 配置说明](nano/source/README.md) 列出了目标模块。其他模型的实际参数与配置见下方训练来源及启动记录。这些超参数证据对应已归档 N75 运行，不应无条件推广到所有历史 A/C checkpoint。

**解释边界：** C→A 不只改变噪声，还改变时间条件；Nano 和 Predict2.5 也改变条件标记。A→N75 保持同一角色设计，主要改变 context sigma 及其匹配 timestep。因此不能把 C/A/N75 的结果差异全部归因于噪声强度，不能用不同 backbone 的训练 loss 数值直接比较质量。

### 实际权重、代码支持与仓库归档范围

| Backbone | 项目已记录训练组 | 本仓库主要训练证据 | 本次首步 clean 视频对照 |
|---|---|---|---|
| Wan | A、C、N75 | N75 step1000 demo 的训练记录；源码为当前工作区版本 | A/N75 step1000 |
| Nano | A、C、N75 | N75 step1500 训练源码快照、配置及哈希 | A/N75 step1500 |
| Predict2.5 | C、N75；A 仅代码支持，未作为已完成正式训练列出 | N75 step1500 自定义源码快照和原生 commit 证据 | N75 step1500 |

“项目有该组训练记录”不表示本仓库已经归档每个组的完整训练命令、配置和权重。当前上传的训练证据主要围绕 N75 demo；没有上传模型权重，也没有执行从头重训来证明训练复现。A 的推理配置和生成视频可以查看 [首步 clean 两组对照](../demos/condition-role-clean-first/README.md)。

### 代码对应位置

| Backbone | 噪声、角色和 loss 实现 | 配置／来源 |
|---|---|---|
| Wan | [train_full_sa_pybullet_aux.py](wan/workspace/scripts_mytrain/train/train_full_sa_pybullet_aux.py)：`_context_noise_pair`、`_flow_forward` | [resolved_config.json](wan/records/resolved_config.json)；注意 `context_timestep_mode` |
| Nano | [common.py](nano/source/common.py)：`context_pair`、`noise_input`、`pack`、`future_loss`；[train.py](nano/source/train.py) | [experiment.json](nano/source/experiment.json)、[N75.json](nano/configs/N75.json)；入口 `--mode A/C/N75` |
| Predict2.5 | [mechanism.py](predict25/projects/generic_acn75/mechanism.py)：`pack_context`；[model.py](predict25/projects/generic_acn75/model.py)：`forward` | [n75_mb4_config.py](predict25/projects/generic_acn75/n75_mb4_config.py)、[config.py](predict25/projects/generic_acn75/config.py) |

### 与最近推理实验的关系

[首步 clean → 重建／自由演化 · 全程条件角色](../scripts_mytrain/inference/single_case/CONDITION_ROLE.md) 是固定训练权重之后的推理消融，没有重新训练。它强制保持条件角色及配套时间，而 A/N75 训练时的 context 设置见上表；因此不能称为与 A/N75 训练完全一致的推理。`clean_rebuild` 每步重建输入，`clean_free` 沿 scheduler 演化输入，两者共享这项条件角色干预。

## 版本证据与限制

- **Wan**：W&B 训练记录确认入口路径和实际 argv，resolved config 确认 context_noise_scale=0.75、match_noise、future-only flow loss、effective batch8。没有找到训练当日的完整源码快照或整个入口的当时哈希；上传的是该入口及依赖的**当前工作区版本**，不能宣称与 2026-09-24 训练时逐字节一致。配置里保留了当时启用和未启用的其他模式参数，但该权重使用 raw loss / RL LoRA 分支。
- **Nano**：上传训练运行目录中冻结的源码，所有 13 个文件逐一核对原始 `source_hashes.json`。使用 [step1500 complete.json](nano/records/step-001500_complete.json) 的配置，避免误用训练结束后被 resume 覆盖的顶层 resolved config。
- **Predict2.5**：自定义训练工程来自运行目录源码快照，保留官方训练循环。训练记录的原生仓库 commit 为 `a2c298b0a3df3778b973fe65e9e58877b292d8a7`；上传的 6 个原生文件均与该 commit 的 Git blob 一致。此检查不证明训练时不存在未记录的工作区改动，详见 [native_checkout.json](predict25/records/native_checkout.json)。

所有复制文件的原始位置、SHA-256 和证据类别见 [source_manifest.json](source_manifest.json)。这次只整理和上传源码，没有重跑训练，也没有新增训练复现成功的声明。

## 训练命令与目录恢复

这些文件保留原始导入布局、配置和绝对路径，依赖原始模型工程、数据、缓存和 Python 环境；不是可直接在空仓库运行的完整训练发行包。权重、数据和缓存未上传。复训时应使用新输出目录。

### Wan

将 `wan/workspace/` 按相对路径放入对应 PhysRVG 工程。`frozen_motion_probe.py` 是顶层导入所需的小依赖，复制到入口本身已有的 fallback 目录，原源码未改写。FastVideo 模型、LoRA 配置和数据加载代码也一并附上。

[recorded_command.txt](wan/recorded_command.txt) 的入口 argv 来自当时 W&B 记录；`torch.distributed.run --standalone --nproc_per_node=2` 部分按记录的 world_size=2 重建，并非保存下来的完整原始 shell 命令。原运行训练到 2500 步，demo 取 step1000。

### Nano

原始 N75 配置见 [configs/N75.json](nano/configs/N75.json)，准备缓存和训练的说明见 [source/README.md](nano/source/README.md)。以下按入口和配置重建训练调用，使用原工程 Python 环境：

```bash
CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/data/gaoya/agent-data/envs/cosmos_nano_acn75/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=2 training/nano/source/train.py \
  --config training/nano/configs/N75.json --mode N75
```

配置：LoRA rank32 / alpha64，生成注意力与生成 FFN；每卡 batch1 × 2 卡 × 累积4，effective batch8；context scale0.75、shift5、1000级网格；future-only loss。原运行计划3000步，demo 取 step1500。训练环境及路径仍需原资产支持。

### Predict2.5

保留 `projects/generic_acn75/` 布局放入对应原生工程。完整原始启动脚本见 [run_n75_mb4.sh](predict25/projects/generic_acn75/run_n75_mb4.sh)，实际训练命令是：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/train.py --config=projects/generic_acn75/n75_mb4_config.py
```

脚本还设置 CUDA/Python 环境、`COSMOS_PHASE=train`、`COSMOS_MECHANISM=N75`，并执行缓存/显存验证。它依赖原先的缓存 contract、ready 记录和显存实验记录；这些外部资产没有随源码一起上传，不应将检查删除后当成同一复现实验。

配置：LoRA rank32 / alpha64，每卡 batch4 × 2 卡 × 累积1，effective batch8；固定验证过的 batch2 编码缓存；context scale0.75、shift5；demo 取 step1500。附带的 `native/` 是入口和主要训练核心供核查使用，完整 NVIDIA 工程仍是依赖。

原始版权和许可声明保留；原生 Predict2.5 的许可见 [LICENSE](predict25/native/LICENSE)。
