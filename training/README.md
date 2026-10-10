# Demo 权重对应的训练代码

这里对应已上传的 gap038 demo 所用 **N75 权重**。同一 backbone 的 clean、原 noisy、equal_sigma 三个推理版本共用同一份训练权重；推理 clean 不表示它来自 C/clean 训练，equal_sigma 也没有重新训练。

| Backbone | Demo checkpoint | 实际训练入口 | 训练核心 | 来源证据 |
|---|---|---|---|---|
| Wan | N75 step1000 | [train_full_sa_pybullet_rl_lora_raw.py](wan/workspace/scripts_mytrain/train/train_full_sa_pybullet_rl_lora_raw.py) | [train_full_sa_pybullet_aux.py](wan/workspace/scripts_mytrain/train/train_full_sa_pybullet_aux.py) | [训练启动记录](wan/records/launch.json)、[resolved config](wan/records/resolved_config.json) |
| Cosmos3-Nano | N75 step1500 | [train.py](nano/source/train.py) | [common.py](nano/source/common.py)：context_pair / noise_input / future_loss | 运行目录 `source/` 快照；13 个文件全部与[当时的哈希清单](nano/source_hashes.json)一致 |
| Cosmos-Predict2.5 | N75 step1500 | [官方 scripts/train.py](predict25/native/scripts/train.py) + [n75_mb4_config.py](predict25/projects/generic_acn75/n75_mb4_config.py) | [model.py](predict25/projects/generic_acn75/model.py)、[mechanism.py](predict25/projects/generic_acn75/mechanism.py) | 运行目录 `code_snapshot/`；[启动记录](predict25/records/launch.json)及[checkpoint metadata](predict25/records/step001500_metadata.json) |

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
