# 首步 clean → 重建／自由演化 · 全程条件角色

实际运行入口：[condition_role_case.py](condition_role_case.py)。实现：[clean_first_hooks.py](clean_first_hooks.py)；自由演化共用 [initial_only_hooks.py](initial_only_hooks.py)。请保留整个仓库结构，不能只复制入口文件。

## 实验定义

令 k=0 为首次模型调用，c0 为前8帧编码得到的2个 context latent，future 为其余11个 latent。

| 设置 | 首步 clean → 重建 | 首步 clean → 自由演化 |
|---|---|---|
| `--protocol` | `clean_rebuild` | `clean_free` |
| 首次 context 输入 | c0 | c0 |
| 后续 context | 每步重建 `(1-sigma_f[k])*c0 + sigma_f[k]*epsilon_c`，epsilon_c 整段固定 | 直接使用上一步 scheduler 输出，不重建、不补噪、不回填 c0 |
| 状态更新 | 保留各 backbone 原重建版本的更新策略 | 完整 context+future 预测速度交给原生多步 scheduler，保留历史状态 |
| 输入角色 | 全程条件角色 | 全程条件角色 |

两组均使用 `--mode equal_sigma --no-cache`，首步 clean 由 protocol 覆盖。重建版保留原 context 处理：Wan 只推进 future；Nano 屏蔽 context 速度并回填；Predict2.5 使用解析 context 速度。自由演化版仍读取、推进 context 预测速度，最终不回填干净 context。

“条件角色”及配套时间设置：

- Wan：没有独立角色通道，context token timestep=0。
- Nano：context 不在 noisy frame indexes 中，不获得 diffusion time embedding；自由演化仍读取 context 输出。
- Predict2.5：context 条件 mask=1、future mask=0；context 使用原生 conditional_frame_timestep（本次0.1）。

future 时间不变。角色 hook 在输入构造之后执行，不改 latent 数值。这不是固定时间仅修改 mask 的对照。`condition_role_audit.json` 记录实际模型角色和时间；基础 hook 的 trace 记录角色覆盖之前的时间。

## 配置与复现

[presets/condition_role](presets/condition_role) 包含 Wan A/N75 step1000、Nano A/N75 step1500、Predict2.5 N75 step1500，共5份实际配置，每份运行两个 protocol，共10组。原始输入各 backbone 不完全相同，应在同一配置内比较。

从仓库根目录执行，输出目录须不存在或为空：

```bash
# Nano A：重建；自由演化改为 clean_free，并更换输出目录
python scripts_mytrain/inference/single_case/condition_role_case.py nano \
  --protocol clean_rebuild --config scripts_mytrain/inference/single_case/presets/condition_role/nano_A.json \
  --mode equal_sigma --no-cache --gpu 2 --output /path/to/output/nano_A_clean_rebuild

# Wan A：自由演化；重建改为 clean_rebuild，并更换输出目录
python scripts_mytrain/inference/single_case/condition_role_case.py wan \
  --protocol clean_free --config scripts_mytrain/inference/single_case/presets/condition_role/wan_A.json \
  --mode equal_sigma --no-cache --gpu 2 --output /path/to/output/wan_A_clean_free

# Predict2.5 N75：重建；自由演化改为 clean_free，并更换输出目录
# 先激活 Predict2.5 环境，并设置其 CUDA/cuDNN 库路径
python scripts_mytrain/inference/single_case/condition_role_case.py predict25 \
  --protocol clean_rebuild --config scripts_mytrain/inference/single_case/presets/condition_role/predict25_N75.json \
  --mode equal_sigma --no-cache --gpu 2 --output /path/to/output/predict25_N75_clean_rebuild
```

Wan/Nano N75 替换为对应 `wan_N75.json` / `nano_N75.json`。公开配置中的路径已替换为 /path/to/... 占位符，运行前必须填写；需要外部原模型工程、权重、输入视频及环境。修改配置或使用 `--repo` / `--model-path` / `--adapter` / `--video` 覆盖路径。Wan `--repo` 指向完整 PhysRVG/FastVideo 工程；Predict2.5 `--model-path` 指向原生代码工程。依赖说明见 [单 case README](README.md)。`--no-cache` 在线编码输入及 prompt，不表示禁止读取模型权重文件。

## 验证与可视化

[验证记录](condition_role_verification.json) 保存10组已完成运行的视频 SHA256、49帧检查和实际角色／时间验证。Wan/Nano各40步，Predict2.5配置35步、实际36步。CFG使 Nano/Predict2.5 每步调用模型两次；Wan 本次关闭 CFG。发布代码时没有重跑 GPU，也不保证跨环境视频逐字节一致。

CPU 检查：

```bash
cd scripts_mytrain/inference/single_case
python -m unittest test_condition_role test_clean_first test_initial_only
```

本次上传源码、配置模板、测试和验证记录，不包含新视频、数据集或权重。本机服务地址及环境路径未公开。
