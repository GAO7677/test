# 历史 clean / context 加噪推理复现

只整理 Wan、Cosmos3-Nano、Cosmos-Predict2.5 已有的 clean / 固定 context 加噪实验。**不包含 RePaint、回跳重采样或提前 n 步实验。**

六个入口可以分别执行；文件内保留原主循环，`frozen/` 保存同目录辅助代码和不同历史版本。它们共用 `reproduce.py` 做版本选择、参数恢复、校验与输出隔离，不能只拷走单个文件运行。权重、视频、缓存仍使用 `/data/gaoya` 中的原资产，没有复制到代码目录。

**2026-10-10 实测状态：** Wan、Nano 的 clean／noisy 默认样本共 4 个视频逐字节一致；Predict2.5 两个视频尚未达到逐字节复现，不能宣称六个入口全部通过。Predict2.5 的输入、权重加载和完整 context／timestep 审计一致，但相对旧视频的 RGB PSNR 分别为 44.54 dB（clean）、42.72 dB（noisy），这里是与旧生成结果比较，不是与 GT 比较。未拆分的原始脚本另跑一例也不能还原旧视频，其差异来源尚未完全锁定。原代码保留 `cudnn.deterministic=False`、`benchmark=True`；没有为消除差异而擅自修改历史算法。详细证据见 [verification.json](verification.json)。

| Backbone | clean 入口 | 加噪入口 | 主要实现位置 |
|---|---|---|---|
| Wan | `wan_clean.py` | `wan_noisy.py` | `ContextNoise`、`sample`、`Experiment`、`generate` |
| Cosmos3-Nano | `nano_clean.py` | `nano_noisy.py` | `main`、`CleanContextAudit`；版本对应的 `NanoContext` / `ContextNoise` |
| Cosmos-Predict2.5 | `predict25_clean.py` | `predict25_noisy.py` | `main`、`install_predict` 内的 `denoise`；`frozen/predict25/inference_runtime.py` |

## 使用方式

工作目录：`/home/gaoya/code_V2V_baselines/PhysRVG-main`。

最重要的参数是 `--reference 原结果/result.json`：它决定输入、权重、步数、CFG、seed、噪声等级和历史实现版本。**不会根据 N75 名称擅自改成另一种推理算法，也不会把同名的旧 Test70 和 Test70_v2 case 混用。** clean 入口拒绝 noisy 结果，noisy 入口拒绝 clean 结果。

先检查，不加载 GPU、不写文件：

```bash
/usr/bin/python3 scripts_mytrain/inference/historical/nano_noisy.py --check-only
```

默认参考均为 `v2v_gap_038`、N75：Wan 是旧 Test70 的 step1000；Nano 是 Test70_v2 的 step1500；Predict2.5 是旧 Test70 triplet 的 step1500。各默认 noisy 对应原来的 0.75 倍最近训练网格分支。默认参考的精确路径可在 `--check-only` 输出的 `reference` 字段查看。

生成结果必须指定一个**全新或空的** `/data/gaoya/...` 目录；旧结果不会被覆盖，生成后不自动发布可视化页面。以下命令在前台执行。GPU 编号按实际空闲情况选择，示例使用 2。

```bash
/data/gaoya/agent-data/envs/physrvg-full-sa/bin/python scripts_mytrain/inference/historical/wan_clean.py --gpu 2 --output /data/gaoya/agent-data/outputs/historical_repro/wan_clean
/data/gaoya/agent-data/envs/physrvg-full-sa/bin/python scripts_mytrain/inference/historical/wan_noisy.py --gpu 2 --output /data/gaoya/agent-data/outputs/historical_repro/wan_noisy
/data/gaoya/agent-data/envs/cosmos_nano_acn75/bin/python scripts_mytrain/inference/historical/nano_clean.py --gpu 2 --output /data/gaoya/agent-data/outputs/historical_repro/nano_clean
/data/gaoya/agent-data/envs/cosmos_nano_acn75/bin/python scripts_mytrain/inference/historical/nano_noisy.py --gpu 2 --output /data/gaoya/agent-data/outputs/historical_repro/nano_noisy
```

Predict2.5 需要原 CUDA 库环境。下面两条命令各在子 shell 内执行，不改变父 shell 的工作目录和环境：

```bash
(
  cd /home/gaoya/Code_Video/Code_MODEL/cosmos-predict2.5
  source /home/gaoya/code_V2V_baselines/PhysRVG-main/scripts_mytrain/inference/historical/frozen/predict25/env.sh
  python /home/gaoya/code_V2V_baselines/PhysRVG-main/scripts_mytrain/inference/historical/predict25_clean.py --gpu 2 --output /data/gaoya/agent-data/outputs/historical_repro/predict25_clean
)
(
  cd /home/gaoya/Code_Video/Code_MODEL/cosmos-predict2.5
  source /home/gaoya/code_V2V_baselines/PhysRVG-main/scripts_mytrain/inference/historical/frozen/predict25/env.sh
  python /home/gaoya/code_V2V_baselines/PhysRVG-main/scripts_mytrain/inference/historical/predict25_noisy.py --gpu 2 --output /data/gaoya/agent-data/outputs/historical_repro/predict25_noisy
)
```

例如，复现 Nano A 的 0.9 倍噪声：

```bash
/data/gaoya/agent-data/envs/cosmos_nano_acn75/bin/python scripts_mytrain/inference/historical/nano_noisy.py \
  --reference /data/gaoya/AAA_test_video/Context_Video_bench/Test70_v2/noise/nano_20cases/ac-noise-sweep/A/step1500/noise090/results/v2v_gap_038/result.json \
  --gpu 2 --output /data/gaoya/agent-data/outputs/historical_repro/nano_A_noise090
```

Nano 优先从结果附近寻找完整 launch config；找不到时用 `--config 原始完整配置.json`，不是只包含评价参数的 `config.json`。脚本会检查采样参数和 checkpoint 内的 noise / LoRA 配置是否一致。

## 保留的历史差异

这里 `c` 是干净 context latent，`s_f(i)` 是 native scheduler 的 future sigma，`Q` 是该实验训练 sigma 表的最近邻。所有公式仍按原代码的 dtype、转换顺序执行。

- **Wan**：固定 `epsilon_c = initial_noise[:, :, :2]`；输入为 `(1-s_c)c+s_c epsilon_c`。比例 0/1 走精确端点，中间比例用训练配对表；context token 的 timestep 取配对值。只对 future 调用 scheduler，再拼回 clean context。40 步、seed42、`do_cfg=False`；配置里写 guidance5 并不表示启用了 CFG。clean 入口复现原 campaign 的 scale0，已保留原端点处理。
- **Nano 早期 Test89**：`frozen/nano/early/campaign/noise_hook.py` 重放 native generator 获得同一份初始噪声；scale0 完全保留 native clean role，scale1 使用精确 future sigma / timestep，中间档用训练网格。
- **Nano 后期 Test70_v2**：独立 generator，context seed20261007，固定 FP32 `epsilon_c`；`s_c=Q(lambda*s_f)`，`t_c=N*s_c`。context 输入为 FP32 插值后转换到模型 dtype。native clean role 与 noisy role 分开；noisy context 和 future 使用同一 transformer，但 context 的 token role / timestep 被替换；scheduler 后恢复 clean context。保存的 flow_shift10 不会覆盖模型实际 Karras 分支，整理时未改时间表。
- **Nano 后期各版本**：`v2` 保留 `matched` / `shared075`；`n75_sweep` 对应 N75 的 0.5/0.9；`n75_100` 对应 N75 的 1；`ac_sweep` 对应 A/C 的 0.5/0.9/1。A 的 `matched` 和 `scaled` 不是同一种端点策略。入口根据旧记录选择冻结的类，不把它们合并重写。
- **Predict2.5**：clean 调用 native denoise；triplet noisy 使用独立固定 FP32 context 噪声、训练 sigma 网格及逐帧 timestep，将条件输入 mask 置零后直接调用 net，并把 context 速度替换回 `native_noise - native_gt`；保留 native 全状态 scheduler / decoder。`clean_noisy` 虽然值是 clean，但属于 noisy role，必须通过 noisy 入口。triplet 的共享 context cache 仅适用于旧 gap038，缺失时直接报错，不重新制造一个“等价”缓存。

Predict2.5 原 70-case manifest 路径已迁移。triplet 复现使用其归档的一例清单，校验输入路径、提示词和视频哈希；因此复现结果的协议路径哈希与旧记录可以不同，但采样输入不变。

## 校验与输出

- `source_manifest.json`：冻结辅助文件的原始位置与 SHA256。
- `runtime_manifest.json`：三个环境的包版本、模型 / scheduler 关键源文件哈希；运行前拒绝未记录的修改。
- `reproduction_plan.json`：本次从旧结果恢复的具体参数与资产路径。
- `reproduction_sources.json`：入口、启动器、冻结文件和运行库的版本记录。
- `results/.../result.json`：沿用原来的生成回执。
- `reproduction_comparison.json`：视频文件是否逐字节相同，以及旧回执中存在的输入 / timestep / trace / latent 哈希比较。`DIFFERENT_REQUIRES_REVIEW` 不能当作成功复现；进程返回退出码 2，同时保留生成文件供检查。

新回执中的源码路径、单例 manifest、计时和输出路径会变化，不应要求整个 JSON 逐字节一致。逐字节视频复现只对已验证的软件环境和设备条件作证；不同 CUDA、attention kernel 或编码器版本需要重新验证。

CPU 检查命令：

```bash
/usr/bin/python3 -m unittest discover -s scripts_mytrain/inference/historical -p test_reproduction.py -v
```

GPU 复跑结果与覆盖范围见 `verification.json`。预检查通过不等于已经复跑该结果；未列为 GPU 验证的组合不作逐字节复现承诺。
