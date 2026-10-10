# 单 case 推理

新增：[首步 clean → 重建／自由演化 · 全程条件角色：代码、配置及10组验证](CONDITION_ROLE.md)。

三个入口：[wan.py](wan.py)、[nano.py](nano.py)、[predict25.py](predict25.py)。每次处理一个视频与一条 prompt，支持 `--mode clean|noisy|equal_sigma`，直接输出 `video.mp4`、`config.json`、`result.json`。不读取历史 result.json 或批量任务清单；验证参考视频只在生成结束后用于比较，不作为模型输入。

## 本次9组实测结果

以GitHub已上传的9个demo为参考，**6/9逐字节通过，Predict2.5三组尚未严格复现**。没有把近似结果标成成功，也没有修改参考视频。

| Backbone | Clean | Noisy | Equal sigma |
|---|---|---|---|
| Wan | BYTE_EXACT | BYTE_EXACT | BYTE_EXACT |
| Nano | BYTE_EXACT | BYTE_EXACT | BYTE_EXACT |
| Predict2.5 | 像素不同 | 像素不同 | 像素不同 |

Predict2.5全部输入/context噪声/future初始噪声及context轨迹与原记录一致。clean本次输出另外与最初历史视频逐字节一致，但与后来上传的复跑demo不同；两套参考在本任务开始前即已不同。对已上传demo的RGB PSNR：clean 44.54 dB、noisy 42.69 dB、equal_sigma 42.51 dB。这是与旧生成视频对比，不是与GT对比。

独立诊断固定最终latent、单独重新加载VAE解码，可得到不同输出；历史设置下同进程两次解码一致，换解码进程则不同，改变CuDNN设置也改变结果。说明差异至少包含解码阶段；不能保证旧的未固定解码算法选择的视频在当前运行中逐字节还原。**原入口 noisy 对照进一步确认**：最终latent张量逐元素相同、最大差0，但解码张量和MP4不同；原入口这次生成的视频与已上传noisy demo逐字节一致。由此可定位这对运行的差异发生在VAE解码阶段，而非采样；不据此推断其他所有历史差异的唯一原因。具体CuDNN算法选择尚未记录和锁定。完整诊断及原入口对照见 [verification.json](verification.json)，诊断源码放在 `diagnostics/`（使用原实验环境绝对路径）。

## 默认 gap038 实测配置

| 入口 | 权重 | Seed | Steps | Guidance |
|---|---|---|---|---|
| Wan | N75 step1000 | 42 | 40 | CFG 关闭 |
| Nano | N75 step1500 | 42 | 40 | 6 |
| Predict2.5 | N75 step1500 | 0 | 配置35，实际36个timestep | 7 |

默认读 `presets/<backbone>_gap038.json`。固定输出49帧、512×896，读取前8个RGB帧、保留2个context latent。配置显式记录原始输入、权重、prompt、负提示词及缓存路径。模式含义与 [历史说明](../historical/README.md)、[同等级实验](../equal_sigma/README.md)一致：noisy 使用最近训练网格的0.75倍future sigma；equal_sigma 直接复制future sigma及timestep，固定context噪声、mask策略和采样器保持原协议。

## 运行

从原 PhysRVG 工程根目录运行；生成目录必须全新或为空，且位于 `/data/gaoya`。先检查参数而不加载GPU：

```bash
python3 scripts_mytrain/inference/single_case/wan.py --mode noisy \
  --output /data/gaoya/agent-data/outputs/single_example/wan --check-only
```

默认gap038：

```bash
/data/gaoya/agent-data/envs/physrvg-full-sa/bin/python scripts_mytrain/inference/single_case/wan.py \
  --mode noisy --gpu 2 --output /data/gaoya/agent-data/outputs/single_example/wan

/data/gaoya/agent-data/envs/cosmos_nano_acn75/bin/python scripts_mytrain/inference/single_case/nano.py \
  --mode noisy --gpu 3 --output /data/gaoya/agent-data/outputs/single_example/nano

(
  source scripts_mytrain/inference/historical/frozen/predict25/env.sh
  /data/gaoya/agent-data/envs/cosmos_predict25_official/bin/python scripts_mytrain/inference/single_case/predict25.py \
    --mode noisy --gpu 3 --output /data/gaoya/agent-data/outputs/single_example/predict25
)
```

同一入口将 `--mode` 换成 `clean` 或 `equal_sigma`，并换一个输出目录，即为另两种组合。

自定义单个输入：

```bash
/data/gaoya/agent-data/envs/cosmos_nano_acn75/bin/python scripts_mytrain/inference/single_case/nano.py \
  --video /data/gaoya/my_case/context.mp4 \
  --prompt 'A ball is moving across a raised platform.' \
  --adapter /data/gaoya/my_checkpoint/step-001500 \
  --mode equal_sigma --seed 42 --steps 40 --gpu 3 \
  --output /data/gaoya/agent-data/outputs/my_case/nano_equal
```

`--config` 可选择另一份配置，CLI覆盖其值。基础模型使用 `--model-path`；Wan还支持 `--physrvg-dit-checkpoint`。Predict2.5的 `model_path` 指原生代码工程路径，实际base、VAE、text权重位于配置的 `settings.assets`；换权重时应同步修改资产哈希及训练配置，不仅改LoRA路径。接口遵循当前N75模型及LoRA格式，不保证任意其他训练方案兼容。

## 缓存与依赖

- Wan默认显式加载历史prompt/context缓存并核对源视频及prompt。换视频或prompt会自动去掉默认旧缓存；可用 `--no-cache` 强制在线编码。
- Predict2.5默认显式加载共享context latent，并核对前8帧及VAE哈希；新输入则在线编码，不写全局缓存。
- Nano使用与历史相同的在线前缀编码。
- “禁用缓存后重新编码”不是旧缓存的逐字节等价承诺。更换视频、prompt、权重、环境或参数也不适用gap038验证结论。

这些入口共用 [sampling.py](sampling.py)，并复用 `../historical/frozen/` 中模型加载和输入处理模块；需保留仓库结构。Wan仍依赖原PhysRVG/FastVideo工程（可用 `--repo` 指定），Predict2.5依赖原NVIDIA工程。仓库不包含模型权重、数据、缓存或完整原生工程。

## 验证全部9组

从此GitHub仓库根目录运行（默认参考 `demos/gap038`，原工作区使用 `--reference-dir` 指向下载的demo目录）：

```bash
python3 scripts_mytrain/inference/single_case/verify_matrix.py \
  --backbone all --gpu 3 \
  --output /data/gaoya/agent-data/outputs/single_case_verify
```

该命令串行生成全部9组；也可按 `--backbone wan|nano|predict25` 分开指定空闲GPU。Predict2.5自动加载对应env.sh。它使用本机固定环境路径，其他机器需先按历史文档准备环境和配置。

单次生成可增加 `--verify-against /path/to/existing.mp4`。对比写入 `comparison.json`：MP4 SHA-256、解码帧完全相等、全部49帧MSE/PSNR和最大像素差。任何非逐字节相同结果都返回退出码2，即使像素相等；不会自动放宽阈值或将相近结果标成成功。

Predict2.5额外保存 `final_latent.pt`，用于判断差异是否出现在采样或解码阶段；该诊断产物仅保存在数据盘，不上传GitHub。

实测矩阵和未解决差异见 [verification.json](verification.json)。这里的复现目标是已上传的9个demo；不能据此宣称此前所有历史视频或任意环境都可逐字节复现。
