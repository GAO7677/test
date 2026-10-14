# FewShotLab

**few-shot 真实图片 + caption → 训练 LoRA → 输入 prompt → 生成图片及逐图 JSON。**

`tips：基于两个基模（SD2.1、FLUX.2 Klein Base 4B）实现，可以任选`
`目前是两种方法，BeyondObjects 支持 SD2.1、FLUX.2 Klein Base 4B、ChimeraLoRA 支持 SD2.1，使用统一命令管理训练和生成结果。`
`推荐FLUX.2 Klein Base 4B；`

## 1. 安装

需要 Linux、Python 3.10、NVIDIA CUDA GPU、`git` 和 `tmux`。下载或克隆本仓库后，在仓库根目录执行以下命令。示例中的 `/data/...` 均可替换为自己的磁盘路径。

```bash
git clone https://github.com/GAO7677/few-shot.git
cd few-shot
```

**BeyondObjects SD2.1 / ChimeraLoRA：**

```bash
python3.10 -m venv /data/envs/fewshot-sd
source /data/envs/fewshot-sd/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/sd21.txt
```

**FLUX.2 使用独立环境：**

```bash
python3.10 -m venv /data/envs/fewshot-flux
source /data/envs/fewshot-flux/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/flux2.txt
```

4-shot 短训练实测峰值显存约为 SD2.1 11 GiB、FLUX.2 15.35 GiB；实际用量随样本数和配置变化。

## 2. 下载底座模型

`paths.base_model` 填下载后的本地路径。

| 基模 | Hugging Face 模型 |
|---|---|
| BeyondObjects SD2.1 / ChimeraLoRA | [sd2-community/stable-diffusion-2-1-base](https://huggingface.co/sd2-community/stable-diffusion-2-1-base) |
| FLUX.2 Klein Base 4B | [black-forest-labs/FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) |



下载命令可以参考使用 [Hugging Face CLI](https://hf-mirror.com/)

## 3. 准备数据

`asset/my_data/` 提供两组 RarePlanes 示例，各4张图片。使用自己的数据时，按相同结构组织：

```text
asset/my_data/
├── medium_crop/
│   ├── caption.txt
│   └── 01.png … 04.png
└── small_tile/
    ├── caption.txt
    └── 01.png … 04.png
```

同一文件夹的图片共用一条 caption，不同描述分文件夹存放。例如 `medium_crop/caption.txt`：

```text
an overhead satellite RGB image of a medium civil transport aircraft at an airport
```

Caption 写清主体、视角和场景，应适用于组内所有图片。描述错误或过于笼统会影响生成效果。推理时可以使用原 caption，也可以修改描述尝试新的场景或构图。

## 4. 训练与生成

以 BeyondObjects SD2.1 为例，配置文件 `configs/rareplanes_sd21_example.yaml` 如下：

```yaml
name: rareplanes_sd21_example
model: beyondobjects_sd21
paths:
  base_model: ${SD21_MODEL}
  output_root: ${FEWSHOT_OUTPUT}
data:
  root: ${FEWSHOT_DATA}
train:
  rank: 4
  steps: 400
  batch_size: 4
tracking:
  enabled: false
  project: fewshot-generation
```

本例将两组共8张图片一起训练，rank=4、400步。W&B 默认关闭。

在 SD 环境、仓库根目录执行，按需修改模型和输出路径：

```bash
export SD21_MODEL=/data/models/sd21-base
export FEWSHOT_DATA=asset/my_data
export FEWSHOT_OUTPUT=/data/experiments/rareplanes_sd21_example

# 训练完成后，自动使用训练caption生成底座/LoRA对照图
python run.py run --config configs/rareplanes_sd21_example.yaml
```

**只训练**时，将 `run` 换成 `train`：

```bash
python run.py train --config configs/rareplanes_sd21_example.yaml
```

**已有权重，只生成**：直接使用同一个 `FEWSHOT_OUTPUT`。

```bash
python run.py generate --run-dir "$FEWSHOT_OUTPUT" \
  --prompt "an overhead satellite RGB image of a medium civil transport aircraft at an airport" \
  --num-images 10
```

`--run-dir` 选择权重，`--prompt` 控制内容；省略 prompt 时循环使用训练 caption。上述命令生成10张 LoRA 图及10张相同 seed 的底座对照图。

## 5. 查看输出

`FEWSHOT_OUTPUT` 就是本次实验的 `RUN_DIR`，配置和 LoRA 权重直接保存在其中，不再添加时间戳子目录。新训练请使用不存在或空的目录；已有实验用 `generate` 生成；SD/FLUX 可用 `resume` 续训。

```text
FEWSHOT_OUTPUT/                # 即 RUN_DIR
├── resolved_config.yaml   # 完整配置
├── run.json               # 模型与环境信息
├── status.json            # 当前进度
├── source_snapshot.zip    # 源码快照
├── inputs/                # 训练图片和caption副本
├── manifests/             # 输入清单及检查结果
├── logs/                  # console.log、metrics.jsonl
├── adapter/               # LoRA权重及训练checkpoint
├── generations/<批次>/
│   ├── base/              # 底座图：000000.png + 000000.json
│   ├── lora/              # LoRA图：000000.png + 000000.json
│   ├── config.json        # 生成参数
│   ├── index.json         # 完整记录及校验信息
│   └── complete.json      # 完成记录
└── report/index.html      # 图片与prompt对照页面
```

每张图片对应一个同名 JSON，仅保留图片名、prompt、seed、模型/LoRA路径、训练步数、尺寸和采样参数；路径均相对于该 JSON。negative prompt、类别条件和 Chimera 混合权重仅在使用时记录。

完整校验信息集中在批次 `index.json`，用于断点生成和同 seed 对照。直接用浏览器打开 `$FEWSHOT_OUTPUT/report/index.html` 查看结果。

已完成的生成示例：[FLUX.2 LoRA 图片与逐图 JSON](examples/flux2/README.md)。
