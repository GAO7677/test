# Cosmos Nano A/C/N75：配置驱动训练

入口配置：`experiment.json`。所有入口均支持 `--config /absolute/path/to/config.json`；省略时读取脚本旁的 `experiment.json`。JSON 不支持注释。

## 修改训练模块

`lora.target_modules` 使用完整模块名或以点分隔的模块名后缀匹配，不使用通配符。例如：

```json
"lora": {
  "rank": 32,
  "alpha": 64,
  "dropout": 0.0,
  "target_modules": [
    "self_attn.add_q_proj",
    "self_attn.add_k_proj",
    "self_attn.add_v_proj",
    "self_attn.to_add_out",
    "mlp_moe_gen.gate_proj",
    "mlp_moe_gen.up_proj",
    "mlp_moe_gen.down_proj"
  ]
}
```

默认 36 层各注入 7 处，共 252 处，87,293,952 个可训练参数。所有原始参数始终冻结，仅新增 LoRA A/B 可训练。删除三个 `mlp_moe_gen.*` 可改为仅生成注意力。指定 `layers.0.self_attn.add_q_proj` 可只选第 0 层。文本分支 `self_attn.to_q/to_k/to_v/to_out` 和 `mlp.*` 默认不注入；不要使用过宽后缀（如 `up_proj`）以免选中两个分支。没有匹配、重叠匹配或匹配非 Linear 会立即报错。

修改后先检查，不加载权重、不占 GPU：

```bash
/data/gaoya/agent-data/envs/cosmos_nano_acn75/bin/python /home/gaoya/Code_Video/Code_MODEL/cosmos-framework/experiments/generic2175_acn75/inspect_config.py --config /home/gaoya/Code_Video/Code_MODEL/cosmos-framework/experiments/generic2175_acn75/experiment.json
```

## 其他配置

| 字段 | 控制内容 |
|---|---|
| `paths` | 模型、原始数据、样本列表、缓存和输出目录 |
| `lora` | 目标模块、rank、alpha、dropout |
| `data` | 样本数、帧数、分辨率、FPS、caption 字段、条件 latent 数、评估条件 RGB 帧数 |
| `training` | seed、步数、梯度累积、保存间隔、精度、梯度检查点、AdamW 学习率/betas/eps/weight decay、梯度裁剪、学习率调度和 warmup |
| `noise` | waver 系数、噪声 shift、时间步数、各实验条件噪声比例 |
| `runtime` | 训练/缓存/评估物理 GPU 编号和 CPU 线程数 |
| `evaluation` | 采样步数、seed、CFG guidance、flow shift、benchmark 列表 |

有效 batch = `per_gpu_batch × len(train_gpus) × grad_accum`，默认 `1 × 2 × 4 = 8`。当前打包实现仅支持每卡 batch 1，修改为其他值会报错。支持 FP32/BF16 主权重、BF16 计算；时间嵌入 MLP 固定 FP32。优化器固定为 AdamW。物理 GPU 4 禁止使用。

训练固定采用 future flow matching MSE，仅从 `context_latents` 之后计算损失。scale=0 使用干净条件；scale=1 使用与未来相同的噪声；中间值映射到最近配对 sigma。VAE 缓存固定 BF16 编码、posterior mode、前缀窗口和拉伸双线性 resize；修改帧数/分辨率/文本/样本列表时请指定新缓存目录并执行 `prepare.py --config ...`。缓存准备仍要求原始 generic case 格式及历史 caption 索引。Test70/Test89 的匹配评估限定 8 RGB/2 latent 条件帧，修改不支持的组合会报错。

## 短程验证 / 正式训练

```bash
CUDA_VISIBLE_DEVICES=5,6 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 /data/gaoya/agent-data/envs/cosmos_nano_acn75/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 /home/gaoya/Code_Video/Code_MODEL/cosmos-framework/experiments/generic2175_acn75/train.py --config /home/gaoya/Code_Video/Code_MODEL/cosmos-framework/experiments/generic2175_acn75/experiment.json --mode C --smoke --run-name my_smoke --stop-after 2 --verify-roundtrip
```

正式训练去掉 `--smoke --run-name my_smoke --stop-after 2 --verify-roundtrip`。模式用 `--mode C`、`A` 或 `N75`。运行环境 `CUDA_VISIBLE_DEVICES` 必须与配置中的物理卡列表一致。运行时可用 `--max-steps / --grad-accum / --save-every / --master-dtype` 覆盖配置；最终生效值写入 `resolved_config.json`。

每个 run 保存 `trainable_modules.json`（实际可训练参数名和统计）与完整 `resolved_config.json`。检查点写入 `adapter_config.json`；推理从检查点读取 LoRA 目标、rank、alpha，而不是套用当前训练配置。恢复会比较完整实验配置及缓存指纹，不允许将旧 rank16 或不同模块的检查点直接恢复成新方案。更改实验请使用新输出目录。

`run_campaign.py --config ...` 依次执行缓存、训练和评估，要求输出目录已有与当前配置/代码哈希一致的 `preflight_pass.json`、`source_hashes.json`。修改配置后直接用上述短程命令验证；不要复制旧实验的 PASS 文件绕过匹配检查。历史输出里的 `source/` 是不可变快照，不会随本目录代码更新。
