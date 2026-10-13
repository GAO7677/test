# Test70_v2：单 case 入口与复现状态

本目录包含7个分支：6个分支三例均MP4逐字节一致；Predict2.5 baseline按用户要求收录，但三例均未通过严格复现。每个入口接收一份video/prompt，通过原归档入口或runtime执行；没有重写forward、scheduler、CFG或VAE。

**依赖边界：主文件是轻量包装，原模型仓库、历史辅助模块、环境及权重仍需在配置所列本机路径存在。不是仅clone即可运行的独立安装包；未上传模型权重、数据或复制完整辅助推理代码。**

| 分支 | 已完成 | 字节一致 | 本目录发布 |
|---|---:|---:|---|
| nano_baseline | 3/3 | 3/3 | [nano_baseline.py](nano_baseline.py) |
| predict25_baseline | 3/3 | 0/3 | [predict25_baseline.py](predict25_baseline.py)；已收录，复现未通过 |
| wan_lora_off | 3/3 | 3/3 | [wan_lora_off.py](wan_lora_off.py) |
| physrvg_official | 3/3 | 3/3 | [physrvg_official.py](physrvg_official.py) |
| nano_official_ft | 3/3 | 3/3 | [nano_official_ft.py](nano_official_ft.py) |
| predict25_official_ft | 3/3 | 3/3 | [predict25_official_ft.py](predict25_official_ft.py) |
| predict25_official_ft_noema | 3/3 | 3/3 | [predict25_official_ft_noema.py](predict25_official_ft_noema.py) |

[逐例哈希与像素验收](verification.json) · [实际环境版本](environment.json)

验证case：v2v_gap_038、v2v_bowl_r180、v2v_domino_g045。参考来源为统一页面Test70_v2，不能将本次结果外推到其他checkpoint或完整88例。

Nano official-ft step1500；Predict2.5 official-ft step1000；Predict2.5 official-ft-noEMA step1500。baseline无项目LoRA；Wan LoRA-off仍使用页面指定PhysRVG DiT，不能标为未经任何训练的原始Wan。PhysRVG保留固定prompt。

Nano official-ft保留8RGB→重复末帧到9→编码取2latent；Predict2.5保留prefix8。均49帧、512×896、30FPS。Nano40步/CFG6/seed42；Predict2.5 num_steps35/CFG7/seed0；Wan/PhysRVG40步/seed42/无CFG。K600的24FPS参数未因此改变。

## 本机运行方法

先进入仓库根目录。每次使用新的输出目录；参数支持`--check-only`在不加载GPU的情况下核对源码和输入。禁止物理GPU4。以下命令的原始环境初始化保留在验收活动中：

`/data/gaoya/agent-data/outputs_v1/k600_context_mechanisms/20261013_page_single_case_v1/run_one.sh`

该环境脚本没有随本目录复制；它设置原环境、CUDA库与Python模块路径。可在原验收机器直接使用下面命令；迁移到其他机器需先准备对应环境和外部源码，不能省略配置中的源码哈希验收。

### nano_baseline

```bash
bash /data/gaoya/agent-data/outputs_v1/k600_context_mechanisms/20261013_page_single_case_v1/run_one.sh nano scripts_mytrain/inference/page_single_case/nano_baseline.py --config scripts_mytrain/inference/page_single_case/configs/nano_baseline.json --video /data/gaoya/agent-data/outputs_v1/geometry_transfer/20261006_test70_v2_latest/test70_v2/samples/v2v_gap_038/videos/rgb_cycles.mp4 --prompt 'A ball is moving across a raised platform.' --case v2v_gap_038 --gpu 5 --output /data/gaoya/agent-data/outputs_v1/k600_context_mechanisms/repeat_nano_baseline_NEW
```

### predict25_baseline（复现未通过）

使用官方post-trained基础权重、无项目LoRA；前8RGB编码为2个条件latent，全程clean条件角色（mask=1、原生条件时间0.1）。保留历史入口及原生采样器，num_steps35实际36次迭代、CFG7、seed0、49帧、512×896、30FPS，在线编码。

三例均成功生成，但与历史参考存在像素差异：gap038 MSE=11.167318、bowl_r180=9.359859、domino_g045=7.483973（解码RGB 0–255尺度）。输入、已核对源码、权重及采样参数一致，差异阶段尚未定位；不能将本次收录视为复现通过。[逐例证据](predict25_baseline_verification.json)。仍依赖配置中列出的外部入口/runtime。

```bash
bash /data/gaoya/agent-data/outputs_v1/k600_context_mechanisms/20261013_page_single_case_v1/run_one.sh predict25 scripts_mytrain/inference/page_single_case/predict25_baseline.py --config scripts_mytrain/inference/page_single_case/configs/predict25_baseline.json --video /data/gaoya/agent-data/outputs_v1/geometry_transfer/20261006_test70_v2_latest/test70_v2/samples/v2v_gap_038/videos/rgb_cycles.mp4 --prompt 'A ball is moving across a raised platform.' --case v2v_gap_038 --gpu 5 --output /data/gaoya/agent-data/outputs_v1/k600_context_mechanisms/repeat_predict25_baseline_NEW
```

### wan_lora_off

```bash
bash /data/gaoya/agent-data/outputs_v1/k600_context_mechanisms/20261013_page_single_case_v1/run_one.sh wan scripts_mytrain/inference/page_single_case/wan_lora_off.py --config scripts_mytrain/inference/page_single_case/configs/wan_lora_off.json --video /data/gaoya/agent-data/outputs_v1/geometry_transfer/20261006_test70_v2_latest/test70_v2/samples/v2v_gap_038/videos/rgb_cycles.mp4 --prompt 'A ball is moving across a raised platform.' --case v2v_gap_038 --gpu 5 --output /data/gaoya/agent-data/outputs_v1/k600_context_mechanisms/repeat_wan_lora_off_NEW
```

### physrvg_official

```bash
bash /data/gaoya/agent-data/outputs_v1/k600_context_mechanisms/20261013_page_single_case_v1/run_one.sh wan scripts_mytrain/inference/page_single_case/physrvg_official.py --config scripts_mytrain/inference/page_single_case/configs/physrvg_official.json --video /data/gaoya/agent-data/outputs_v1/geometry_transfer/20261006_test70_v2_latest/test70_v2/samples/v2v_gap_038/videos/rgb_cycles.mp4 --prompt 'A ball is moving across a raised platform.' --case v2v_gap_038 --gpu 5 --output /data/gaoya/agent-data/outputs_v1/k600_context_mechanisms/repeat_physrvg_official_NEW
```

### nano_official_ft

```bash
bash /data/gaoya/agent-data/outputs_v1/k600_context_mechanisms/20261013_page_single_case_v1/run_one.sh nano_native scripts_mytrain/inference/page_single_case/nano_official_ft.py --config scripts_mytrain/inference/page_single_case/configs/nano_official_ft.json --video /data/gaoya/agent-data/outputs_v1/geometry_transfer/20261006_test70_v2_latest/test70_v2/samples/v2v_gap_038/videos/rgb_cycles.mp4 --prompt 'A ball is moving across a raised platform.' --case v2v_gap_038 --gpu 5 --output /data/gaoya/agent-data/outputs_v1/k600_context_mechanisms/repeat_nano_official_ft_NEW
```

### predict25_official_ft

```bash
bash /data/gaoya/agent-data/outputs_v1/k600_context_mechanisms/20261013_page_single_case_v1/run_one.sh predict25_native scripts_mytrain/inference/page_single_case/predict25_official_ft.py --config scripts_mytrain/inference/page_single_case/configs/predict25_official_ft.json --video /data/gaoya/agent-data/outputs_v1/geometry_transfer/20261006_test70_v2_latest/test70_v2/samples/v2v_gap_038/videos/rgb_cycles.mp4 --prompt 'A ball is moving across a raised platform.' --case v2v_gap_038 --gpu 5 --output /data/gaoya/agent-data/outputs_v1/k600_context_mechanisms/repeat_predict25_official_ft_NEW
```

### predict25_official_ft_noema

```bash
bash /data/gaoya/agent-data/outputs_v1/k600_context_mechanisms/20261013_page_single_case_v1/run_one.sh predict25_native scripts_mytrain/inference/page_single_case/predict25_official_ft_noema.py --config scripts_mytrain/inference/page_single_case/configs/predict25_official_ft_noema.json --video /data/gaoya/agent-data/outputs_v1/geometry_transfer/20261006_test70_v2_latest/test70_v2/samples/v2v_gap_038/videos/rgb_cycles.mp4 --prompt 'A ball is moving across a raised platform.' --case v2v_gap_038 --gpu 5 --output /data/gaoya/agent-data/outputs_v1/k600_context_mechanisms/repeat_predict25_official_ft_noema_NEW
```

## 产物与变更范围

每次产生`single_case_config.json`、`single_case_result.json`、`video.mp4`及原生审计记录。参考视频只由外部验收程序在生成完成后读取，不参与采样。单case适配只更改样本选择、运行GPU和输出目录；保留原模型加载、采样参数及输入预处理。

完整本地证据：`/data/gaoya/agent-data/outputs_v1/k600_context_mechanisms/20261013_page_single_case_v1/REPORT.md`。历史代码/视频没有覆盖，K600自动队列没有切换。
