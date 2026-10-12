# A equal_sigma：运行命令与结果路径

本次使用 GitHub commit `45250881998aff9d5c839bb7e2b08c1bc49fd657` 的原始 `wan.py` / `nano.py` / `sampling.py`，未修改推理代码。Wan A step1000 与 Nano A step1500，gap038，online 编码、无 cache。两份输出与历史参考 MP4 **逐字节一致**，49帧逐像素一致，MSE=0、max_abs=0。

## GitHub 结果

|模型|视频|配置|验收记录|
|---|---|---|---|
|Wan A step1000|[MP4](wan_A_equal_sigma.mp4) · [直接下载](https://raw.githubusercontent.com/GAO7677/test/main/demos/a-equal-sigma/wan_A_equal_sigma.mp4)|[配置](wan_A_config.json)|[比较结果](wan_A_equal_sigma_comparison.json)|
|Nano A step1500|[MP4](nano_A_equal_sigma.mp4) · [直接下载](https://raw.githubusercontent.com/GAO7677/test/main/demos/a-equal-sigma/nano_A_equal_sigma.mp4)|[配置](nano_A_config.json)|[比较结果](nano_A_equal_sigma_comparison.json)|

[完整验收](verification.json) · [环境版本](environment.json) · [代码来源](provenance.json)。GitHub blob 页面不能播放时使用直接下载链接。

## 推理协议与依赖

从首步起每一步使用固定噪声重建 `c_k=(1-sigma_f,k)*c0+sigma_f,k*eps_c`，`sigma_c=sigma_f`、`t_c=t_f`。Wan 使用噪声时间；Nano 全部13帧使用 noisy 角色。不是首步 clean，也不是 context 自由演化。

这是 A 的 context 输入机制对应结果，不表示所有训推参数完全一致：模型沿用原生时间取整；Nano 训练24 FPS、推理30 FPS。配置中的 `context_scale=0.75` 是历史保留字段，`equal_sigma` 分支实际使用比例1，40步轨迹已核对。

需要本机已有环境、模型工程、权重和输入文件，具体绝对路径在配置中。权重及数据未上传。这不是新机器仅 clone 仓库即可运行的独立安装包。输入和参考 MP4不用于训练；`--verify-against` 仅在生成后比较，不参与采样。

## 当前机器运行命令

以下使用已下载并验收的固定版本代码；每次创建新输出目录，避免覆盖。GPU0与GPU1应空闲。

```bash
TASK=/data/gaoya/agent-data/outputs_v1/context_inference_audit/20261012_github_a_reproduction
REPRO_OUT=$(mktemp -d "$TASK/repeat_XXXXXXXX")
cd "$TASK/repo"

# 此目录在已验证采样版本上仅新增结果/文档；核心源码与上述commit一致。
/data/gaoya/agent-data/envs/physrvg-full-sa/bin/python \
  scripts_mytrain/inference/single_case/wan.py \
  --config demos/a-equal-sigma/wan_A_config.json \
  --mode equal_sigma --no-cache --gpu 0 \
  --output "$REPRO_OUT/wan_A" \
  --verify-against demos/a-equal-sigma/wan_A_equal_sigma.mp4

/data/gaoya/agent-data/envs/cosmos_nano_acn75/bin/python \
  scripts_mytrain/inference/single_case/nano.py \
  --config demos/a-equal-sigma/nano_A_config.json \
  --mode equal_sigma --no-cache --gpu 1 \
  --output "$REPRO_OUT/nano_A" \
  --verify-against demos/a-equal-sigma/nano_A_equal_sigma.mp4
```

在本机另一目录 clone 后，可使用同样命令，将工作目录换成新仓库；环境、模型工程和配置中的绝对路径仍需有效。每次输出包括 `video.mp4`、`config.json`、`sources.json`、`result.json`、`comparison.json`。严格成功标准为 `comparison.json` 的 `status=BYTE_EXACT`；非逐字节一致时脚本返回退出码2。

## 本次实际结果路径

根目录：`/data/gaoya/agent-data/outputs_v1/context_inference_audit/20261012_github_a_reproduction/`

- Wan 视频：`wan_A/video.mp4`；验收：`wan_A/comparison.json`。
- Nano 视频：`nano_A/video.mp4`；验收：`nano_A/comparison.json`。
- 实际原始执行命令：`commands.sh`；任务参数：`jobs.json`。
- 日志：`wan.log`、`nano.log`；环境：`environment.json`；汇总：`verification.json`。
- 历史参考：`/data/gaoya/agent-data/outputs_v1/context_inference_audit/20261010_input_reference/site/{wan,nano}_A_equal_sigma/video.mp4`。

本轮没有覆盖历史文件、训练模型或运行完整benchmark。

## 视频 SHA256

- wan: `924e9e78284c2e02f6e9469e5b97bad03095f86fc42346e3269a5b090898e53d`
- nano: `7c90da3c3ab92dd4c5a9646b6cd7b619ccf8f33e385259de6db9b4c549f92b91`
