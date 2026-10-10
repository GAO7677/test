# gap038 推理 Demo

六个视频来自 2026-10-10 对整理后的六个入口的实际复跑，均为 `v2v_gap_038` 样本，使用 N75 权重。文件保持原始 MP4 字节，未剪辑、拼接或重新编码。

| Backbone | 权重 step | Clean | Context noise | 与历史结果的核对 |
|---|---|---|---|---|
| Wan | 1000 | [播放／下载](wan_clean.mp4) · [脚本](../../scripts_mytrain/inference/historical/wan_clean.py) | [播放／下载](wan_noisy.mp4) · [脚本](../../scripts_mytrain/inference/historical/wan_noisy.py) | 两种推理均与历史视频逐字节一致 |
| Cosmos3-Nano | 1500 | [播放／下载](nano_clean.mp4) · [脚本](../../scripts_mytrain/inference/historical/nano_clean.py) | [播放／下载](nano_noisy.mp4) · [脚本](../../scripts_mytrain/inference/historical/nano_noisy.py) | 两种推理均与历史视频逐字节一致 |
| Cosmos-Predict2.5 | 1500 | [播放／下载](predict25_clean.mp4) · [脚本](../../scripts_mytrain/inference/historical/predict25_clean.py) | [播放／下载](predict25_noisy.mp4) · [脚本](../../scripts_mytrain/inference/historical/predict25_noisy.py) | 本次复跑输出；与历史视频不逐字节一致 |

点击视频链接可查看 GitHub 视频文件页；若浏览器未提供播放控件，可通过 Raw／Download 下载播放。

Predict2.5 的输入及完整 context / timestep 审计与历史记录一致，但 clean 和 noisy 视频均有像素差异；原始未拆分脚本对照也有差异，原因尚未确定。这两个 demo 展示当前复跑输出，不表示严格复现已通过。

[复现命令与依赖](../../scripts_mytrain/inference/historical/README.md) · [详细验证记录](../../scripts_mytrain/inference/historical/verification.json) · [视频 SHA-256 与脚本对应表](manifest.json)
