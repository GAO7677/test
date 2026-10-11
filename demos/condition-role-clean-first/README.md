# 首步 clean · 全程条件角色：视频对照

10个视频对应实际完成的5份权重 × 2种推理方式。直接复制原始 MP4，未重新编码；文件 SHA256 与历史运行记录逐项一致。

| 权重 | 首步 clean → 重建 | 首步 clean → 自由演化 |
|---|---|---|
| wan A | [视频](wan_A_clean_rebuild_condition.mp4) | [视频](wan_A_clean_free_condition.mp4) |
| wan N75 | [视频](wan_N75_clean_rebuild_condition.mp4) | [视频](wan_N75_clean_free_condition.mp4) |
| nano A | [视频](nano_A_clean_rebuild_condition.mp4) | [视频](nano_A_clean_free_condition.mp4) |
| nano N75 | [视频](nano_N75_clean_rebuild_condition.mp4) | [视频](nano_N75_clean_free_condition.mp4) |
| predict25 N75 | [视频](predict25_N75_clean_rebuild_condition.mp4) | [视频](predict25_N75_clean_free_condition.mp4) |

重建：首步 context 为 c0，后续每步从 c0 加噪到 future 当前 sigma。自由演化：仅首步设置 c0，后续沿 scheduler 推进，不重建、不回填。两组始终保持对应 backbone 的条件角色及配套时间设置。

[脚本及复现说明](../../scripts_mytrain/inference/single_case/CONDITION_ROLE.md) · [视频校验清单](manifest.json) · [并排播放页面](comparison.html)

Wan A/N75 为 step1000；Nano A/N75、Predict2.5 N75 为 step1500。每段49帧、512×896。跨 backbone 的输入不完全相同，优先比较同一行。视频展示生成结果，不代表质量评测已通过。

可选：在仓库根目录以前台方式启动视频页面：

```bash
python3 -m http.server 8845 --bind 0.0.0.0 --directory demos/condition-role-clean-first
```

打开 http://localhost:8845/comparison.html 。本次上传未启动或修改本地服务。
