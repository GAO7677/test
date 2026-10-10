# Historical Context Noise Inference

Wan、Cosmos3-Nano、Cosmos-Predict2.5 的历史 clean / context 加噪推理代码，共六个入口。

[完整使用说明与复现命令](scripts_mytrain/inference/historical/README.md) · [六个推理 Demo 视频](demos/gap038/README.md)

| Backbone | Clean | Context noise |
|---|---|---|
| Wan | [wan_clean.py](scripts_mytrain/inference/historical/wan_clean.py) | [wan_noisy.py](scripts_mytrain/inference/historical/wan_noisy.py) |
| Cosmos3-Nano | [nano_clean.py](scripts_mytrain/inference/historical/nano_clean.py) | [nano_noisy.py](scripts_mytrain/inference/historical/nano_noisy.py) |
| Cosmos-Predict2.5 | [predict25_clean.py](scripts_mytrain/inference/historical/predict25_clean.py) | [predict25_noisy.py](scripts_mytrain/inference/historical/predict25_noisy.py) |

本仓库包含推理源码、冻结辅助代码、配置、验证记录及六个推理 demo 视频，不包含权重、完整数据集或完整 backbone 工程。复现依赖原模型工程、Python 环境及外部资产；配置保留原实验的绝对路径，需要在具有这些依赖的环境运行。六个入口共享辅助代码，不能单独复制一个文件运行。

验证状态：Wan、Nano 的 clean / noisy 各一例共四个视频逐字节复现；Predict2.5 输入和完整 context / timestep 审计一致，但两个视频尚未逐字节复现，原始未拆分脚本对照也存在差异。详见 [验证记录](scripts_mytrain/inference/historical/verification.json)。

不包含后续 RePaint、回跳或提前 n 步变体。保留源文件中的既有版权与许可声明；上游模型和依赖遵循各自的许可。
