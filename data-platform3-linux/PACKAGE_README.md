# GitHub 源码交付说明

本分支 `data-platform-linux-20261014` 在原仓库内容旁新增 `data-platform3-linux/`。包括当前数据中台源码、前端构建、文档、迁移/检查脚本；按用户选择不上传模型权重，也不上传标注、运行数据库或真实凭据。

## 使用前

1. 从项目根目录复制 `config.json.example` 为 `config.json`，填写 MySQL 密码并调整路径。示例保留118完整功能配置结构，路径需要按目标机器调整；JWT占位值将在首次启动时自动生成随机密钥。不要提交真实 `config.json`。
2. 按 `LINUX.md` 准备 `backend/.venv` 和 MySQL；前端 `frontend/dist` 已包含。源码修改后需要安装前端依赖并重新构建。
3. 项目原有 YOLO/VAD/Whisper/GLiNER 权重保留在已交付的 `data-platform3-linux_20261014_no_annotations.zip` 中。将 ZIP 内的 `models/` 和 `backend/models/` 恢复到本项目对应目录。原交付 ZIP 位于交付机器 `/data/gaoya/agent-data/outputs_v1/data_platform3_linux/20261014_project_zip/`，不作为本仓库可直接下载的附件。
4. 少样本 FLUX/SD 与视频 SAM/CoTracker 环境、权重及素材数据集仍在项目外，需要按配置另外准备；原 ZIP 也不包含这些外部资源。
5. 在项目根目录先执行 `bash start.sh --check`；检查路径/依赖通过后用 `bash start.sh` 前台启动，使用目标机器实际局域网 IP 访问8000端口。118专用 MySQL 启动脚本并不替代安装数据库。
6. Windows入口继承自原项目；新功能尚未完成 Windows 一键转换和部署验收。

完整功能及限制见 `CHANGES_FROM_WINDOWS_TAR.md`，后续记录追加到 `CHANGELOG_LINUX.md`。该对比说明中的文件统计针对运行副本，不等于本分支的交付文件数。`PACKAGE_MANIFEST.json` 记录本次实际提交文件和未提交模型资源的哈希。

本次仅发布源码，不启动模型、迁移生产数据库或重启服务。回归脚本需使用隔离测试配置，不能直接在正式库执行。
