# 离线模型部署说明（v3.0）

## 目录结构

```text
backend/models/
├── yolov8n.onnx           # 图像/视频 COCO 检测（已随仓库提供，12.8MB）
├── silero_vad.onnx        # 语音活动检测（需手动下载，约 1MB）
└── zh_core_web_sm/        # Spacy 中文 NER（需手动下载/解压，约 30MB）
    ├── meta.json
    ├── tokenizer/
    └── ...
```

> `yolov8n.onnx` 已包含在当前代码包中；VAD/NER 模型因体积与许可分发原因不随源码包强制携带，按下方说明放入后即可启用。未放置时，对应深度学习规则会显示“模型未就绪”，规则算法与已放置的模型不受影响。

## 1. Silero VAD

- 模型文件：`silero_vad.onnx`
- 放置位置：`backend/models/silero_vad.onnx`
- 下载地址：`https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx`
- 校验：文件大小约 1MB；加载时会校验 ONNX 输入/输出节点名必须为 `input` / `output`。

## 2. Spacy zh_core_web_sm

- 模型目录：`zh_core_web_sm/`
- 放置位置：`backend/models/zh_core_web_sm/`
- 下载地址：`https://github.com/explosion/spacy-models/releases/download/zh_core_web_sm-3.7.0/zh_core_web_sm-3.7.0-py3-none-any.whl`
- 安装/解压方式：
  ```bash
  # 方式一：在线环境 pip 安装后复制模型目录
  pip install zh_core_web_sm-3.7.0-py3-none-any.whl
  python -c "import spacy; print(spacy.util.get_package_path('zh_core_web_sm'))"
  # 将打印目录中的 zh_core_web_sm 复制到 backend/models/zh_core_web_sm/

  # 方式二：纯离线解压 whl
  unzip zh_core_web_sm-3.7.0-py3-none-any.whl -d spacy_model_tmp
  mv spacy_model_tmp/zh_core_web_sm-3.7.0/zh_core_web_sm backend/models/zh_core_web_sm
  ```
- 校验：目录内必须存在 `backend/models/zh_core_web_sm/meta.json`。

## 3. 自定义路径

如需将模型放在其他位置（例如独立数据盘），修改 `config.json`：

```json
{
  "models": {
    "vad_path": "D:\\Models\\silero_vad.onnx",
    "ner_path": "D:\\Models\\zh_core_web_sm"
  }
}
```

留空则使用默认 `backend/models/` 目录；PyInstaller 打包时会将 `backend/models` 一并收集，未放置的模型也可在打包后放到 exe 同级 `models/` 目录。

## 4. 就绪检查

启动服务后访问：

```text
GET http://localhost:8000/api/auto-label/rules
```

- `silero_vad` 规则：`model_ready=true` 表示 VAD 模型可用。
- `spacy_ner` 规则：`model_ready=true` 表示 NER 模型可用。
- 前端“规则自动标注”页会同步显示模型状态；模型未就绪时对应规则提交按钮禁用。
