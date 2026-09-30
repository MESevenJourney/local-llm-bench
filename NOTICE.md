# NOTICE

本仓库（local-llm-bench）以 MIT 许可发布（见 [LICENSE](LICENSE)）。它是一个基准测试
工具，本身不捆绑、不分发以下任何第三方软件或模型权重；运行基准需要你自行获取并遵守
各自的许可条款。

## 引擎

- **TensorFold** — <https://github.com/ashhart/TensorFold>。本仓库仅在
  `requirements/tensorfold-v*.txt` 中以「git URL + 40-hex commit」形式钉住其版本，
  未包含其源代码。请遵守其上游许可。
- **oMLX** — <https://github.com/jundot/omlx>。以 macOS App bundle 形式分发；
  `requirements/omlx-v070rc1.json` 只记录版本元数据。请遵守其上游许可。

## 运行时依赖

- **MLX / mlx-lm / mlx-metal**（Apple）— TensorFold 与 oMLX 栈的底层阵列框架与
  语言建模工具。各 requirements 快照钉住了实测时的具体版本；请遵守其各自的开源许可。

## 模型权重（不在本仓库内）

- `lmstudio-community/Qwen3.8-27B-MLX-4bit` — Qwen 系列模型的社区量化版。
  Qwen 模型权重受其自身的模型许可条款约束；商用或再分发前请自行确认。
- `z-lab/Qwen3.8-27B-DFlash2` — DFlash2 草稿模型权重，同样受其发布许可约束。

本仓库的 profile 只引用上述 repo id 与本地路径的**环境变量名**；下载与使用权重由
使用者自行决定并自行承担许可责任（`run --download-models` 只在你显式要求时调用
huggingface_hub）。

## 基准结果

本仓库产出的数字为特定机器、版本与场景下的实测，引用时请附带样本数与 README 中的
口径与限制说明；这些数字是对上述第三方引擎的测量描述，不代表任何官方性能承诺。
