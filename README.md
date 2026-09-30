# local-llm-bench

本地 LLM 推理基准：同一份 Qwen3.8-27B + DFlash2 草稿，在 TensorFold 和 oMLX 上
跑同样的八轮 coding 测试，比较解码速度和投机解码收益。

## 需要修改的地方（只有这些）

| 位置 | 内容 |
|---|---|
| `run-tensorfold.sh` 顶部 | `MODEL_DIR` / `DRAFTER_DIR` 模型路径 |
| `run-omlx.sh` 顶部 | `MODEL_DIR` / `DRAFTER_DIR` / `OMLX_CLI`（oMLX 安装位置）|
| 其余参数 | 两个脚本顶部配置区：端口、内存预算、重复次数等，有注释 |

## 跑法

```bash
# 一次性：装 TensorFold 环境（依赖快照钉死版本）
./setup.sh

# TensorFold，草稿开/关各跑 3 次
./run-tensorfold.sh draft
./run-tensorfold.sh serial

# oMLX（需本机已装 oMLX App；隔离启动，不动日常配置）
./run-omlx.sh draft
./run-omlx.sh serial
```

报告在 `logs/scenarios-*/report.md`，逐轮数据在同名目录的 `results.json` / `turns.csv`。

## 文件说明

```
setup.sh                  # 建 .venv，按 requirements/tensorfold-v0361.txt 钉死安装
run-tensorfold.sh         # 起服务 → 重放 ×3 → 停（含草稿挂载确认，失败即停）
run-omlx.sh               # 隔离起 oMLX → 重放 ×3 → 停
bench.py                  # 流式测量客户端：固定历史重放、逐轮计时、写报告
omlx_bench.py             # oMLX 隔离启动/停止/测试（独立配置目录+一次性密钥）
scenarios.json            # 测试提示词（合成的 Python 缓存排错任务等）
replay-qwen38-coding.json # 冻结的八轮历史：两个引擎收到完全相同的输入
requirements/             # TensorFold 依赖快照（40-hex commit 钉死）
```

## 口径

解码速度 = `(completion_tokens - 1) / (最后一个非空生成片段 − 第一个非空生成片段)`，
客户端流式近似，含思考。跨引擎比较用同一冻结历史（`replay-qwen38-coding.json`），
谁的回答长短都不影响下一轮输入。TensorFold 服务端字段与客户端口径实测比值 1.001，
可等同看待。多次重复注明样本数与离散度；这是产品栈对比，不是受控内核比较。

草稿模式有防伪：TensorFold 靠启动日志确认 drafter 挂载（确认不了直接停），
oMLX 靠隔离服务日志里的挂载记录，确认不了同样拒绝测速。

## 许可

代码 MIT（[LICENSE](LICENSE)）。TensorFold / oMLX / MLX 及模型权重归各自上游，
见 [NOTICE.md](NOTICE.md)；模型路径只写在脚本配置区，仓库不含权重、不会自动下载。

