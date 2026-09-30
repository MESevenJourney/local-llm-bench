#!/bin/bash
# oMLX 基准：隔离启动（不碰日常 App 配置）→ 固定历史重放 × REPEATS → 停止。
# 用法：./run-omlx.sh [draft|serial]   （默认 draft）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# ================= 配置区（有变动改这里）=================
MODEL_DIR="$HOME/.omlx/models/lmstudio-community/Qwen3.8-27B-MLX-4bit"   # 【模型路径】
DRAFTER_DIR="$HOME/.omlx/models/z-lab/Qwen3.8-27B-DFlash2"               # 【草稿路径】
OMLX_CLI="/Applications/oMLX.app/Contents/MacOS/omlx-cli"                # 【oMLX 安装位置】
CEILING_GB=28           # 内存护栏上限 GiB
PORT=8002
REPEATS=3
TOKENS=2048
SEED=1234
# ==========================================================

MODE="${1:-draft}"
[[ "$MODE" == draft || "$MODE" == serial ]] || { echo "用法：$0 [draft|serial]" >&2; exit 2; }
[[ -x "$OMLX_CLI" ]] || { echo "找不到 oMLX CLI：$OMLX_CLI（改 run-omlx.sh 顶部 OMLX_CLI）" >&2; exit 1; }

export MODEL_DIR DRAFTER_DIR
export OMLX_BENCH_CEILING_GB="$CEILING_GB" OMLX_BENCH_PORT="$PORT" OMLX_CLI_OVERRIDE="$OMLX_CLI"

echo "隔离启动 oMLX（${MODE}，ceiling ${CEILING_GB}GB）——不修改 ~/.omlx 日常配置"
python3 "$ROOT/omlx_bench.py" start "$MODE"
for i in $(seq 1 "$REPEATS"); do
  echo "=== $MODE 第 $i/$REPEATS 次 ==="
  python3 "$ROOT/omlx_bench.py" test --scenario coding --tokens "$TOKENS" --seed "$SEED" \
    --replay-from "$ROOT/replay-qwen38-coding.json" --label "omlx-$MODE-r$i"
done
python3 "$ROOT/omlx_bench.py" stop
echo "完成。报告在 logs/ 下最新的 scenarios-* 目录"
