#!/bin/bash
# TensorFold 基准：启动服务 → 固定历史重放 × REPEATS → 停止服务。
# 用法：./run-tensorfold.sh [draft|serial]   （默认 draft）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# ================= 配置区（有变动改这里）=================
MODEL_DIR="$HOME/.omlx/models/lmstudio-community/Qwen3.8-27B-MLX-4bit"   # 【模型路径】
DRAFTER_DIR="$HOME/.omlx/models/z-lab/Qwen3.8-27B-DFlash2"               # 【草稿路径】
PORT=8081
MEMORY_GB=25.2          # 内存预算 GiB
CONTEXT=32768
REPEATS=3
TOKENS=2048             # 每轮输出上限
SEED=1234
# ==========================================================

MODE="${1:-draft}"
[[ "$MODE" == draft || "$MODE" == serial ]] || { echo "用法：$0 [draft|serial]" >&2; exit 2; }
VENV="$ROOT/.venv"
[[ -x "$VENV/bin/tensorfold" ]] || { echo "先跑 ./setup.sh" >&2; exit 1; }
for d in "$MODEL_DIR" "$DRAFTER_DIR"; do
  [[ -f "$d/config.json" ]] || { echo "模型不存在：$d（改 run-tensorfold.sh 顶部 MODEL_DIR/DRAFTER_DIR）" >&2; exit 1; }
done

LOG="$ROOT/logs/tensorfold-$MODE.log"; mkdir -p "$ROOT/logs"
echo "启动 TensorFold（${MODE}），日志 $LOG"
TENSORFOLD_MEMORY_LIMIT_GB="$MEMORY_GB" "$VENV/bin/tensorfold" serve "$MODEL_DIR" \
  --drafter "$DRAFTER_DIR" --name qwen38-local --host 127.0.0.1 --port "$PORT" \
  --context "$CONTEXT" --max-tokens 4096 --prompt-cache-gib 2 --mlx-cache-gib 2 \
  --parallel 1 --snapshot-dir none --no-update-check >"$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT

for i in $(seq 1 180); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "服务未就绪，看 $LOG"; exit 1; }
grep -q 'drafter' "$LOG" && grep -q 'drafts: on' "$LOG" || { echo "草稿未确认挂载，看 $LOG"; exit 1; }
echo "服务就绪，跑 $REPEATS 次固定历史重放"

BENCH_ARGS=(--engine tensorfold --scenario coding --tokens "$TOKENS" --seed "$SEED"
  --port "$PORT" --model qwen38-local --replay-from "$ROOT/replay-qwen38-coding.json")
[[ "$MODE" == serial ]] && BENCH_ARGS+=(--no-drafts)
for i in $(seq 1 "$REPEATS"); do
  echo "=== $MODE 第 $i/$REPEATS 次 ==="
  "$VENV/bin/python" "$ROOT/bench.py" "${BENCH_ARGS[@]}" --label "tf-$MODE-r$i"
done

kill $SRV; wait $SRV 2>/dev/null || true; trap - EXIT
echo "完成。报告在 logs/ 下最新的 scenarios-* 目录"
