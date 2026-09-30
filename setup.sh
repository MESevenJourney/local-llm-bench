#!/bin/bash
# 一次性环境安装：创建 .venv 并按钉死的依赖快照安装 TensorFold。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
python3 -m venv .venv
.venv/bin/pip install -r requirements/tensorfold-v0361.txt
echo "完成：.venv/bin/tensorfold 已就绪"
