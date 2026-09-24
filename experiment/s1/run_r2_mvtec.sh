#!/usr/bin/env bash
# Round-2 审查 R2：MVTec 受控扰动网格，2x2 表征识别设计 + 层深对照
#   用法: run_r2_mvtec.sh <gpu> <backbone> <agg>
set -u
GPU=$1; BB=$2; AGG=${3:-meantop1p}
ROOT="${IADSHIFT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT/experiment/s1"
TAG="${BB}_${AGG}"
exec env CUDA_VISIBLE_DEVICES=$GPU ALL_PROXY= all_proxy= PYTHONUNBUFFERED=1 \
  python3 mvtec_illum.py --backbone "$BB" --agg "$AGG" \
    --out "$ROOT/reports/r2/xbb2_${TAG}.json" \
    --dump-scores "$ROOT/reports/r2/scores_${TAG}.npz"
