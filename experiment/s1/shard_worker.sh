#!/usr/bin/env bash
# 按 (backbone, category) 分片的工作队列 —— 任一空闲 GPU 都能接活。
# 用法: shard_worker.sh <gpu_id>
set -u
GPU=$1
ROOT="${IADSHIFT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
Q="$ROOT/experiment/s1/queue"; L="$ROOT/experiment/s1/logs/r2"
cd "$ROOT/experiment/s1" || exit 1
while true; do
  T=""
  for f in "$Q"/todo/*.task; do
    [ -e "$f" ] || break
    b=$(basename "$f")
    if mv "$f" "$Q/doing/$b" 2>/dev/null; then T="$Q/doing/$b"; break; fi
  done
  if [ -z "$T" ]; then echo "[gpu$GPU] queue empty, exiting $(date +%H:%M:%S)"; break; fi
  read -r BB CAT < "$T"
  echo "[gpu$GPU] START $BB/$CAT $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES="$GPU" ALL_PROXY= all_proxy= PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    python3 mvtec_illum.py --backbone "$BB" --agg meantop1p --categories "$CAT" \
      --out "$ROOT/reports/r2/shards/${BB}_${CAT}.json" \
      --dump-scores "$ROOT/reports/r2/shards/scores_${BB}_${CAT}.npz" \
      > "$L/shard_${BB}_${CAT}.log" 2>&1
  RC=$?
  echo "[gpu$GPU] END   $BB/$CAT rc=$RC $(date +%H:%M:%S)"
  if [ "$RC" -eq 0 ]; then mv "$T" "$Q/done/$(basename "$T")"; else mv "$T" "$Q/todo/$(basename "$T")"; sleep 10; fi
done
