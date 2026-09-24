#!/usr/bin/env bash
# M2AD 队列工作进程。用法: m2ad_worker.sh <gpu_id>
# 与 shard_worker.sh 的区别：队列是边下载边填的，空队列要**等**而不是退出。
set -u
GPU=$1
ROOT="${IADSHIFT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
Q="$ROOT/experiment/s1/queue_m2ad"; L="$ROOT/experiment/s1/logs/m2ad"
VIEWS="000 090 180 270"      # 预注册修订 A 锁定
mkdir -p "$L" "$ROOT/reports/m2ad"
cd "$ROOT/experiment/s1" || exit 1
IDLE=0
while true; do
  T=""
  for f in "$Q"/todo/*.task; do
    [ -e "$f" ] || break
    b=$(basename "$f")
    if mv "$f" "$Q/doing/$b" 2>/dev/null; then T="$Q/doing/$b"; break; fi
  done
  if [ -z "$T" ]; then
    IDLE=$((IDLE+1))
    [ -f "$Q/.closed" ] && { echo "[gpu$GPU] 队列已关闭且为空，退出"; break; }
    [ "$IDLE" -gt 180 ] && { echo "[gpu$GPU] 空闲超时，退出"; break; }
    sleep 60; continue
  fi
  IDLE=0
  read -r BB CAT AGG < "$T"
  AGG=${AGG:-meantop1p}
  echo "[gpu$GPU] START $BB/$CAT/$AGG $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES="$GPU" ALL_PROXY= all_proxy= PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    python3 m2ad_shift.py --backbone "$BB" --agg "$AGG" --categories "$CAT" \
      --views $VIEWS --k 4 --seeds 4 \
      --out "$ROOT/reports/m2ad/${BB}_${CAT}_${AGG}.json" \
      --dump-scores "$ROOT/reports/m2ad/scores_${BB}_${CAT}_${AGG}.npz" \
      > "$L/${BB}_${CAT}_${AGG}.log" 2>&1
  RC=$?
  echo "[gpu$GPU] END   $BB/$CAT/$AGG rc=$RC $(date +%H:%M:%S)"
  if [ "$RC" -eq 0 ]; then mv "$T" "$Q/done/$(basename "$T")"
  else mv "$T" "$Q/todo/$(basename "$T")"; sleep 30; fi
done
