#!/bin/bash
# k=8 阶段专用：接受显式 seed 列表，使多块 GPU 可分担同一 (protocol,k) 的不同 seed。
# 各 GPU 的 seed 集合互不相交 —— 避免两个进程写同一个 CSV。
set -u
ROOT="${IADSHIFT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
AD="$ROOT/experiment/AnomalyDINO"; LOG="$ROOT/experiment/s1/logs"
PREP=force_mask_rotation; EXPECT_ROWS=1640
PROTO="$1"; GPU="$2"; shift 2; SEEDS="$@"
cd "$AD" || exit 1
for seed in $SEEDS; do
  RES="results_AeBAD_${PROTO}/dinov2_vits14_448/8-shot_preprocess=${PREP}"
  CSV="$RES/measurements_seed=${seed}.csv"
  rows=$(wc -l < "$CSV" 2>/dev/null || echo 0)
  if [ "$rows" -ge "$EXPECT_ROWS" ]; then echo "[$PROTO k=8 seed=$seed] 已完成，跳过"; continue; fi
  echo "[$PROTO k=8 seed=$seed] 启动 $(date +%H:%M:%S) on $GPU"
  env -u http_proxy -u https_proxy -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    python3 run_anomalydino.py --dataset "AeBAD_$PROTO" \
      --data_root "$ROOT/data/AeBAD_S_domains/$PROTO" \
      --shots 8 --just_seed "$seed" --preprocess "$PREP" \
      --no-save_examples --no-eval_clf --no-eval_segm --faiss_on_cpu --device "$GPU" \
      >> "$LOG/${PROTO}_k8_${PREP}.log" 2>&1
  rows=$(wc -l < "$CSV" 2>/dev/null || echo 0)
  [ "$rows" -ge "$EXPECT_ROWS" ] && echo "[$PROTO k=8 seed=$seed] ✓" || echo "[$PROTO k=8 seed=$seed] ⚠ 仅 $rows 行"
done
echo "K8_LANE_DONE $PROTO $GPU seeds=$SEEDS"
