#!/bin/bash
# S1 主扫描：protocol × k × seed × domain
#
# 设计决定：
#  - 每个 seed 独立调用（--just_seed）→ 单 seed 故障不拖垮其余。
#  - 关闭 AnomalyDINO 自带评估（--no-eval_clf/--no-eval_segm）：其 eval_clf 路径
#    依赖 eval_segm 才写的 tiff，对自定义数据集不可用。异常分数的写入与该开关无关
#    （save_patch_dists 只影响它自己的评估），故指标改由 experiment/s1/analyze.py 计算
#    （已对 sklearn 交叉验证）。
#  - 完成判据 = CSV 行数达到 EXPECT_ROWS（1639 样本 + 1 表头）。
set -u
ROOT="${IADSHIFT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
AD="$ROOT/experiment/AnomalyDINO"
LOG="$ROOT/experiment/s1/logs"
EXPECT_ROWS=1640
mkdir -p "$LOG"

PROTO="$1"; GPU="$2"; PREP="${PREP:-agnostic}"; shift 2; KS="$@"

cd "$AD" || exit 1
for k in $KS; do
  nseed=$(( 64 / k )); [ "$nseed" -gt 8 ] && nseed=8
  for seed in $(seq 0 $((nseed-1))); do
    RES="results_AeBAD_${PROTO}/dinov2_vits14_448/${k}-shot_preprocess=${PREP}"
    CSV="$RES/measurements_seed=${seed}.csv"
    rows=$(wc -l < "$CSV" 2>/dev/null || echo 0)
    if [ "$rows" -ge "$EXPECT_ROWS" ]; then
      echo "[$PROTO/$PREP k=$k seed=$seed] 已完成 ($rows 行)，跳过"; continue
    fi
    echo "[$PROTO/$PREP k=$k seed=$seed] 启动 $(date +%H:%M:%S)"
    env -u http_proxy -u https_proxy -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY \
      python3 run_anomalydino.py \
        --dataset "AeBAD_$PROTO" \
        --data_root "$ROOT/data/AeBAD_S_domains/$PROTO" \
        --shots "$k" --just_seed "$seed" --preprocess "$PREP" \
        --no-save_examples --no-eval_clf --no-eval_segm \
        --faiss_on_cpu --device "$GPU" \
        >> "$LOG/${PROTO}_k${k}_${PREP}.log" 2>&1
    rc=$?
    rows=$(wc -l < "$CSV" 2>/dev/null || echo 0)
    if [ "$rows" -ge "$EXPECT_ROWS" ]; then
      echo "[$PROTO/$PREP k=$k seed=$seed] ✓ $rows 行"
    else
      echo "[$PROTO/$PREP k=$k seed=$seed] ⚠ rc=$rc 仅 $rows 行（期望 $EXPECT_ROWS）"
    fi
  done
done
echo "SWEEP_DONE proto=$PROTO ks=$KS"
