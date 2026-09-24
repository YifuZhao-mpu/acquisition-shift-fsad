#!/bin/bash
# 方案 B 接力调度：
#  阶段1  等 k=4 两协议各 8/8 完成 → 立即终止 GPU1/GPU3 的 run_sweep（阻止其串行开跑 k=8）
#  阶段2  等 k=1,2 两协议各 8/8 完成（GPU0/GPU2 收工）
#  阶段3  把 k=8 的 16 个 seed（2 协议 × 8）拆到 4 块 GPU，seed 集合互不相交
S1="$ROOT/experiment/s1"
AD="$ROOT/experiment/AnomalyDINO"
PREP=force_mask_rotation

done_count() {  # $1=proto $2=k  -> 完整 CSV 数
  local p="$AD/results_AeBAD_$1/dinov2_vits14_448/$2-shot_preprocess=$PREP" n=0 f
  for f in "$p"/measurements_seed=*.csv; do
    [ -f "$f" ] && [ "$(wc -l < "$f")" -ge 1640 ] && n=$((n+1))
  done; echo $n
}

echo "[$(date +%H:%M:%S)] 阶段1：等待 k=4 完成"
while [ "$(done_count single 4)" -lt 8 ] || [ "$(done_count mixed 4)" -lt 8 ]; do sleep 60; done
echo "[$(date +%H:%M:%S)] k=4 已 8/8 —— 终止 GPU1/GPU3 的 run_sweep，阻止其串行开跑 k=8"
for pat in "run_sweep.sh single cuda:1" "run_sweep.sh mixed cuda:3"; do
  for pid in $(pgrep -f "$pat"); do
    pkill -P "$pid" 2>/dev/null      # 先杀其 python 子进程
    kill "$pid" 2>/dev/null
  done
done
sleep 5
pkill -f "run_anomalydino.py --dataset AeBAD_single --data_root.*--shots 8" 2>/dev/null
pkill -f "run_anomalydino.py --dataset AeBAD_mixed --data_root.*--shots 8" 2>/dev/null

echo "[$(date +%H:%M:%S)] 阶段2：等待 k=1,2 完成（GPU0/GPU2）"
while [ "$(done_count single 1)" -lt 8 ] || [ "$(done_count mixed 1)" -lt 8 ] \
   || [ "$(done_count single 2)" -lt 8 ] || [ "$(done_count mixed 2)" -lt 8 ]; do sleep 60; done

echo "[$(date +%H:%M:%S)] 阶段3：k=8 四路并行，seed 互不相交"
bash "$S1/run_k8.sh" single cuda:0 0 1 2 3 > "$S1/logs/k8_s_a.out" 2>&1 &
bash "$S1/run_k8.sh" single cuda:1 4 5 6 7 > "$S1/logs/k8_s_b.out" 2>&1 &
bash "$S1/run_k8.sh" mixed  cuda:2 0 1 2 3 > "$S1/logs/k8_m_a.out" 2>&1 &
bash "$S1/run_k8.sh" mixed  cuda:3 4 5 6 7 > "$S1/logs/k8_m_b.out" 2>&1 &
wait
echo "MASK_ABLATION_DONE $(date +%H:%M:%S)" > "$S1/logs/MASK_DONE"
echo "[$(date +%H:%M:%S)] 全部完成"
