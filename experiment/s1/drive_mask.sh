#!/bin/bash
# 遮罩消融：force_mask_rotation（mask=True, rotation=True）
# 与主扫描的 agnostic（mask=False, rotation=True）仅差遮罩一项 —— 单因子对照。
S1="$ROOT/experiment/s1"
export PREP=force_mask_rotation
bash "$S1/run_sweep.sh" single cuda:0 1 2 > "$S1/logs/dm_single_12.out" 2>&1 &
bash "$S1/run_sweep.sh" single cuda:1 4 8 > "$S1/logs/dm_single_48.out" 2>&1 &
bash "$S1/run_sweep.sh" mixed  cuda:2 1 2 > "$S1/logs/dm_mixed_12.out"  2>&1 &
bash "$S1/run_sweep.sh" mixed  cuda:3 4 8 > "$S1/logs/dm_mixed_48.out"  2>&1 &
wait
echo "MASK_ABLATION_DONE $(date +%H:%M:%S)" > "$S1/logs/MASK_DONE"
