#!/bin/bash
# 四路并行驱动器，脱离 harness 后台任务监管运行。
# 依据：实测单进程 RSS = 1.48 GB，四进程共 ~6 GB / 系统 188 GB（可用 174 GB）。
# 此前的 OOM 杀进程源于 `free` 被 172 GB 可回收 page cache 占满导致的误判。
S1="$ROOT/experiment/s1"
bash "$S1/run_sweep.sh" single cuda:0 1 2 > "$S1/logs/drive_single_12.out" 2>&1 &
bash "$S1/run_sweep.sh" single cuda:1 4 8 > "$S1/logs/drive_single_48.out" 2>&1 &
bash "$S1/run_sweep.sh" mixed  cuda:2 1 2 > "$S1/logs/drive_mixed_12.out"  2>&1 &
bash "$S1/run_sweep.sh" mixed  cuda:3 4 8 > "$S1/logs/drive_mixed_48.out"  2>&1 &
wait
echo "ALL_SWEEPS_DONE $(date +%H:%M:%S)" > "$S1/logs/ALL_DONE"
