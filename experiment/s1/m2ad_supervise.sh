#!/usr/bin/env bash
# 下载守护：每 5 分钟检查一次，未下完就再拉起 fetch_m2ad.sh。
# fetch 自带 flock 单实例锁，所以重复拉起是安全的（不会两个实例写同一文件）。
# 存在的理由：2026-09-22 01:50–04:12 有一次静默停摆，白白损失 2.4 小时。
set -u
ROOT="${IADSHIFT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
D="$ROOT/data/M2AD"; LG="$D/log/supervise.log"
CATS="Bird Car Cube Dice Doll Holder Motor Ring Teapot Tube"
while true; do
  n=0; for c in $CATS; do [ -f "$D/.extracted_$c" ] && n=$((n+1)); done
  [ "$n" -ge 10 ] && { echo "$(date +%T) 10 类全部就位，守护退出" >>"$LG"; break; }
  if ! pgrep -f "[f]etch_m2ad.sh" >/dev/null; then
    echo "$(date +%T) fetch 未运行（已就位 $n/10），重新拉起" >>"$LG"
    ( cd "$ROOT" && setsid nohup bash experiment/s1/fetch_m2ad.sh >>"$D/log/fetch_all.log" 2>&1 < /dev/null & )
  fi
  if ! pgrep -f "[m]2ad_enqueue.sh" >/dev/null; then
    echo "$(date +%T) enqueue 未运行，重新拉起" >>"$LG"
    ( cd "$ROOT" && setsid nohup bash experiment/s1/m2ad_enqueue.sh >>"$D/log/enqueue.log" 2>&1 < /dev/null & )
  fi
  sleep 300
done
