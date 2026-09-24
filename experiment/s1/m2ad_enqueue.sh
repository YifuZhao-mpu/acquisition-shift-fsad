#!/usr/bin/env bash
# 监视 M2AD 下载：某类别的 zip 一完整就解压并把该类别的 5 个 backbone 任务投进队列。
# 与下载并行，不必等全部 153 GB 落地。
set -u
ROOT="${IADSHIFT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
D="$ROOT/data/M2AD"; Q="$ROOT/experiment/s1/queue_m2ad"
BASE="https://huggingface.co/datasets/ChengYuQi99/M2AD/resolve/main"
CATS="Bird Car Cube Dice Doll Holder Motor Ring Teapot Tube"
BBS="dinov2_vits14 deit_small_patch16 dino_resnet50 resnet50 wide_resnet50_2"
mkdir -p "$Q"/{todo,doing,done} "$D/images"

declare -A WANT
for c in $CATS; do
  WANT[$c]=$(curl -sIL -m 60 "$BASE/$c.zip" | grep -i '^x-linked-size' | tail -1 | tr -d '\r' | awk '{print $2}')
done

done_n=0
while [ "$done_n" -lt 10 ]; do
  done_n=0
  for c in $CATS; do
    if [ -f "$D/.extracted_$c" ]; then done_n=$((done_n+1)); continue; fi
    z="$D/zips/$c.zip"; [ -f "$z" ] || continue
    [ "$(stat -c%s "$z")" = "${WANT[$c]:-x}" ] || continue
    echo "$(date +%T) 解压 $c …"
    if unzip -o -q "$z" -d "$D/images"; then
      touch "$D/.extracted_$c"; rm -f "$z"        # 省 15 GB 磁盘
      for b in $BBS; do echo "$b $c" > "$Q/todo/${b}__${c}.task"; done
      echo "$(date +%T) $c 解压完成，已投 5 个任务"
      done_n=$((done_n+1))
    else
      echo "$(date +%T) $c 解压失败"
    fi
  done
  [ "$done_n" -lt 10 ] && sleep 120
done
echo "$(date +%T) 全部 10 类已解压入队"
