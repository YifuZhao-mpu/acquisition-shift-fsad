#!/usr/bin/env bash
# 下载 M2AD（10 类，约 153 GB，Apache-2.0）。可重复运行：已完整的类会跳过，未完成的断点续传。
set -u
DST="$(dirname "$0")/../../data/M2AD"
BASE="https://huggingface.co/datasets/ChengYuQi99/M2AD/resolve/main"
PAR=${PAR:-2}                       # 并发类数（2 个流即可跑满带宽，且让首个类别早完成）
mkdir -p "$DST/zips" "$DST/log"

# 单实例锁：两个实例同时对同一文件 --continue-at 续传会把 zip 写坏（已踩过）
exec 9>"$DST/.fetch.lock"
flock -n 9 || { echo "已有 fetch_m2ad.sh 在运行，退出"; exit 0; }

fetch(){
  local c="$1"
  local z="$DST/zips/$c.zip"        # 注意：不能与上一行合并成一条 local，
  local lg="$DST/log/$c.log"        # 同一条 local 里 $c 尚未赋值就被展开
  local want
  want=$(curl -sIL -m 60 "$BASE/$c.zip" | grep -i '^x-linked-size' | tail -1 | tr -d '\r' | awk '{print $2}')
  [ -z "$want" ] && { echo "$c: 拿不到大小，跳过" >>"$lg"; return 1; }
  for try in 1 2 3 4 5 6 7 8 9 10; do
    local have=0; [ -f "$z" ] && have=$(stat -c%s "$z")
    if [ "$have" -eq "$want" ]; then echo "$c: 完整 $want" >>"$lg"; return 0; fi
    echo "$(date +%T) $c: 第 $try 次，已有 $have / $want" >>"$lg"
    curl -sL --continue-at - --retry 5 --retry-delay 5 --speed-limit 10240 --speed-time 120 \
         "$BASE/$c.zip" -o "$z" >>"$lg" 2>&1
  done
  local have=0; [ -f "$z" ] && have=$(stat -c%s "$z")
  [ "$have" -eq "$want" ] && return 0 || { echo "$c: 失败 $have/$want" >>"$lg"; return 1; }
}
export -f fetch; export DST BASE

CATS="Bird Car Cube Dice Doll Holder Motor Ring Teapot Tube"
printf '%s\n' $CATS | xargs -P "$PAR" -I{} bash -c 'fetch {} && echo "DONE {}" || echo "FAIL {}"'
echo "=== 汇总 ==="
for c in $CATS; do
  z="$DST/zips/$c.zip"; [ -f "$z" ] && printf "  %-8s %8.2f GB\n" "$c" "$(echo "$(stat -c%s "$z")/1073741824"|bc -l)" || echo "  $c 缺失"
done
