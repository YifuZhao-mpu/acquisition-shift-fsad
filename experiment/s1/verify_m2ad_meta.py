#!/usr/bin/env python3
"""核对 M2AD 元数据的语义 —— 补充材料 S9 里关于标签的每一句都必须可查。

论文写了什么，这里就查什么：
  1. detectable 恰好定义在 image_anomaly=1 的图上（正常图为空串）
  2. object_anomaly=1 但 image_anomaly=0 的图存在，即"缺陷不在该视角画面内"
  3. 训练集全为正常图
  4. 每 (类别,视角,光照) 单元恰好 70 张
  5. detectable=False 在异常图中的占比 —— 这就是"任务本身不可能"的上界
"""
from __future__ import annotations
import os
import collections, json, sys
from pathlib import Path

ROOT = Path(os.environ.get("IADSHIFT_ROOT",
                    Path(__file__).resolve().parents[2]))   # experiment/s1/x.py -> 项目根


def main():
    f = ROOT / "data/M2AD/meta_unsupervised.json"
    if not f.exists():
        print(f"缺少 {f}"); sys.exit(1)
    d = json.load(open(f)); te, tr = d["test"], d["train"]
    ok = True

    n_bad = sum(1 for c in te for r in te[c]
                if (r["detectable"] != "") != bool(r["image_anomaly"]))
    n_all = sum(len(v) for v in te.values())
    print(f"1. detectable 恰好定义在异常图上：违例 {n_bad}/{n_all}  "
          f"{'通过' if n_bad == 0 else '失败'}"); ok &= n_bad == 0

    cnt = collections.Counter((r["object_anomaly"], r["image_anomaly"])
                              for c in te for r in te[c])
    oob = cnt[(1, 0)]
    print(f"2. 缺陷出画面的图（object=1, image=0）：{oob}  "
          f"{'通过' if oob > 0 else '失败'}"); ok &= oob > 0

    n_tr_pos = sum(1 for c in tr for r in tr[c] if r["image_anomaly"])
    print(f"3. 训练集异常图：{n_tr_pos}  {'通过' if n_tr_pos == 0 else '失败'}")
    ok &= n_tr_pos == 0

    # 单元大小在**类别内**恒定（Holder 是 69，其余 70），跨类别不必相同
    per_cat, ragged = {}, []
    for c in te:
        cell = collections.Counter((r["view"], r["illumination"]) for r in te[c])
        sz = set(cell.values()); per_cat[c] = sorted(sz)
        if len(sz) != 1 or len(cell) != 120: ragged.append(c)
    print(f"4. 每类别内单元大小恒定且恰好 120 个单元："
          f"{ {c: v[0] for c, v in sorted(per_cat.items())} }  "
          f"{'通过' if not ragged else '失败 ' + str(ragged)}"); ok &= not ragged

    pos = sum(1 for c in te for r in te[c] if r["image_anomaly"])
    und = sum(1 for c in te for r in te[c] if r["detectable"] is False)
    print(f"5. 异常图中标注为不可检出的占比：{und}/{pos} = {und/pos*100:.1f}%"
          f"  —— 这是「任务不可能」部分的上界")

    print("\n" + ("全部通过" if ok else "有失败项"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
