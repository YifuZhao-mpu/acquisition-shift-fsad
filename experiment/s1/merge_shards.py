#!/usr/bin/env python3
"""把 (backbone, category) 分片合并成整配置的结果文件。

分片是为了让任一空闲 GPU 都能接活；合并后的文件与整包跑出来的**逐行等价**
（同一 backbone/agg/category 的行由同一段代码产生，分片只改变调度）。
只有 15 个类别齐了才写出整配置文件，避免下游分析拿到残缺网格。
"""
from __future__ import annotations
import os
import argparse, glob, json
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("IADSHIFT_ROOT",
                    Path(__file__).resolve().parents[2]))   # experiment/s1/x.py -> 项目根
SH = ROOT / "reports/r2/shards"
N_CATS = 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true",
                    help="类别不齐也合并（文件标记 partial=True，配对分析会自动取单元交集）")
    ap.add_argument("--min-cats", type=int, default=4, help="--partial 时的最少类别数")
    a = ap.parse_args()
    groups = {}
    for f in sorted(SH.glob("*.json")):
        d = json.load(open(f)); m = d["meta"]
        groups.setdefault((m["backbone"], m["agg"]), []).append((f, d))
    for (bb, agg), items in sorted(groups.items()):
        cats = sorted({c for _, d in items for c in d["meta"]["categories"]})
        tag = f"{bb}/{agg}"
        partial = len(cats) < N_CATS
        if partial and not (a.partial and len(cats) >= a.min_cats):
            print(f"  {tag:34} {len(cats):2}/{N_CATS} categories — waiting")
            continue
        rows = [r for _, d in items for r in d["rows"]]
        meta = dict(items[0][1]["meta"]); meta["categories"] = cats
        meta["sharded"] = True; meta["n_shards"] = len(items); meta["partial"] = partial
        out = ROOT / f"reports/r2/xbb2_{bb}_{agg}.json"
        json.dump(dict(meta=meta, rows=rows), open(out, "w"), indent=1)
        units = {(r["category"], r["defect"]) for r in rows}
        print(f"  {tag:34} {'PARTIAL' if partial else 'MERGED '} {len(cats):2}/{N_CATS} cats, "
              f"{len(rows)} rows, {len(units)} units -> {out.name}")

        # 逐图分数：合并成单个 npz（键已带类别前缀，天然不冲突）
        z = {}
        ok = True
        for f, _ in items:
            sp = f.with_name("scores_" + f.name.replace(".json", ".npz"))
            if not sp.exists(): ok = False; break
            with np.load(sp, allow_pickle=True) as zz:
                for k in zz.files:
                    if k != "__meta__": z[k] = zz[k]
        if ok and z:
            z["__meta__"] = np.array([json.dumps(meta)])
            sout = ROOT / f"reports/r2/scores_{bb}_{agg}.npz"
            np.savez_compressed(sout, **z)
            print(f"  {'':34} scores -> {sout.name} ({len([k for k in z if k.endswith('/scores')])} categories)")


if __name__ == "__main__":
    main()
