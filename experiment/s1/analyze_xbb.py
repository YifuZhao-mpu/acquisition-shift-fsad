#!/usr/bin/env python3
"""跨 backbone / 跨打分函数的 Finding-1 复现检验。

问题：「DINOv2 对全局色调近乎不变、对空间光照极度敏感」是否为
      DINOv2 特异 / AnomalyDINO 打分特异？

做法：对每个 (backbone, agg) 配置独立计算
      色调组（exposure/gamma/wb）与空间组（gradient/shadow/specular）的平均 AUROC 损失，
      并报告二者之比。若该不对称性跨配置一致 → 是表征的普遍性质而非某一模型的怪癖。
"""
import glob, json, sys
from collections import defaultdict
import numpy as np

TONAL = ["exposure", "gamma", "wb"]
SPATIAL = ["gradient", "shadow", "specular"]


def load(pat):
    out = defaultdict(list)
    for f in sorted(glob.glob(pat)):
        d = json.load(open(f))
        bb = d["meta"]["backbone"]; ag = d["meta"].get("agg", "meantop1p")
        for r in d["rows"]:
            out[(bb, ag)].append(r)
    return out


def main(pats):
    cfgs = {}
    for p in pats:
        for k, v in load(p).items(): cfgs.setdefault(k, []).extend(v)
    print("=" * 100)
    print("Finding 1 跨配置复现：色调扰动 vs 空间扰动的 AUROC 损失")
    print("=" * 100)
    print(f"{'backbone':18s} {'agg':10s} {'基线':>7s} | {'色调组':>8s} {'空间组':>8s} {'比值':>7s} | "
          f"{'gamma':>8s} {'specular':>9s}")
    print("-" * 100)
    rows = []
    for (bb, ag), rs in sorted(cfgs.items()):
        agg = defaultdict(list)
        for r in rs: agg[(r["mode"], r["severity"])].append(r["auroc"])
        base = np.mean(agg[("none", 0.0)])
        def drop(modes, sev):
            v = [np.mean(agg[(m, sev)]) for m in modes if (m, sev) in agg]
            return base - np.mean(v) if v else np.nan
        # 空间组用各自未饱和的强度；色调组用最高强度（最强也很弱）
        t = drop(TONAL, 1.0)
        s = np.nanmean([drop(["gradient"], 0.67), drop(["shadow"], 0.67), drop(["specular"], 0.33)])
        g = drop(["gamma"], 1.0); sp = drop(["specular"], 0.33)
        ratio = s / t if t and t > 0 else np.inf
        print(f"{bb:18s} {ag:10s} {base:7.4f} | {t:8.4f} {s:8.4f} {ratio:7.2f} | {g:8.4f} {sp:9.4f}")
        rows.append(dict(backbone=bb, agg=ag, base=float(base), tonal=float(t),
                         spatial=float(s), ratio=float(ratio), gamma=float(g), specular=float(sp)))
    print("\n数值为 AUROC 损失（正数 = 变差）。比值 = 空间组损失 / 色调组损失。")
    print("若所有配置的比值都远大于 1 → 该不对称性是普遍性质，非 DINOv2 或某打分函数特异。")
    return rows


if __name__ == "__main__":
    main(sys.argv[1:] or ["../../reports/mvtec_illum_g*.json", "../../reports/xbb_*.json"])
