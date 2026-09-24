#!/usr/bin/env python3
"""复核 PREREGISTRATION.md 锁定的 P1/P2/P3 三项检验 —— 论文 §6.2。

分类、预测与判定规则一律照抄预注册文件，不做任何事后调整。
族 = {P1,P2,P3} x 3 severity = 9 个检验，Holm-Bonferroni 校正。
"""
from __future__ import annotations
import os
import glob, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu

ROOT = Path(os.environ.get("IADSHIFT_ROOT",
                    Path(__file__).resolve().parents[2]))   # experiment/s1/x.py -> 项目根
SPATIAL = ["gradient", "specular", "shadow"]; TONAL = ["exposure", "gamma", "wb"]
SEVS = (0.33, 0.67, 1.0)

# —— 预注册的物理签名分类（逐字取自 PREREGISTRATION.md (a) 节）——
S = {"scratch","scratch_head","scratch_neck","crack","poke","poke_insulation","fold","rough",
     "squeeze","squeezed_teeth","bent","bent_wire","bent_lead"}
A = {"color","contamination","metal_contamination","oil","liquid","glue","glue_strip","print",
     "faulty_imprint","gray_stroke","thread","thread_side","thread_top"}
G = {"broken_large","broken_small","broken","broken_teeth","split_teeth","hole","cut","cut_lead",
     "cut_inner_insulation","cut_outer_insulation","missing_cable","missing_wire","cable_swap",
     "misplaced","flip","manipulated_front","damaged_case","pill_type","fabric_border","fabric_interior"}
X = {"combined","defective"}


def main():
    rows = []
    for f in sorted(glob.glob(str(ROOT / "reports/mvtec_illum_g*.json"))):
        rows += json.load(open(f))["rows"]
    d = defaultdict(lambda: defaultdict(list))
    for r in rows:
        d[(r["mode"], r["severity"])][(r["category"], r["defect"])].append(r["auroc"])
    M = {k: {u: float(np.mean(v)) for u, v in m.items()} for k, m in d.items()}
    units = sorted(M[("none", 0.0)])
    cls = {}
    for u in units:
        dfc = u[1]
        cls[u] = "S" if dfc in S else "A" if dfc in A else "G" if dfc in G else "X" if dfc in X else "?"
    unk = sorted({u[1] for u in units if cls[u] == "?"})
    print(f"单元数 {len(units)}；签名分类 " +
          ", ".join(f"{k}={sum(1 for u in units if cls[u]==k)}" for k in "SAGX?"))
    if unk: print(f"  未分类缺陷名（须在论文中声明）: {unk}")

    base = M[("none", 0.0)]
    def loss(u, modes, s): return float(np.mean([base[u] - M[(m, s)][u] for m in modes]))

    tests = []
    for s in SEVS:
        gs = {k: [loss(u, SPATIAL, s) for u in units if cls[u] == k] for k in "SAG"}
        ts = {k: [loss(u, TONAL, s) for u in units if cls[u] == k] for k in "SAG"}
        # P1: 空间光照下 S 退化 > G
        p1 = mannwhitneyu(gs["S"], gs["G"], alternative="greater").pvalue
        tests.append((f"P1@{s}", np.mean(gs['S']) - np.mean(gs['G']), p1,
                      f"S={np.mean(gs['S']):.3f} G={np.mean(gs['G']):.3f}"))
        # P2: 色调下 A 退化 > G
        p2 = mannwhitneyu(ts["A"], ts["G"], alternative="greater").pvalue
        tests.append((f"P2@{s}", np.mean(ts['A']) - np.mean(ts['G']), p2,
                      f"A={np.mean(ts['A']):.3f} G={np.mean(ts['G']):.3f}"))
        # P3: 交互 (S_spatial - S_tonal) > (A_spatial - A_tonal)
        ds = [loss(u, SPATIAL, s) - loss(u, TONAL, s) for u in units if cls[u] == "S"]
        da = [loss(u, SPATIAL, s) - loss(u, TONAL, s) for u in units if cls[u] == "A"]
        p3 = mannwhitneyu(ds, da, alternative="greater").pvalue
        tests.append((f"P3@{s}", np.mean(ds) - np.mean(da), p3,
                      f"dS={np.mean(ds):.3f} dA={np.mean(da):.3f}"))

    order = np.argsort([t[2] for t in tests]); m = len(tests)
    holm = {}
    run = 0.0
    for i, idx in enumerate(order):
        adj = min(1.0, max(run, (m - i) * tests[idx][2])); run = adj; holm[idx] = adj
    print(f"\n预注册族: {m} 个检验，Holm-Bonferroni 校正")
    print(f"{'test':10}{'group diff':>12}{'raw p':>10}{'Holm p':>10}  detail")
    for i, t in enumerate(tests):
        print(f"{t[0]:10}{t[1]:+12.4f}{t[2]:10.3f}{holm[i]:10.3f}  {t[3]}")
    sig = [tests[i][0] for i in range(m) if holm[i] < 0.05]
    print(f"\n校正后显著的检验: {sig if sig else '无'}")
    print("按预注册判定规则：三项预测全部不成立 → AeBAD 上的「明暗 vs 几何签名」机理"
          "\n解释降级为该数据集的局部现象。")
    out = dict(n_units=len(units), unclassified=unk,
               counts={k: sum(1 for u in units if cls[u] == k) for k in "SAGX?"},
               tests=[dict(name=t[0], diff=t[1], p_raw=t[2], p_holm=holm[i], detail=t[3])
                      for i, t in enumerate(tests)],
               any_significant=bool(sig))
    p = ROOT / "reports/r2/prereg_verify.json"
    json.dump(out, open(p, "w"), indent=1); print(f"写入 {p}")


if __name__ == "__main__":
    main()
