#!/usr/bin/env python3
"""评测引擎的三项等价性核验 —— 论文 §VII。

(a) fscache vs 参考实现（AnomalyDINO run_anomalydino.py）：256 个 AeBAD 单元
(b) 批处理匹配 vs 逐图匹配：真实数据上的逐位比对
(c) 逐域缓存 vs 整批缓存（Round-2 为绕开 OOM 所做的改造）

(b)(c) 在同一组对照里同时被检验：Round-1 的 replay_nomask_VALIDATION.json 由
**逐图 + 整批缓存**的旧引擎产出；Round-2 的 aebad_*.json 由**批处理 + 逐域缓存**
的新引擎产出。两者若逐位相同，则两项改造都不改变数值。
"""
from __future__ import annotations
import os
import json
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("IADSHIFT_ROOT",
                    Path(__file__).resolve().parents[2]))   # experiment/s1/x.py -> 项目根


def load_rows(p, scope="all"):
    d = json.load(open(p))
    rows = d["rows"] if isinstance(d, dict) else d
    return {(r["protocol"], r["k"], r["seed"], r["domain"]): r
            for r in rows if r.get("scope") == scope}


def main():
    out = {}

    # ── (a) fscache vs 参考实现 ──
    A = load_rows(ROOT / "reports/replay_nomask_VALIDATION.json")
    B = {(r["protocol"], r["k"], r["seed"], r["domain"]): r
         for r in json.load(open(ROOT / "reports/s1_metrics.json"))["rows"]
         if r.get("preprocess") == "agnostic"}
    common = sorted(set(A) & set(B))
    d = np.array([A[k]["auroc"] - B[k]["auroc"] for k in common])
    units = np.array([abs(A[k]["auroc"] - B[k]["auroc"]) /
                      (1.0 / (B[k]["n_pos"] * (B[k]["n"] - B[k]["n_pos"]))) for k in common])
    near = np.round(units * 2) / 2
    resid = np.abs(units - near)
    corr = float(np.corrcoef([A[k]["auroc"] for k in common], [B[k]["auroc"] for k in common])[0, 1])
    print("(a) fscache engine vs reference implementation")
    print(f"    units={len(common)}  max|Δ|={np.abs(d).max():.3e}  corr={corr:.9f}")
    print(f"    每个差值到最近半整数秩对单位的最大偏差 = {resid.max():.6f}"
          f"  ({'全部精确' if resid.max() < 1e-9 else '不精确'})")
    print("    倍数分布: " + ", ".join(f"{v:g}u×{c}" for v, c in sorted(Counter(near).items())))
    out["reference"] = dict(units=len(common), max_abs_diff=float(np.abs(d).max()), corr=corr,
                            max_residual_from_half_integer=float(resid.max()),
                            distribution={str(v): int(c) for v, c in sorted(Counter(near).items())})

    # ── (b)+(c) 批处理 + 逐域缓存 vs 逐图 + 整批缓存 ──
    f = ROOT / "reports/r2/aebad_dinov2_vits14_meantop1p.json"
    if f.exists():
        C = load_rows(f)
        common2 = sorted(set(A) & set(C))
        d2 = np.array([C[k]["auroc"] - A[k]["auroc"] for k in common2])
        print("\n(b)+(c) batched matching + per-domain caching vs per-image + whole-set caching")
        print(f"    units={len(common2)}  max|Δ|={np.abs(d2).max():.3e}  nonzero={(d2 != 0).sum()}")
        print(f"    -> {'bit-identical' if np.abs(d2).max() == 0 else 'DIFFERS'}")
        out["batched_and_perdomain"] = dict(units=len(common2),
                                            max_abs_diff=float(np.abs(d2).max()),
                                            n_nonzero=int((d2 != 0).sum()))
    # ── (d) MVTec 全网格上的批处理等价性（73 单元 x 19 条件 x 8 seed）──
    g = ROOT / "reports/r2/xbb2_dinov2_vits14_meantop1p.json"
    if g.exists():
        import glob as _g
        old_rows = []
        for f2 in sorted(_g.glob(str(ROOT / "reports/mvtec_illum_g*.json"))):
            old_rows += json.load(open(f2))["rows"]
        key = lambda r: (r["category"], r["defect"], r["mode"], r["severity"], r["seed"])
        O = {key(r): r["auroc"] for r in old_rows}
        N = {key(r): r["auroc"] for r in json.load(open(g))["rows"]}
        c3 = sorted(set(O) & set(N))
        d3 = np.array([N[k] - O[k] for k in c3])
        print("\n(d) batched matching vs per-image, full MVTec grid (DINOv2 ViT-S/14)")
        print(f"    rows={len(c3)}  max|Δ|={np.abs(d3).max():.3e}  nonzero={(d3 != 0).sum()}")
        print(f"    -> {'bit-identical' if np.abs(d3).max() == 0 else 'DIFFERS'}")
        out["batched_mvtec"] = dict(rows=len(c3), max_abs_diff=float(np.abs(d3).max()),
                                    n_nonzero=int((d3 != 0).sum()))
    p = ROOT / "reports/r2/engine_equiv.json"
    json.dump(out, open(p, "w"), indent=1)
    print(f"\n写入 {p}")


if __name__ == "__main__":
    main()
