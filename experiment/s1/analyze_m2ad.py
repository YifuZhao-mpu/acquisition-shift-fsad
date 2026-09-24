#!/usr/bin/env python3
"""M2AD 预注册检验 M1–M4（见 M2AD_PREREGISTRATION.md）。

检验族 = {M1 自监督臂, M1 监督臂, M2 ViT 臂, M2 CNN 臂, M3, M4}，共 6 个，Holm–Bonferroni，α=0.05。
族外的一切（E2、逐类别、逐 k）一律打印为**探索性**。
"""
from __future__ import annotations
import os
import glob, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from paired import wilcoxon_signed_rank_p

ROOT = Path(os.environ.get("IADSHIFT_ROOT",
                    Path(__file__).resolve().parents[2]))   # experiment/s1/x.py -> 项目根
RNG = np.random.default_rng(0)
REF_ILLUM = "01"
VIT_S, VIT_SUP = "dinov2_vits14", "deit_small_patch16"
CNN_S, CNN_SUP = "dino_resnet50", "resnet50"
ANCHOR = "wide_resnet50_2"
NICE = {VIT_S: "DINOv2 ViT-S/14", VIT_SUP: "DeiT-S/16", CNN_S: "DINO ResNet-50",
        CNN_SUP: "ResNet-50 (sup.)", ANCHOR: "WideResNet-50-2"}


def paired(d, B=10000):
    d = np.asarray(d, float)
    if len(d) == 0: return np.nan, np.nan, np.nan, 1.0
    bs = d[RNG.integers(0, len(d), (B, len(d)))].mean(1)
    return d.mean(), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), \
        float(wilcoxon_signed_rank_p(d))


def holm(pairs, alpha=0.05):
    """pairs: [(name, p)] -> {name: (p, 阈值, 是否通过)}"""
    order = sorted(pairs, key=lambda x: x[1])
    m, out, blocked = len(order), {}, False
    for i, (nm, p) in enumerate(order):
        thr = alpha / (m - i)
        ok = (p <= thr) and not blocked
        if not ok: blocked = True
        out[nm] = (p, thr, ok)
    return out


def load(agg="meantop1p", rdir=None):
    """-> A[backbone][scope][(arm,cat,view,illum)] = 逐 seed 平均 AUROC"""
    A = defaultdict(lambda: defaultdict(dict))
    acc = defaultdict(list)
    base = Path(rdir) if rdir else ROOT / "reports/m2ad"
    files = sorted(glob.glob(str(base / f"*_{agg}.json")))
    for f in files:
        d = json.load(open(f))
        for r in d["rows"]:
            acc[(r["backbone"], r["scope"],
                 (r["arm"], r["category"], r["view"], r["illum"]))].append(r["auroc"])
    for (bb, sc, key), v in acc.items():
        A[bb][sc][key] = float(np.mean(v))
    print(f"载入 {len(files)} 个结果文件，{len(acc)} 个 (backbone,scope,unit) 组合")
    for bb in sorted(A):
        for sc in sorted(A[bb]):
            e1 = sum(1 for k in A[bb][sc] if k[0] == "E1")
            print(f"  {NICE.get(bb,bb):20} {sc:11} E1 单元 {e1}")
    return A


def delta_vec(A, bb, scope, units):
    """逐单元 Δ = AUROC(偏移档) − AUROC(匹配档)，同 (类别,视角) 内配对。"""
    out = []
    for (arm, cat, view, il) in units:
        base = A[bb][scope].get((arm, cat, view, REF_ILLUM))
        cur = A[bb][scope].get((arm, cat, view, il))
        out.append(np.nan if base is None or cur is None else cur - base)
    return np.array(out, float)


def common_units(A, scope, bbs, arm="E1", shifted_only=True):
    sets = [{k for k in A[b][scope] if k[0] == arm} for b in bbs]
    u = set.intersection(*sets) if sets else set()
    if shifted_only: u = {k for k in u if k[3] != REF_ILLUM}
    # 同时要求匹配档存在
    u = {k for k in u if all((arm, k[1], k[2], REF_ILLUM) in A[b][scope] for b in bbs)}
    return sorted(u)


def contrast(A, scope, a, b, units):
    d = delta_vec(A, a, scope, units) - delta_vec(A, b, scope, units)
    m = ~np.isnan(d)
    return paired(d[m]), int(m.sum())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("agg", nargs="?", default="meantop1p", choices=["meantop1p", "max"])
    ap.add_argument("--reports-dir", default=None, help="覆盖结果目录（冒烟测试用）")
    ap.add_argument("--physical", default=None, help="覆盖 m2ad_physical.json 路径")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    agg = a.agg
    A = load(agg, a.reports_dir)
    have = [b for b in (VIT_S, VIT_SUP, CNN_S, CNN_SUP) if b in A]
    if len(have) < 4:
        print(f"\n缺少 backbone：{set((VIT_S,VIT_SUP,CNN_S,CNN_SUP)) - set(have)}——只做已有部分")
    results = {}

    for scope in ("all", "detectable"):
        print("\n" + "=" * 96)
        print(f"标签口径 = {scope}   （聚合 {agg}）")
        print("=" * 96)
        bbs = [b for b in (VIT_S, VIT_SUP, CNN_S, CNN_SUP, ANCHOR) if b in A and scope in A[b]]
        if len(bbs) < 2: print("  数据不足"); continue
        units = common_units(A, scope, bbs)
        print(f"配对单元（E1 偏移档）：{len(units)}")

        print("\n-- 各表示的平均退化 Δ --")
        for b in bbs:
            d = delta_vec(A, b, scope, units); d = d[~np.isnan(d)]
            base = np.mean([A[b][scope][("E1", c, v, REF_ILLUM)]
                            for (_, c, v, _) in units
                            if ("E1", c, v, REF_ILLUM) in A[b][scope]])
            print(f"  {NICE.get(b,b):22} 匹配档 {base:.4f}   Δ {d.mean():+.4f}   偏移档 {base+d.mean():.4f}")

        tests = []
        if len(have) == 4:
            print("\n-- M1 架构对比（ViT − CNN，预测 > 0）--")
            for nm, a_, b_ in (("M1-自监督臂", VIT_S, CNN_S), ("M1-监督臂", VIT_SUP, CNN_SUP)):
                (m, lo, hi, p), n = contrast(A, scope, a_, b_, units)
                print(f"  {nm:14} {NICE[a_]:18} − {NICE[b_]:18} = {m:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.2e} (n={n})")
                tests.append((nm, p)); results[(scope, nm)] = (m, lo, hi, p, n)

            print("\n-- M2 预训练对比（自监督 − 监督）--")
            for nm, a_, b_ in (("M2-ViT臂", VIT_S, VIT_SUP), ("M2-CNN臂", CNN_S, CNN_SUP)):
                (m, lo, hi, p), n = contrast(A, scope, a_, b_, units)
                print(f"  {nm:14} {NICE[a_]:18} − {NICE[b_]:18} = {m:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.2e} (n={n})")
                tests.append((nm, p)); results[(scope, nm)] = (m, lo, hi, p, n)

            a1 = abs(results[(scope, "M1-自监督臂")][0]); a2 = abs(results[(scope, "M1-监督臂")][0])
            p1 = results[(scope, "M2-ViT臂")][0]; p2 = results[(scope, "M2-CNN臂")][0]
            print(f"\n  M1 判定：两臂同号为正? "
                  f"{results[(scope,'M1-自监督臂')][0] > 0 and results[(scope,'M1-监督臂')][0] > 0}")
            print(f"  M2 判定：预训练对比幅度小于架构? {max(abs(p1),abs(p2)) < min(a1,a2)}"
                  f"；两臂异号? {p1 * p2 < 0}")

        # ---- M3：跨 9 档，架构对比 vs 色调主导指数 ----
        phys_p = Path(a.physical) if a.physical else ROOT / "reports/r2/m2ad_physical.json"
        if len(have) == 4 and phys_p.exists():
            phys = json.load(open(phys_p))["by_illum"]
            xs, ys, per = [], [], {}
            for j in sorted(phys):
                uj = [u for u in units if u[3] == j]
                if not uj: continue
                d = ((delta_vec(A, VIT_S, scope, uj) - delta_vec(A, CNN_S, scope, uj)) +
                     (delta_vec(A, VIT_SUP, scope, uj) - delta_vec(A, CNN_SUP, scope, uj))) / 2
                d = d[~np.isnan(d)]
                if len(d) == 0: continue
                xs.append(phys[j]["tone_idx"]); ys.append(float(d.mean())); per[j] = (xs[-1], ys[-1], len(d))
            if len(xs) >= 4:
                def spear(x, y):
                    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
                    rx -= rx.mean(); ry -= ry.mean()
                    return float((rx @ ry) / np.sqrt((rx @ rx) * (ry @ ry)))
                rho = spear(np.array(xs), np.array(ys))
                idx = np.arange(len(xs))
                bs = [spear(np.array(xs)[s], np.array(ys)[s])
                      for s in (RNG.integers(0, len(xs), (4000, len(xs))))
                      if len(set(np.array(xs)[s])) > 2]
                lo, hi = (np.percentile(bs, 2.5), np.percentile(bs, 97.5)) if bs else (np.nan, np.nan)
                # 置换检验（9 档全排列太多，用 20000 次随机置换）
                null = [spear(np.array(xs), RNG.permutation(ys)) for _ in range(20000)]
                p = float((np.abs(null) >= abs(rho)).mean())
                print("\n-- M3 物理预测：架构优势 vs 色调主导指数（预测 ρ > 0）--")
                print(f"  {'illum':7}{'tone_idx':>10}{'ViT−CNN':>12}{'n':>7}")
                for j in sorted(per, key=lambda j: per[j][0]):
                    print(f"  {j:7}{per[j][0]:10.3f}{per[j][1]:+12.4f}{per[j][2]:7d}")
                print(f"  Spearman ρ = {rho:+.3f}  [{lo:+.3f},{hi:+.3f}]  置换 p = {p:.4f}  (9 档)")
                tests.append(("M3", p)); results[(scope, "M3")] = (rho, lo, hi, p, len(xs))
        elif len(have) == 4:
            print(f"\n-- M3 跳过：{phys_p} 不存在，先跑 m2ad_physical.py --")

        # ---- M4：排序反转 ----
        if len(bbs) >= 3:
            base = {b: np.mean([A[b][scope][("E1", c, v, REF_ILLUM)] for (_, c, v, _) in units
                                if ("E1", c, v, REF_ILLUM) in A[b][scope]]) for b in bbs}
            shift = {b: np.nanmean(delta_vec(A, b, scope, units)) + base[b] for b in bbs}
            top_c = max(base, key=base.get); top_s = max(shift, key=shift.get)
            print("\n-- M4 排序反转 --")
            print(f"  匹配档最好：{NICE.get(top_c,top_c)} ({base[top_c]:.4f})")
            print(f"  偏移档最好：{NICE.get(top_s,top_s)} ({shift[top_s]:.4f})")
            print(f"  发生反转：{top_c != top_s}")
            if top_c != top_s:
                d = delta_vec(A, top_s, scope, units) - delta_vec(A, top_c, scope, units)
                d = d[~np.isnan(d)]
                m, lo, hi, p = paired(d)
                print(f"  反转幅度（偏移档最好 − 匹配档最好 的 Δ）= {m:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.2e}")
                tests.append(("M4", p)); results[(scope, "M4")] = (m, lo, hi, p, len(d))
            else:
                tests.append(("M4", 1.0)); results[(scope, "M4")] = (0.0, 0.0, 0.0, 1.0, 0)

        if tests:
            print("\n-- Holm–Bonferroni（族内 6 个检验，α=0.05）--")
            for nm, (p, thr, ok) in sorted(holm(tests).items(), key=lambda x: x[1][0]):
                print(f"  {nm:14} p={p:.3e}  阈值={thr:.4f}  {'通过' if ok else '未通过'}")

    # ---- 探索性：E2 视角臂 ----
    print("\n" + "=" * 96); print("探索性（不在检验族内）：E2 纯视角偏移"); print("=" * 96)
    for scope in ("all",):
        bbs = [b for b in (VIT_S, VIT_SUP, CNN_S, CNN_SUP, ANCHOR) if b in A and scope in A[b]]
        u2 = [k for k in set.intersection(*[{k for k in A[b][scope] if k[0] == "E2"} for b in bbs])
              if k[2] != "000"] if bbs else []
        if not u2: print("  无 E2 数据"); continue
        for b in bbs:
            dd = []
            for (arm, c, v, il) in u2:
                base = A[b][scope].get(("E2", c, "000", il)); cur = A[b][scope].get((arm, c, v, il))
                if base is not None and cur is not None: dd.append(cur - base)
            print(f"  {NICE.get(b,b):22} 视角偏移 Δ = {np.mean(dd):+.4f}  (n={len(dd)})")

    out = {f"{s}|{n}": dict(zip(("est", "lo", "hi", "p", "n"), v)) for (s, n), v in results.items()}
    p = Path(a.out) if a.out else ROOT / f"reports/r2/m2ad_tests_{agg}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(p, "w"), indent=1)
    print(f"\n写入 {p}")


if __name__ == "__main__":
    main()
