#!/usr/bin/env python3
"""按 PREREGISTRATION.md 的锁定规则分析 MVTec 扰动实验。

分类直接从预注册文件解析，杜绝"事后微调分类"。
检验族 = {P1,P2,P3} × 3 severity = 9，Holm-Bonferroni 校正。
族外比较一律标注为探索性。
"""
import glob, json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
SPATIAL = ["gradient", "specular", "shadow"]   # 改变光照空间分布
TONAL   = ["exposure", "gamma", "wb"]          # 改变全局色调映射


def load_prereg():
    txt = (HERE / "PREREGISTRATION.md").read_text()
    out = {}
    for tag in ("S", "A", "G", "X"):
        m = re.search(rf'### {tag} = .*?\n(.*?)(?=\n###|\n## )', txt, re.S)
        for d in re.findall(r'`([a-z_]+)`', m.group(1)):
            out[d] = tag
    return out


def mannwhitney_u_p(x, y):
    """双侧 Mann-Whitney U，正态近似 + 并列校正。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0: return 1.0
    a = np.concatenate([x, y])
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a)); r[order] = np.arange(1, len(a) + 1)
    uq, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    s = np.zeros(len(uq)); np.add.at(s, inv, r); r = (s / cnt)[inv]
    R1 = r[:n1].sum()
    U1 = R1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    tie = (cnt ** 3 - cnt).sum()
    sd = np.sqrt(n1 * n2 / 12 * ((n1 + n2 + 1) - tie / ((n1 + n2) * (n1 + n2 - 1))))
    if sd == 0: return 1.0
    from math import erfc
    z = (abs(U1 - mu) - 0.5) / sd
    return float(erfc(z / np.sqrt(2)))


def boot_diff_ci(x, y, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = np.array([rng.choice(x, len(x), True).mean() - rng.choice(y, len(y), True).mean()
                  for _ in range(n)])
    return float(np.quantile(d, .025)), float(np.quantile(d, .975))


def holm(pvals, labels, alpha=0.05):
    order = np.argsort(pvals)
    m = len(pvals); out = {}
    prev = 0.0
    for i, idx in enumerate(order):
        thr = alpha / (m - i)
        adj = max(prev, min(1.0, pvals[idx] * (m - i)))
        prev = adj
        out[labels[idx]] = (pvals[idx], adj, adj < alpha)
    return out


def main(pattern):
    SIG = load_prereg()
    rows = []
    for f in sorted(glob.glob(pattern)):
        rows += json.load(open(f))["rows"]
    print(f"载入 {len(rows)} 行，来自 {len(glob.glob(pattern))} 个文件")

    # (cat,defect,mode,sev) -> 跨 seed 均值
    agg = defaultdict(list)
    for r in rows: agg[(r["category"], r["defect"], r["mode"], r["severity"])].append(r["auroc"])
    A = {k: float(np.mean(v)) for k, v in agg.items()}
    base = {(c, d): A[(c, d, "none", 0.0)] for (c, d, m, s) in A if m == "none"}

    units = sorted({(c, d) for (c, d, m, s) in A})
    print(f"(类别,缺陷) 单元: {len(units)}")
    by_sig = defaultdict(list)
    for c, d in units: by_sig[SIG.get(d, "?")].append((c, d))
    print("  签名分布: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(by_sig.items())))

    sevs = sorted({s for (_, _, m, s) in A if s > 0})

    def delta(c, d, modes, sev):
        vs = [A[(c, d, m, sev)] - base[(c, d)] for m in modes if (c, d, m, sev) in A]
        return float(np.mean(vs)) if vs else None

    pv, lab, detail = [], [], {}
    for sev in sevs:
        dS_sp = [delta(c, d, SPATIAL, sev) for c, d in by_sig["S"]]
        dG_sp = [delta(c, d, SPATIAL, sev) for c, d in by_sig["G"]]
        dA_to = [delta(c, d, TONAL, sev) for c, d in by_sig["A"]]
        dG_to = [delta(c, d, TONAL, sev) for c, d in by_sig["G"]]
        iS = [delta(c, d, SPATIAL, sev) - delta(c, d, TONAL, sev) for c, d in by_sig["S"]]
        iA = [delta(c, d, SPATIAL, sev) - delta(c, d, TONAL, sev) for c, d in by_sig["A"]]
        for name, x, y in (("P1", dS_sp, dG_sp), ("P2", dA_to, dG_to), ("P3", iS, iA)):
            x = [v for v in x if v is not None]; y = [v for v in y if v is not None]
            p = mannwhitney_u_p(x, y); lo, hi = boot_diff_ci(x, y)
            k = f"{name}@sev={sev}"
            pv.append(p); lab.append(k)
            detail[k] = (np.mean(x), np.mean(y), np.mean(x) - np.mean(y), lo, hi, len(x), len(y))

    res = holm(pv, lab)
    print(f"\n{'='*104}\n预注册检验族（9 个，Holm-Bonferroni 校正，alpha=0.05）\n{'='*104}")
    print(f"{'检验':14s} {'组均值差':>10s} {'x̄':>9s} {'ȳ':>9s} {'95%CI':>20s} {'p原始':>9s} {'p校正':>9s}  判定")
    print("-" * 104)
    for k in lab:
        mx, my, d, lo, hi, nx, ny = detail[k]
        p, adj, sig = res[k]
        print(f"{k:14s} {d:+10.4f} {mx:+9.4f} {my:+9.4f} [{lo:+.4f},{hi:+.4f}] "
              f"{p:9.4f} {adj:9.4f}  {'✓ 显著' if sig else '✗ 不显著'}")
    print("\nP1: S在空间光照下退化 vs G；P2: A在色调下退化 vs G；P3: (空间−色调)退化差，S vs A")
    print("Δ 为负表示 AUROC 下降；组均值差为负表示前者退化更重。")
    return detail, res


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../../reports/mvtec_illum_g*.json")
