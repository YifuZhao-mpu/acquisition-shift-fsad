#!/usr/bin/env python3
"""C3：AeBAD-S 报告的 illumination 效应（75.2 -> 74.6）在其样本量下的可分辨性。

表述纪律：这是对**测量精度**的说明，不是对已发表结果的重分析。没有原文的
逐图预测分数，就无法估计那一次比较的实际方差；能做的只是刻画「在该样本量下，
一次 AUROC 比较能分辨多大的差异」。

两条独立路径：
  (a) Hanley-McNeil 解析近似（指数分布假设下的 AUROC 标准误）
  (b) 蒙特卡洛：在两域真值相同（无域效应）的零假设下，模拟分数并重算 AUROC
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm

RNG = np.random.default_rng(20260920)
# AeBAD-S 测试集实测计数（data/AeBAD_S_domains/single/*/test）
N = {"same": (230, 459), "illumination": (75, 198)}   # (normal, anomalous)
A_SAME, A_ILLUM = 0.752, 0.746                        # 原文 Table 2 的 PatchCore 行


def hanley_mcneil_se(A, n_neg, n_pos):
    Q1 = A / (2 - A); Q2 = 2 * A * A / (1 + A)
    v = (A * (1 - A) + (n_pos - 1) * (Q1 - A * A) + (n_neg - 1) * (Q2 - A * A)) / (n_pos * n_neg)
    return float(np.sqrt(v))


def simulate_auroc(A, n_neg, n_pos, B=20000):
    """在双正态等方差模型下模拟：mu 由目标 AUROC 反解，A = Phi(mu/sqrt(2))。"""
    mu = norm.ppf(A) * np.sqrt(2.0)
    out = np.empty(B)
    for b in range(B):
        x = RNG.normal(0, 1, n_neg); y = RNG.normal(mu, 1, n_pos)
        s = np.concatenate([x, y]); lab = np.r_[np.zeros(n_neg), np.ones(n_pos)]
        o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); rr = np.empty(len(s))
        ss = s[o]; i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and ss[j + 1] == ss[i]: j += 1
            rr[i:j + 1] = (i + j) / 2 + 1; i = j + 1
        r[o] = rr
        out[b] = (r[lab == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return out


def main():
    print("AeBAD-S 测试集计数（实测）:", {k: f"{v[0]} normal + {v[1]} anomalous" for k, v in N.items()})
    print(f"原文报告: same {A_SAME:.3f} vs illumination {A_ILLUM:.3f}  → gap {A_SAME-A_ILLUM:.4f}\n")

    print("(a) Hanley-McNeil 解析标准误与 95% 区间宽度")
    for dom, A in [("same", A_SAME), ("illumination", A_ILLUM)]:
        nn, np_ = N[dom]
        se = hanley_mcneil_se(A, nn, np_)
        print(f"    {dom:14} A={A:.3f}  SE={se:.4f}  95% CI 宽度={2*1.96*se:.4f}")
    se_s = hanley_mcneil_se(A_SAME, *N['same'][::1])
    se_i = hanley_mcneil_se(A_ILLUM, *N['illumination'][::1])
    se_d = np.sqrt(se_s**2 + se_i**2)
    print(f"    差值的 SE（视两域独立）= {se_d:.4f}；报告的 gap 0.006 = {0.006/se_d:.2f} SE")
    p_an = 2 * (1 - norm.cdf(0.006 / se_d))
    print(f"    零假设下 P(|Δ| ≥ 0.006) ≈ {p_an:.3f}（解析）\n")

    print("(b) 蒙特卡洛（零假设：两域真值同为 0.749）")
    A0 = (A_SAME + A_ILLUM) / 2
    a_s = simulate_auroc(A0, *N['same'])
    a_i = simulate_auroc(A0, *N['illumination'])
    print(f"    same         95% 区间宽度 = {np.percentile(a_s,97.5)-np.percentile(a_s,2.5):.4f}")
    print(f"    illumination 95% 区间宽度 = {np.percentile(a_i,97.5)-np.percentile(a_i,2.5):.4f}")
    d = np.abs(a_s - a_i)
    print(f"    P(|Δ| ≥ 0.006) = {(d >= 0.006).mean():.3f}")
    print(f"    |Δ| 的中位数   = {np.median(d):.4f}   （报告的 gap 为 0.006）")
    print(f"    报告的 gap 占 illumination 区间宽度的 "
          f"{0.006/(np.percentile(a_i,97.5)-np.percentile(a_i,2.5))*100:.1f}%")
    print("\n结论（限定表述）：在该样本量下，一次 AUROC 比较无法分辨 0.006 量级的差异；"
          "\n该测量不具备检出光照效应的效力。这不主张原文结论为假。")
    import json
    from pathlib import Path as _P
    out = dict(n_same=list(N["same"]), n_illum=list(N["illumination"]),
               a_same=A_SAME, a_illum=A_ILLUM, gap=A_SAME - A_ILLUM,
               hm_se_same=se_s, hm_se_illum=se_i, hm_se_diff=se_d,
               hm_ci_width_illum=2 * 1.96 * se_i, hm_p=p_an,
               mc_ci_width_illum=float(np.percentile(a_i, 97.5) - np.percentile(a_i, 2.5)),
               mc_ci_width_same=float(np.percentile(a_s, 97.5) - np.percentile(a_s, 2.5)),
               mc_p=float((d >= 0.006).mean()), mc_median_gap=float(np.median(d)),
               gap_pct_of_ci=float(0.006 / (np.percentile(a_i, 97.5) - np.percentile(a_i, 2.5)) * 100))
    _p = _P(str(ROOT) + "/reports/r2/c3_power.json")
    _p.parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(_p, "w"), indent=1)
    print(f"写入 {_p}")


if __name__ == "__main__":
    main()
