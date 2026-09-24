#!/usr/bin/env python3
"""
配对分析：这是本实验设计所要求的正确统计方法。

四个域在同一 seed 下使用**逐字节相同**的参考集（见 build_aebad_domains.py 的
ref_set_sha256）。因此 seed 间波动对 same 与 shifted 是**共模**的，
应在配对差 Δ_seed = AUROC(shifted, seed) − AUROC(same, seed) 中抵消。

用边际均值 + 合并标准差比较会把这部分共模方差错误地计入噪声，
显著低估检验效力。
"""
import json, sys
from collections import defaultdict
import numpy as np

DOM = ["illumination", "view", "background"]


def wilcoxon_signed_rank_p(d):
    """精确 Wilcoxon 符号秩检验（n<=10 时枚举全部符号组合）。"""
    d = np.asarray([x for x in d if x != 0], float)
    n = len(d)
    if n == 0: return 1.0
    r = np.argsort(np.argsort(np.abs(d))) + 1.0
    # 并列取平均秩
    a = np.abs(d); uniq, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(uniq)); np.add.at(sums, inv, r); r = (sums / cnt)[inv]
    W = r[d > 0].sum()
    if n > 12:   # 正态近似
        mu = n * (n + 1) / 4; sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        from math import erfc
        z = (W - mu) / sd
        return float(erfc(abs(z) / np.sqrt(2)))
    # 精确枚举
    tot = 0; extreme = 0
    for mask in range(1 << n):
        s = sum(r[i] for i in range(n) if mask >> i & 1)
        tot += 1
        if abs(s - n * (n + 1) / 4) >= abs(W - n * (n + 1) / 4) - 1e-12: extreme += 1
    return extreme / tot


def main(path):
    d = json.load(open(path))
    by = defaultdict(dict)          # (proto,k,seed) -> {domain: auroc}
    fpr = defaultdict(dict)
    for r in d["rows"]:
        by[(r["protocol"], r["k"], r["seed"])][r["domain"]] = r["auroc"]
        fpr[(r["protocol"], r["k"], r["seed"])][r["domain"]] = r["fpr95"]

    print("=" * 104)
    print("配对分析：Δ_seed = AUROC(shifted) − AUROC(same)，同 seed 同参考集")
    print("=" * 104)
    print(f"{'proto':7s} {'k':>2s} {'domain':13s} {'mean Δ':>8s} {'配对sd':>8s} {'d(效应量)':>9s} "
          f"{'95%CI':>18s} {'Wilcoxon p':>11s} {'n':>3s}")
    print("-" * 104)
    out = []
    for proto in ("single", "mixed"):
        for k in (1, 2, 4, 8):
            for dom in DOM:
                ds = []
                for seed in range(8):
                    v = by.get((proto, k, seed))
                    if v and "same" in v and dom in v:
                        ds.append(v[dom] - v["same"])
                if len(ds) < 3: continue
                ds = np.array(ds)
                m, sd = ds.mean(), ds.std(ddof=1)
                se = sd / np.sqrt(len(ds))
                # 配对 bootstrap CI
                rng = np.random.default_rng(0)
                bs = np.array([rng.choice(ds, len(ds), True).mean() for _ in range(10000)])
                lo, hi = np.quantile(bs, [0.025, 0.975])
                p = wilcoxon_signed_rank_p(ds)
                dz = m / sd if sd > 0 else np.inf
                sig = "  ★" if (lo > 0 or hi < 0) else ""
                print(f"{proto:7s} {k:>2d} {dom:13s} {m:+8.4f} {sd:8.4f} {dz:+9.2f} "
                      f"[{lo:+.4f},{hi:+.4f}] {p:11.4f} {len(ds):>3d}{sig}")
                out.append(dict(protocol=proto, k=k, domain=dom, mean_delta=float(m),
                                paired_sd=float(sd), cohens_dz=float(dz),
                                ci_lo=float(lo), ci_hi=float(hi), wilcoxon_p=float(p),
                                n_seeds=len(ds), ci_excludes_zero=bool(lo > 0 or hi < 0)))
    print("\n★ = 配对 bootstrap 95% CI 不含 0")

    # ---- C1 直接检验：illumination 是否比 view 更差（同 seed 配对）----
    print("\n" + "=" * 104)
    print("C1 直接检验：Δ_illum − Δ_view（同 seed 配对；<0 表示光照伤害更大）")
    print("=" * 104)
    print(f"{'proto':7s} {'k':>2s} {'mean':>9s} {'sd':>8s} {'95%CI':>20s} {'Wilcoxon p':>11s} {'判定':>8s}")
    print("-" * 104)
    c1 = []
    for proto in ("single", "mixed"):
        for k in (1, 2, 4, 8):
            ds = []
            for seed in range(8):
                v = by.get((proto, k, seed))
                if v and all(x in v for x in ("same", "illumination", "view")):
                    ds.append((v["illumination"] - v["same"]) - (v["view"] - v["same"]))
            if len(ds) < 3: continue
            ds = np.array(ds); m, sd = ds.mean(), ds.std(ddof=1)
            rng = np.random.default_rng(1)
            bs = np.array([rng.choice(ds, len(ds), True).mean() for _ in range(10000)])
            lo, hi = np.quantile(bs, [0.025, 0.975])
            p = wilcoxon_signed_rank_p(ds)
            verdict = "光照更差" if hi < 0 else ("视角更差" if lo > 0 else "不可区分")
            print(f"{proto:7s} {k:>2d} {m:+9.4f} {sd:8.4f} [{lo:+.4f},{hi:+.4f}] {p:11.4f} {verdict:>8s}")
            c1.append(dict(protocol=proto, k=k, mean=float(m), ci_lo=float(lo),
                           ci_hi=float(hi), wilcoxon_p=float(p), verdict=verdict))

    # ---- C2 直接检验：mixed 相对 single 是否缓解光照退化 ----
    print("\n" + "=" * 104)
    print("C2 直接检验：支撑集多样性的效果 Δ_illum(mixed) − Δ_illum(single)，逐 seed 配对")
    print("=" * 104)
    print(f"{'k':>2s} {'mean':>9s} {'sd':>8s} {'95%CI':>20s} {'Wilcoxon p':>11s} {'判定':>14s}")
    print("-" * 104)
    c2 = []
    for k in (1, 2, 4, 8):
        ds = []
        for seed in range(8):
            a, b = by.get(("mixed", k, seed)), by.get(("single", k, seed))
            if a and b and all("illumination" in x and "same" in x for x in (a, b)):
                ds.append((a["illumination"] - a["same"]) - (b["illumination"] - b["same"]))
        if len(ds) < 3: continue
        ds = np.array(ds); m, sd = ds.mean(), ds.std(ddof=1)
        rng = np.random.default_rng(2)
        bs = np.array([rng.choice(ds, len(ds), True).mean() for _ in range(10000)])
        lo, hi = np.quantile(bs, [0.025, 0.975])
        p = wilcoxon_signed_rank_p(ds)
        verdict = "多样性有效" if lo > 0 else ("多样性有害" if hi < 0 else "不可区分")
        print(f"{k:>2d} {m:+9.4f} {sd:8.4f} [{lo:+.4f},{hi:+.4f}] {p:11.4f} {verdict:>14s}")
        c2.append(dict(k=k, mean=float(m), ci_lo=float(lo), ci_hi=float(hi),
                       wilcoxon_p=float(p), verdict=verdict))

    json.dump(dict(paired=out, c1=c1, c2=c2),
              open(path.replace(".json", "_paired.json"), "w"), indent=2, ensure_ascii=False)
    print(f"\n写入 {path.replace('.json','_paired.json')}")


if __name__ == "__main__":
    main(sys.argv[1])
