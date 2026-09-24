#!/usr/bin/env python3
"""按缺陷类型分层：光照偏移是否对某些缺陷类型伤害更大？

零额外计算——直接复用主扫描已写出的逐图异常分数。
每个 (domain, defect) 的 AUROC = 该域该缺陷 vs **同域的正常图**。
正常集在同一域内共享，故不同缺陷的 AUROC 可比。
"""
import csv, glob, re, sys
from collections import defaultdict
import numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location('an', 'experiment/s1/analyze.py')
an = importlib.util.module_from_spec(spec); spec.loader.exec_module(an)

DEFECTS = ["ablation", "breakdown", "fracture", "groove"]
DOMAINS = ["same", "illumination", "view", "background"]


def load(prep, proto, k, seed):
    p = (f"experiment/AnomalyDINO/results_AeBAD_{proto}/dinov2_vits14_448/"
         f"{k}-shot_preprocess={prep}/measurements_seed={seed}.csv")
    out = defaultdict(lambda: defaultdict(list))   # domain -> cls -> [scores]
    try:
        for r in csv.DictReader(open(p)):
            dom = r["Object"].replace("blade_", "")
            cls = r["Sample"].split("/")[0]
            out[dom][cls].append(float(r["Anomaly_Score"]))
    except OSError:
        return None
    return out


def main(prep="force_mask_rotation", proto="single"):
    print("=" * 100)
    print(f"按缺陷类型分层的光照退化  [preprocess={prep}, protocol={proto}]")
    print("每格 = Δ AUROC(该缺陷 @ illumination域) − AUROC(该缺陷 @ same域)，逐 seed 配对")
    print("=" * 100)
    print(f"{'k':>2s} {'defect':11s} {'n(same)':>8s} {'n(illum)':>9s} {'AUROC same':>11s} "
          f"{'AUROC illum':>12s} {'Δ':>8s} {'95%CI':>19s} {'p':>8s}")
    print("-" * 100)
    for k in (1, 2, 4, 8):
        for dfc in DEFECTS:
            ds, a_s, a_i = [], [], []
            n_s = n_i = 0
            for seed in range(8):
                d = load(prep, proto, k, seed)
                if not d: continue
                for dom, acc in (("same", a_s), ("illumination", a_i)):
                    pass
                try:
                    gs, ps = d["same"]["good"], d["same"][dfc]
                    gi, pi = d["illumination"]["good"], d["illumination"][dfc]
                except KeyError:
                    continue
                if not ps or not pi: continue
                n_s, n_i = len(ps), len(pi)
                ys = np.r_[np.zeros(len(gs)), np.ones(len(ps))].astype(int)
                ss = np.r_[gs, ps]
                yi = np.r_[np.zeros(len(gi)), np.ones(len(pi))].astype(int)
                si = np.r_[gi, pi]
                As, Ai = an.auroc(ys, ss), an.auroc(yi, si)
                a_s.append(As); a_i.append(Ai); ds.append(Ai - As)
            if len(ds) < 3: continue
            ds = np.array(ds); m = ds.mean()
            rng = np.random.default_rng(5)
            bs = np.array([rng.choice(ds, len(ds), True).mean() for _ in range(10000)])
            lo, hi = np.quantile(bs, [0.025, 0.975])
            # 精确 Wilcoxon
            sys.path.insert(0, 'experiment/s1')
            from paired import wilcoxon_signed_rank_p as wp
            p = wp(ds)
            star = " ★" if (lo > 0 or hi < 0) else ""
            print(f"{k:>2d} {dfc:11s} {n_s:>8d} {n_i:>9d} {np.mean(a_s):11.4f} "
                  f"{np.mean(a_i):12.4f} {m:+8.4f} [{lo:+.4f},{hi:+.4f}] {p:8.4f}{star}")
        print()


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
