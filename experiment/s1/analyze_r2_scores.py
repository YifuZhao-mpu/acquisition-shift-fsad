#!/usr/bin/env python3
"""基于逐图异常分数的分析 —— Round-2 审查 R10 / R11。

R10 部署口径：原稿用 FPR@95TPR，其阈值由**测试异常**回溯选出（原 §9.5 自承），
     且在 AUROC≈0.6 时 FPR@95TPR≈0.93 只是低 AUROC 的算术复述。
     本模块改用真实部署协议：
       安装期 —— 在**标称条件**下的一半正常件上，把阈值定到目标 FPR；
       运行期 —— 产线漂移后，在另一半正常件与全部异常件上量 FPR 与 TPR。
     该协议不使用任何测试异常来定阈值，且直接对应产线验收口径。

R11 两类不确定性并列：
     (a) 支撑集变异 —— 重采样 seed（原稿唯一报告的那类）；
     (b) 测试样本变异 —— 分层重采样测试图（原稿 §7 用来质疑 AeBAD 的那类）。
     §9.4 承认二者口径矛盾；本模块同时给出，矛盾消解。
"""
from __future__ import annotations
import os
import argparse, glob, json
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("IADSHIFT_ROOT",
                    Path(__file__).resolve().parents[2]))   # experiment/s1/x.py -> 项目根
RNG = np.random.default_rng(20260920)


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0: return np.nan
    r = np.empty(len(s)); order = np.argsort(s, kind="mergesort")
    ss = s[order]; i = 0; rank = np.empty(len(s))
    while i < len(s):                      # 平均秩，处理并列
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]: j += 1
        rank[i:j + 1] = (i + j) / 2.0 + 1.0; i = j + 1
    r[order] = rank
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def load_scores(path):
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["__meta__"][0]))
    cats = sorted({k.split("/")[0] for k in z.files if "/" in k})
    out = {}
    for c in cats:
        out[c] = dict(keys=z[f"{c}/keys"], cls=z[f"{c}/cls"],
                      conds=[tuple(x.split("|")) for x in z[f"{c}/conds"]],
                      scores=z[f"{c}/scores"])          # (n_cond, n_seed, n_img)
    return meta, out


# ─────────────────── R10：安装期标定 + 运行期漂移 ───────────────────
def deployment_table(data, fpr_targets=(0.10, 0.05)):
    """标称条件下用一半正常件定阈值，漂移后在另一半正常件 + 全部异常件上评估。"""
    rows = {}
    for cat, d in data.items():
        cls, S = d["cls"], d["scores"]
        conds = d["conds"]
        ci_clean = next(i for i, c in enumerate(conds) if c[0] == "none")
        good = np.flatnonzero(cls == "good"); bad = np.flatnonzero(cls != "good")
        if len(good) < 8 or len(bad) == 0: continue
        n_seed = S.shape[1]
        for seed in range(n_seed):
            g = good.copy(); np.random.default_rng(1000 + seed).shuffle(g)
            cal, ev = g[: len(g) // 2], g[len(g) // 2:]
            for tgt in fpr_targets:
                if tgt < 1.0 / len(cal):        # 标定样本无法分辨该分位数 -> 不报
                    rows.setdefault(("__skipped__", 0.0, tgt), []).append((cat, len(cal)))
                    continue
                # 安装期：标称条件下 cal 正常件的 (1-tgt) 分位
                tau = np.quantile(S[ci_clean, seed, cal], 1.0 - tgt)
                for ic, c in enumerate(conds):
                    sc = S[ic, seed]
                    key = (c[0], float(c[1]), tgt)
                    rows.setdefault(key, []).append(
                        (float((sc[ev] > tau).mean()), float((sc[bad] > tau).mean())))
    return rows


def print_deployment(rows, label):
    print("\n" + "=" * 92)
    print(f"R10  部署口径 —— 标称条件下标定阈值，漂移后的实际表现 [{label}]")
    print("=" * 92)
    for tgt in sorted({k[2] for k in rows}, reverse=True):
        sk = rows.get(("__skipped__", 0.0, tgt), [])
        note = f"；{len(set(c for c,_ in sk))} 个类别标定样本不足以分辨该分位数，已排除" if sk else ""
        print(f"\n  安装期目标误报率 FPR = {tgt:.1%}（阈值只用标称条件下的正常件定，不碰测试异常）{note}")
        print(f"  {'条件':20}{'实际 FPR':>12}{'TPR':>10}   {'解读':>0}")
        for c in [("none", 0.0)] + [(m, s) for m in ["gamma", "wb", "exposure",
                                                     "gradient", "shadow", "specular"]
                                    for s in (0.33, 0.67, 1.0)]:
            k = (c[0], c[1], tgt)
            if k not in rows or c[0] == "__skipped__": continue
            v = np.array(rows[k]); fpr, tpr = v[:, 0].mean(), v[:, 1].mean()
            note = ""
            if c != ("none", 0.0):
                note = "误报失控" if fpr > 0.2 else ("误报升高" if fpr > 3 * tgt else "可用")
            print(f"  {c[0]+'@'+str(c[1]):20}{fpr:12.4f}{tpr:10.4f}   {note}")


# ─────────────────── S8：漂移后重新标定阈值的挽回量 ───────────────────
def recalibration_gain(data, fpr_targets=(0.10,)):
    """对比三种运维策略在漂移条件下的表现：
       (a) install  —— 沿用安装期阈值（现状）
       (b) recal    —— 用漂移后的**正常件**重新标定阈值（纯软件，零硬件成本）
       (c) oracle   —— 漂移条件下的 AUROC（阈值无关的上界参考）
    回答审查 S8：「换 backbone」与「重新标定阈值」哪个更值。"""
    rows = {}
    for cat, d in data.items():
        cls, S, conds = d["cls"], d["scores"], d["conds"]
        ci = next(i for i, c in enumerate(conds) if c[0] == "none")
        good = np.flatnonzero(cls == "good"); bad = np.flatnonzero(cls != "good")
        if len(good) < 8 or len(bad) == 0: continue
        for seed in range(S.shape[1]):
            g = good.copy(); np.random.default_rng(1000 + seed).shuffle(g)
            cal, ev = g[: len(g) // 2], g[len(g) // 2:]
            for tgt in fpr_targets:
                if tgt < 1.0 / len(cal): continue
                tau0 = np.quantile(S[ci, seed, cal], 1.0 - tgt)
                for ic, c in enumerate(conds):
                    sc = S[ic, seed]
                    tau1 = np.quantile(sc[cal], 1.0 - tgt)      # 用漂移后的正常件重标定
                    k = (c[0], float(c[1]), tgt)
                    rows.setdefault(k, []).append((
                        float((sc[ev] > tau0).mean()), float((sc[bad] > tau0).mean()),
                        float((sc[ev] > tau1).mean()), float((sc[bad] > tau1).mean())))
    return rows


def print_recal(rows, label):
    print("\n" + "=" * 92)
    print(f"S8  漂移后重新标定阈值的挽回量（目标 FPR=10%）[{label}]")
    print("=" * 92)
    print(f"  {'条件':20}{'FPR(沿用)':>11}{'TPR(沿用)':>11}{'FPR(重标定)':>13}{'TPR(重标定)':>13}{'ΔTPR':>9}")
    for c in [("none", 0.0)] + [(m, s) for m in ["gamma", "wb", "exposure",
                                                 "gradient", "shadow", "specular"]
                                for s in (0.33, 0.67, 1.0)]:
        k = (c[0], c[1], 0.10)
        if k not in rows: continue
        v = np.array(rows[k])
        print(f"  {c[0]+'@'+str(c[1]):20}{v[:,0].mean():11.4f}{v[:,1].mean():11.4f}"
              f"{v[:,2].mean():13.4f}{v[:,3].mean():13.4f}{v[:,3].mean()-v[:,1].mean():+9.4f}")
    # 分组小结：色调组 vs 空间组的重标定挽回率
    for gname, modes in [("色调组 (gamma/wb/exposure)", ["gamma", "wb", "exposure"]),
                         ("空间组 (gradient/shadow/specular)", ["gradient", "shadow", "specular"])]:
        vs = [np.array(rows[(m, s, 0.10)]) for m in modes for s in (0.33, 0.67, 1.0)
              if (m, s, 0.10) in rows]
        if not vs: continue
        base = np.array(rows[("none", 0.0, 0.10)])[:, 3].mean()
        tpr_r = np.mean([v[:, 3].mean() for v in vs]); fpr_r = np.mean([v[:, 2].mean() for v in vs])
        print(f"  [{gname}]  重标定后 FPR={fpr_r:.3f}  TPR={tpr_r:.3f}"
              f"  (无扰动基线 TPR={base:.3f}，缺口 {tpr_r-base:+.3f})")
    print("  → 判读：若重标定后 TPR 基本回到基线，该漂移是**阈值问题**（软件即可解决）；"
          "若 TPR 仍大幅低于基线，才是**表征问题**，换 backbone 才有意义。")


# ─────────────────── R11：两类不确定性 ───────────────────
def two_uncertainties(data, cond=("none", 0.0), B=2000):
    """对每个 (类别,缺陷) 单元，分别给出支撑集变异与测试样本变异的 95% CI 宽度。"""
    out = []
    for cat, d in data.items():
        cls, S, conds = d["cls"], d["scores"], d["conds"]
        try: ic = next(i for i, c in enumerate(conds)
                       if c[0] == cond[0] and float(c[1]) == cond[1])
        except StopIteration: continue
        good = np.flatnonzero(cls == "good")
        for dfc in sorted(set(cls) - {"good"}):
            bad = np.flatnonzero(cls == dfc)
            if len(bad) == 0 or len(good) == 0: continue
            idx = np.concatenate([good, bad]); y = np.r_[np.zeros(len(good)), np.ones(len(bad))]
            per_seed = np.array([auroc(y, S[ic, s, idx]) for s in range(S.shape[1])])
            # (a) 支撑集变异：重采样 seed
            bs_sup = per_seed[RNG.integers(0, len(per_seed), (B, len(per_seed)))].mean(1)
            # (b) 测试样本变异：分层重采样测试图（在 seed 平均分数上）
            mean_sc = S[ic, :, idx].mean(axis=1) if S[ic, :, idx].ndim == 2 else S[ic, 0, idx]
            bs_test = np.empty(B)
            for b in range(B):
                gi = RNG.integers(0, len(good), len(good)); bi = RNG.integers(0, len(bad), len(bad))
                sel = np.r_[gi, len(good) + bi]
                bs_test[b] = auroc(y[sel], mean_sc[sel])
            out.append(dict(cat=cat, defect=dfc, n_good=len(good), n_bad=len(bad),
                            auroc=float(per_seed.mean()),
                            w_support=float(np.percentile(bs_sup, 97.5) - np.percentile(bs_sup, 2.5)),
                            w_test=float(np.nanpercentile(bs_test, 97.5) - np.nanpercentile(bs_test, 2.5))))
    return out


def print_two_unc(rows, label):
    print("\n" + "=" * 92)
    print(f"R11  两类不确定性的 95% CI 宽度对比（无扰动基线）[{label}]")
    print("=" * 92)
    ws = np.array([r["w_support"] for r in rows]); wt = np.array([r["w_test"] for r in rows])
    print(f"  单元数 {len(rows)}；测试集规模 中位 {int(np.median([r['n_good']+r['n_bad'] for r in rows]))}")
    print(f"  支撑集变异  CI 宽度：中位 {np.median(ws):.4f}   四分位 [{np.percentile(ws,25):.4f}, {np.percentile(ws,75):.4f}]")
    print(f"  测试样本变异 CI 宽度：中位 {np.median(wt):.4f}   四分位 [{np.percentile(wt,25):.4f}, {np.percentile(wt,75):.4f}]")
    print(f"  比值（测试/支撑）中位：{np.median(wt/np.maximum(ws,1e-9)):.2f}")
    print("  → 原稿只报支撑集区间，会系统性低估单元级估计的不确定性；两者须并列。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", default=None)
    a = ap.parse_args()
    files = a.scores or sorted(glob.glob(str(ROOT / "reports/r2/scores_*.npz")))
    if not files:
        print("尚无分数文件（reports/r2/scores_*.npz）——扫描完成后再跑。"); return
    for f in files:
        meta, data = load_scores(f)
        lab = f"{meta['backbone']}/{meta.get('agg','?')}"
        print_deployment(deployment_table(data), lab)
        print_recal(recalibration_gain(data), lab)
        print_two_unc(two_uncertainties(data), lab)


if __name__ == "__main__":
    main()
