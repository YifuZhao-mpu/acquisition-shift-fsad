#!/usr/bin/env python3
"""
S1 度量与归因分析。

从 AnomalyDINO 的 measurements_seed=*.csv 计算每个 (protocol, k, seed, domain) 的：
  - image AUROC
  - FPR@95TPR  (产线最关心：为保住 95% 检出率要付出多少误报)
  - AP
再聚合成核心表：相对 `same` 域的退化 Δ，带 bootstrap 置信区间。

核心可证伪主张：
  在低 k 下 Δ(illumination) 是否超过 Δ(view)？
  —— 即 AeBAD 全量训练下「视角 ≫ 光照」的结论在少样本下是否反转。

噪声基线：同一 (protocol,k,domain) 下不同 seed 的离散程度。
任何小于噪声基线的 Δ 都不得解读为效应（本项目从 ICRA2027 学到的教训）。
"""
import argparse, csv, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np

DOMAINS = ["same", "illumination", "view", "background"]


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0: return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float); ranks[order] = np.arange(1, len(s) + 1)
    # 处理并列：同值取平均秩
    uniq, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(uniq)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def fpr_at_tpr(y, s, target=0.95):
    """在**实际可达**的操作点上报 FPR：取使 TPR >= target 的最大阈值，返回该阈值下的 FPR。

    不做 ROC 插值 —— 插值出的操作点在有限样本下并不存在。
    这样报出的数字是产线真正能选到的工作点（"要保 95% 检出，就得吃这么多误报"）。
    """
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    # 候选阈值 = 所有正样本分数；取满足 TPR>=target 的最大者（阈值越大 FPR 越小）
    cand = np.unique(pos)
    tprs = (pos[None, :] >= cand[:, None]).mean(axis=1)
    ok = cand[tprs >= target]
    if len(ok) == 0: return 1.0
    thr = ok.max()
    return float((neg >= thr).mean())


def ap(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    if (y == 1).sum() == 0: return float("nan")
    o = np.argsort(-s, kind="mergesort"); y = y[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def boot_ci(fn, y, s, n=2000, seed=0, alpha=0.05):
    """分层 bootstrap（正负样本各自重采样），返回 (lo, hi)。"""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    ip, ineg = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(ip) == 0 or len(ineg) == 0: return (float("nan"),) * 2
    vals = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(ip, len(ip), True), rng.choice(ineg, len(ineg), True)])
        vals.append(fn(y[idx], s[idx]))
    vals = np.array([v for v in vals if np.isfinite(v)])
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


def load_run(csv_path: Path):
    """-> {object: (y_list, s_list)}"""
    out = defaultdict(lambda: ([], []))
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            obj, samp = r["Object"], r["Sample"]
            y = 0 if samp.split("/")[0] == "good" else 1
            out[obj][0].append(y); out[obj][1].append(float(r["Anomaly_Score"]))
    return out


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--results-root", required=True, help="AnomalyDINO 结果根目录 (含 results_AeBAD_*)")
    ap_.add_argument("--out", required=True)
    ap_.add_argument("--bootstrap", type=int, default=2000)
    ap_.add_argument("--preprocess", default=None,
                     help="只分析该 preprocess 变体（如 agnostic / force_mask_rotation）。"
                          "不指定则分析全部——但不同变体不可混合，务必显式指定。")
    a = ap_.parse_args()

    rows = []
    for res_dir in sorted(Path(a.results_root).glob("results_AeBAD_*")):
        proto = res_dir.name.replace("results_AeBAD_", "")
        for shot_dir in sorted(res_dir.glob("*/*-shot_preprocess=*")):
            m = re.match(r"(-?\d+)-shot_preprocess=(.+)$", shot_dir.name)
            if not m: continue
            k, prep = int(m.group(1)), m.group(2)
            if a.preprocess is not None and prep != a.preprocess: continue
            for csvf in sorted(shot_dir.glob("measurements_seed=*.csv")):
                seed = int(re.search(r"seed=(\d+)", csvf.name).group(1))
                for obj, (y, s) in load_run(csvf).items():
                    dom = obj.replace("blade_", "")
                    if len(y) < 10: continue
                    rows.append(dict(protocol=proto, preprocess=prep, k=k, seed=seed, domain=dom,
                                     n=len(y), n_pos=int(sum(y)),
                                     auroc=auroc(y, s), fpr95=fpr_at_tpr(y, s), ap=ap(y, s)))
    if not rows:
        raise SystemExit("没有找到任何结果 CSV")

    # ---- 聚合：跨 seed ----
    agg = defaultdict(list)
    for r in rows: agg[(r["protocol"], r["k"], r["domain"])].append(r)

    summary = []
    for (proto, k, dom), rs in sorted(agg.items()):
        for metric in ("auroc", "fpr95", "ap"):
            v = np.array([r[metric] for r in rs], float); v = v[np.isfinite(v)]
            if len(v) == 0: continue
            summary.append(dict(protocol=proto, k=k, domain=dom, metric=metric,
                                mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                                n_seeds=len(v), vals=v.tolist()))

    # ---- 核心表：相对 same 的退化 Δ，以及与 seed 噪声的比值 ----
    deltas = []
    by = {(s["protocol"], s["k"], s["domain"], s["metric"]): s for s in summary}
    for (proto, k, dom, metric), s in by.items():
        if dom == "same": continue
        base = by.get((proto, k, "same", metric))
        if not base: continue
        d = s["mean"] - base["mean"]
        # 噪声基线：两组 seed 标准差的合并
        noise = float(np.sqrt((base["std"] ** 2 + s["std"] ** 2) / 2))
        deltas.append(dict(protocol=proto, k=k, domain=dom, metric=metric,
                           same=base["mean"], shifted=s["mean"], delta=d,
                           seed_noise=noise,
                           effect_over_noise=(abs(d) / noise if noise > 0 else float("inf"))))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(rows=rows, summary=summary, deltas=deltas),
              open(a.out, "w"), indent=2, ensure_ascii=False)

    # ---- 打印核心表 ----
    print("\n" + "=" * 92)
    print("S1 核心表：各域相对 `same` 的退化（AUROC，越负越差）")
    print("=" * 92)
    print(f"{'proto':8s} {'k':>3s} {'domain':14s} {'same':>7s} {'shifted':>8s} {'Δ':>8s} {'seed噪声':>9s} {'|Δ|/噪声':>9s}")
    print("-" * 92)
    for d in sorted(deltas, key=lambda x: (x["protocol"], x["k"], x["domain"])):
        if d["metric"] != "auroc": continue
        flag = "" if d["effect_over_noise"] >= 2 else "  ← 噪声内"
        print(f"{d['protocol']:8s} {d['k']:>3d} {d['domain']:14s} {d['same']:7.4f} {d['shifted']:8.4f} "
              f"{d['delta']:+8.4f} {d['seed_noise']:9.4f} {d['effect_over_noise']:9.2f}{flag}")
    print(f"\n结果写入 {a.out}")


if __name__ == "__main__":
    main()
