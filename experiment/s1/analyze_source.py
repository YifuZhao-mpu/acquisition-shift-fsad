#!/usr/bin/env python3
"""受控支撑来源实验的分析。

判别两个竞争假说：
  H_diversity : bank 内条件多样性本身有益  →  MIX 应优于**平衡的纯来源均值**
  H_source    : 只是换了更适配的来源       →  某个纯来源（很可能 6I）单独即可复现 MIX 的收益

主估计量（逐 replicate 配对）：
  G_d = A_d(MIX) − [A_d(6B)+A_d(6I)+A_d(6V)]/3
另报告 MIX − 6B / MIX − 6I / MIX − 6V。
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from paired import wilcoxon_signed_rank_p as wp

DOMAINS = ["same", "illumination", "view", "background"]
DEFECTS = ["ablation", "breakdown", "fracture", "groove"]


def ci(ds, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    bs = np.array([rng.choice(ds, len(ds), True).mean() for _ in range(n)])
    return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def main(path, scope="all"):
    d = json.load(open(path)); meta = d["meta"]
    print("=" * 104)
    print(f"受控支撑来源实验  k={meta['k']}  replicates={meta['replicates']}  "
          f"backbone={meta['backbone']}  masking={meta['masking']}  scope={scope}")
    print("=" * 104)

    A = defaultdict(dict)      # (rep, domain) -> {bank: auroc}
    for r in d["rows"]:
        if r["scope"] != scope: continue
        A[(r["rep"], r["domain"])][r["bank"]] = r["auroc"]

    # ── 绝对水平 ──
    print(f"\n{'domain':13s} {'6B':>8s} {'6I':>8s} {'6V':>8s} {'MIX':>8s}   (各 replicate 均值 AUROC)")
    print("-" * 60)
    for dom in DOMAINS:
        v = {b: np.mean([A[(r, dom)][b] for r in range(meta["replicates"]) if b in A[(r, dom)]])
             for b in ("6B", "6I", "6V", "MIX")}
        print(f"{dom:13s} " + " ".join(f"{v[b]:8.4f}" for b in ("6B", "6I", "6V", "MIX")))

    # ── 主估计量 G_d ──
    print(f"\n{'='*104}\n主估计量 G_d = MIX − 平衡纯来源均值   （>0 表示多样性本身有益）\n{'='*104}")
    print(f"{'domain':13s} {'G_d':>9s} {'sd':>8s} {'95%CI':>20s} {'p':>9s}  判定")
    print("-" * 104)
    out = {}
    for dom in DOMAINS:
        ds = []
        for r in range(meta["replicates"]):
            a = A[(r, dom)]
            if not all(b in a for b in ("6B", "6I", "6V", "MIX")): continue
            ds.append(a["MIX"] - (a["6B"] + a["6I"] + a["6V"]) / 3)
        if len(ds) < 3: continue
        ds = np.array(ds); lo, hi = ci(ds); p = wp(ds)
        verd = "多样性有益" if lo > 0 else ("多样性有害" if hi < 0 else "不可区分")
        print(f"{dom:13s} {ds.mean():+9.4f} {ds.std(ddof=1):8.4f} [{lo:+.4f},{hi:+.4f}] {p:9.4f}  {verd}")
        out[dom] = dict(G=float(ds.mean()), lo=lo, hi=hi, p=p, verdict=verd)

    # ── 判别 H_source：MIX vs 各纯来源 ──
    print(f"\n{'='*104}\n判别「来源选择」假说：MIX 与各单一来源的配对差\n{'='*104}")
    print(f"{'domain':13s} {'vs 6B':>22s} {'vs 6I':>22s} {'vs 6V':>22s}")
    print("-" * 104)
    for dom in DOMAINS:
        cells = []
        for b in ("6B", "6I", "6V"):
            ds = [A[(r, dom)]["MIX"] - A[(r, dom)][b] for r in range(meta["replicates"])
                  if b in A[(r, dom)] and "MIX" in A[(r, dom)]]
            ds = np.array(ds); lo, hi = ci(ds)
            mark = "★" if (lo > 0 or hi < 0) else " "
            cells.append(f"{ds.mean():+7.4f}[{lo:+.3f},{hi:+.3f}]{mark}")
        print(f"{dom:13s} " + " ".join(f"{c:>22s}" for c in cells))
    print("\n★ = 配对 95% CI 不含 0。若 'vs 6I' 不显著而 'vs 6B' 显著 → 支持「来源选择」而非「多样性」。")
    return out


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3] or ["all"]))
