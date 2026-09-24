#!/usr/bin/env python3
"""受控支撑来源实验（Round-1 审稿指定的设计）。

回答的识别性问题：C2 的改善究竟来自 **bank 内条件多样性**，
还是仅仅因为把 background 参考换成了更"适配"的来源？
原设计只有 single(=6×background) vs mixed，无法区分二者。

设计：
  k = 6（可精确平衡 2+2+2）
  32 次随机化 replicate；每个 replicate 从三个来源池各抽 6 张
  四种 bank： 6B / 6I / 6V / 2B+2I+2V（混合子集取自同一批已抽样本）
  检测器、旋转增强、打分、测试集全部固定；主实验**关闭遮罩**

核心估计量（对每个测试域 d）：
  G_d = A_d(2+2+2) − [ A_d(6B) + A_d(6I) + A_d(6V) ] / 3
  另单独报告 混合 vs 各纯来源，以判别"来源选择"假说。
"""
import os
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from fscache import Encoder, TestCache, per_ref_mindist, bank_scores
import importlib.util
_s = importlib.util.spec_from_file_location('an', str(Path(__file__).parent / 'analyze.py'))
an = importlib.util.module_from_spec(_s); _s.loader.exec_module(an)

ROOT = Path(os.environ.get("IADSHIFT_ROOT",
                    Path(__file__).resolve().parents[2]))   # experiment/s1/x.py -> 项目根
DOMAINS = ["same", "illumination", "view", "background"]
DEFECTS = ["ablation", "breakdown", "fracture", "groove"]
SOURCES = ["background", "illumination", "view"]


def list_png(d: Path):
    return sorted(p for p in d.iterdir() if p.suffix == ".png") if d.is_dir() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--replicates", type=int, default=32)
    ap.add_argument("--backbone", default="dinov2_vits14")
    ap.add_argument("--resolution", type=int, default=448)
    ap.add_argument("--masking", action="store_true", help="开启前景遮罩（主实验关闭）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--equalize-pools", type=int, default=0, metavar="N",
                    help="Round-2 审查 R12：把三个来源池各随机子采样到 N 张后再抽 replicate。"
                         "原实验的池规模为 background 265 / illumination 62 / view 194，"
                         "6I 抽自最小的池，32 次 replicate 间重叠更多，其有效多样性天然更低 —— "
                         "该混淆此前被声明为「内生于数据集、无法消除」，实际上等量子采样即可消除。"
                         "N=0 表示不做等量化（复现原实验）。")
    a = ap.parse_args()
    assert a.k % 3 == 0, "k 必须能被 3 整除以构造平衡混合"
    per = a.k // 3

    src_root = ROOT / "data/AeBAD/AeBAD_S"
    dom_root = ROOT / "data/AeBAD_S_domains/single"     # 测试集布局四协议相同，任取其一
    pools = {s: list_png(src_root / "train/good" / s) for s in SOURCES}
    print("来源池(原始): " + ", ".join(f"{s}={len(pools[s])}" for s in SOURCES))
    if a.equalize_pools:
        N = a.equalize_pools
        eq_rng = np.random.default_rng(90210 + a.seed)   # 与 replicate 抽样的 rng 分离
        for s in SOURCES:
            assert len(pools[s]) >= N, f"来源池 {s} 只有 {len(pools[s])} 张，不足 {N}"
            idx = np.sort(eq_rng.choice(len(pools[s]), N, replace=False))
            pools[s] = [pools[s][i] for i in idx]
        print(f"来源池(等量化到 N={N}): " + ", ".join(f"{s}={len(pools[s])}" for s in SOURCES))

    enc = Encoder(a.backbone, a.resolution)

    # ── 测试集只编码一次，四个域一起 ──
    items, dom_of, cls_of = [], [], []
    for dom in DOMAINS:
        obj = dom_root / f"blade_{dom}"
        for cls in ["good"] + DEFECTS:
            for p in list_png(obj / "test" / cls):
                items.append((f"{dom}|{cls}|{p.name}", p, 0 if cls == "good" else 1))
                dom_of.append(dom); cls_of.append(cls)
    dom_of, cls_of = np.array(dom_of), np.array(cls_of)
    print(f"测试图总数 {len(items)}（一次性编码）")
    cache = TestCache(enc, items, masking=a.masking)
    y = cache.labels

    rng = np.random.default_rng(a.seed)
    BANKS = {"6B": [("background", a.k)], "6I": [("illumination", a.k)], "6V": [("view", a.k)],
             "MIX": [("background", per), ("illumination", per), ("view", per)]}
    rows = []
    for rep in range(a.replicates):
        pick = {s: list(rng.choice(len(pools[s]), a.k, replace=False)) for s in SOURCES}
        # 同一批抽样内缓存 per-ref 距离图 —— 四种 bank 共享
        maps = {}
        for s in SOURCES:
            for idx in pick[s]:
                maps[(s, idx)] = per_ref_mindist(
                    cache, enc.encode_reference(pools[s][idx], masking_ref=False, rotation=True))
        for bname, spec in BANKS.items():
            members = [maps[(s, i)] for s, n in spec for i in pick[s][:n]]
            sc = bank_scores(cache, members)
            for dom in DOMAINS:
                m = dom_of == dom
                rows.append(dict(rep=rep, bank=bname, domain=dom, scope="all",
                                 auroc=an.auroc(y[m], sc[m]), fpr95=an.fpr_at_tpr(y[m], sc[m]),
                                 n=int(m.sum())))
                for dfc in DEFECTS:                      # 按缺陷类型分层
                    mm = m & ((cls_of == "good") | (cls_of == dfc))
                    if (y[mm] == 1).sum() == 0: continue
                    rows.append(dict(rep=rep, bank=bname, domain=dom, scope=dfc,
                                     auroc=an.auroc(y[mm], sc[mm]),
                                     fpr95=an.fpr_at_tpr(y[mm], sc[mm]), n=int(mm.sum())))
        if (rep + 1) % 4 == 0: print(f"  replicate {rep+1}/{a.replicates} 完成", flush=True)

    meta = dict(k=a.k, replicates=a.replicates, backbone=a.backbone,
                resolution=a.resolution, masking=a.masking, seed=a.seed,
                pool_sizes={s: len(pools[s]) for s in SOURCES}, n_test=len(items),
                equalize_pools=a.equalize_pools)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(meta=meta, rows=rows), open(a.out, "w"), indent=1)
    print(f"\n写入 {a.out}  （{len(rows)} 行）")


if __name__ == "__main__":
    main()
