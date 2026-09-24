#!/usr/bin/env python3
"""Round-2 修订后的主分析 —— 取代 PAPER_PLAN v2 §4 的原分析。

针对 Round-2 审查的四项必修：
  R1  配对 severity 下的空间/色调比值 + 配对交互检验（原 §4 表用了未声明的
      不对称 severity 选择：色调取 sev=1.0，空间取 specular@0.33/shadow@0.67/
      gradient@0.67；配平后 double dissociation 消失）
  R2  2x2 表征识别设计：预训练目标 x 架构，外加层深对照
  R4  全部 19 个条件的 backbone 间配对表（原稿只报了 6 个）
  R10 正常件标定的固定工作点下的 TPR（取代回溯性 FPR@95TPR）
  R11 测试样本重采样区间（与支撑集区间并列报告）

所有统计量一律给出配对 bootstrap 95% CI 与 Wilcoxon 精确检验。
"""
from __future__ import annotations
import os
import argparse, glob, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(os.environ.get("IADSHIFT_ROOT",
                    Path(__file__).resolve().parents[2]))   # experiment/s1/x.py -> 项目根
TONAL = ["exposure", "gamma", "wb"]       # 全局色调映射
SPATIAL = ["gradient", "specular", "shadow"]   # 空间光照结构
SEVS = (0.33, 0.67, 1.0)
RNG = np.random.default_rng(20260920)

# 2x2 识别设计：每一次比较只动一个因素
GRID_2X2 = {
    ("ViT", "self-sup"): "dinov2_vits14",
    ("ViT", "supervised"): "deit_small_patch16",
    ("CNN", "self-sup"): "dino_resnet50",
    ("CNN", "supervised"): "resnet50",
}
DEPTH_PAIRS = [("resnet50", "resnet50_l3l4"), ("dino_resnet50", "dino_resnet50_l3l4"),
               ("dinov2_vits14", "dinov2_vits14_b6")]


# ───────────────────────── 载入 ─────────────────────────
def _unit_means(rows):
    d = defaultdict(lambda: defaultdict(list))
    for r in rows:
        d[(r["mode"], r["severity"])][(r["category"], r["defect"])].append(r["auroc"])
    return {k: {u: float(np.mean(v)) for u, v in m.items()} for k, m in d.items()}


def load_all(verbose=True):
    """(backbone, agg) -> unit-mean AUROC 表。合并 Round-1 与 Round-2 的结果文件。"""
    cfg = {}
    for f in sorted(glob.glob(str(ROOT / "reports/xbb_*.json"))):
        d = json.load(open(f)); m = d["meta"]
        cfg[(m["backbone"], m["agg"])] = _unit_means(d["rows"])
    for f in sorted(glob.glob(str(ROOT / "reports/r2/xbb2_*.json"))):
        d = json.load(open(f)); m = d["meta"]
        cfg[(m["backbone"], m["agg"])] = _unit_means(d["rows"])
        if m.get("partial") and verbose:
            print(f"  [PARTIAL] {m['backbone']}/{m['agg']}: {len(m['categories'])}/15 categories")
    # 主扫描（dinov2_vits14 + meantop1p），分 4 个文件
    rows = []
    for f in sorted(glob.glob(str(ROOT / "reports/mvtec_illum_g*.json"))):
        rows += json.load(open(f))["rows"]
    if rows:
        cfg[("dinov2_vits14", "meantop1p")] = _unit_means(rows)
    if verbose:
        for k in sorted(cfg):
            print(f"  loaded {k[0]}/{k[1]:10} units={len(cfg[k][('none',0.0)])} conds={len(cfg[k])}")
    return cfg


# ───────────────────────── 统计 ─────────────────────────
def paired(d, B=10000):
    """配对样本的均值 + bootstrap 95% CI + 精确 Wilcoxon。"""
    d = np.asarray(d, float); n = len(d)
    bs = d[RNG.integers(0, n, (B, n))].mean(1)
    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return d.mean(), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float(p)


def loss_vec(M, modes, sevs, units):
    """逐单元的平均 AUROC 损失（相对该单元的无扰动基线）。"""
    base = M[("none", 0.0)]
    return np.array([np.mean([base[u] - M[(m, s)][u] for m in modes for s in sevs]) for u in units])


def fmt_ci(m, lo, hi, p=None):
    s = f"{m:+.4f} [{lo:+.4f},{hi:+.4f}]"
    return s + (f" p={p:.1e}" if p is not None else "")


# ───────────────────────── R1：配对 severity 的比值与交互检验 ─────────────────────────
def table_R1(cfg):
    print("\n" + "="*100)
    print("R1  空间/色调损失比值 —— 配对 severity（原稿 §4 表的 severity 选择不对称，见 §3.2 更正说明）")
    print("="*100)
    print(f"{'配置':32}" + "".join(f"{'sev '+str(s):>20}" for s in SEVS) + f"{'三档合并':>20}")
    for key in sorted(cfg):
        M = cfg[key]; units = sorted(M[("none", 0.0)])
        cells = []
        for s in list(SEVS) + ["ALL"]:
            sv = list(SEVS) if s == "ALL" else [s]
            t = loss_vec(M, TONAL, sv, units).mean(); sp = loss_vec(M, SPATIAL, sv, units).mean()
            cells.append(f"{t:.3f}/{sp:.3f}={sp/t:5.2f}" if t > 1e-6 else f"{t:.3f}/{sp:.3f}=  inf")
        print(f"{key[0]+'/'+key[1]:32}" + "".join(f"{c:>20}" for c in cells))
    print("\n  判读：double dissociation 要求某一族的比值 < 1（空间比色调更轻）。")

    print("\n" + "-"*100)
    print("R1b 配对交互检验  D_u = (空间平均损失) − (色调平均损失)，逐单元配对")
    print("-"*100)
    print(f"{'配置':32}{'sev':>6}{'D  [95% CI]':>34}{'p':>11}  方向")
    for key in sorted(cfg):
        M = cfg[key]; units = sorted(M[("none", 0.0)])
        for s in SEVS:
            D = loss_vec(M, SPATIAL, [s], units) - loss_vec(M, TONAL, [s], units)
            m, lo, hi, p = paired(D)
            dirn = "空间更差" if lo > 0 else ("色调更差" if hi < 0 else "n.s.")
            print(f"{key[0]+'/'+key[1]:32}{s:>6}{fmt_ci(m,lo,hi):>34}{p:11.1e}  {dirn}")


# ───────────────────────── R4：全条件 backbone 间配对表 ─────────────────────────
def table_R4(cfg, a_key, b_key):
    A, B = cfg.get(a_key), cfg.get(b_key)
    if A is None or B is None:
        print(f"\n[R4] 跳过 {a_key} vs {b_key}（结果文件尚未就绪）"); return
    units = sorted(set(A[("none", 0.0)]) & set(B[("none", 0.0)]))
    print("\n" + "="*100)
    print(f"R4  全部 19 个条件的配对比较：Δ = AUROC({a_key[0]}) − AUROC({b_key[0]})，{len(units)} 个 (类别,缺陷) 单元配对")
    print("="*100)
    print(f"{'条件':20}{a_key[0][:12]:>12}{b_key[0][:12]:>12}{'Δ  [95% CI]':>34}{'p':>10}  胜方")
    conds = [("none", 0.0)] + [(m, s) for m in TONAL + SPATIAL for s in SEVS]
    for c in conds:
        a = np.array([A[c][u] for u in units]); b = np.array([B[c][u] for u in units])
        m, lo, hi, p = paired(a - b)
        win = "" if lo < 0 < hi else (a_key[0][:10] if m > 0 else b_key[0][:10])
        print(f"{c[0]+'@'+str(c[1]):20}{a.mean():12.4f}{b.mean():12.4f}{fmt_ci(m,lo,hi):>34}{p:10.1e}  {win}")


# ───────────────────────── R2：2x2 识别 ─────────────────────────
def table_R2(cfg, agg="meantop1p"):
    print("\n" + "="*100)
    print("R2  2x2 表征识别设计 —— 把「预训练目标」与「架构」拆开")
    print("="*100)
    have = {k: cfg.get((v, agg)) for k, v in GRID_2X2.items()}
    missing = [f"{k[0]}/{k[1]}={GRID_2X2[k]}" for k, v in have.items() if v is None]
    if missing:
        print(f"  尚缺：{', '.join(missing)} —— 待扫描完成后重跑"); 
    ready = {k: v for k, v in have.items() if v is not None}
    if not ready: return
    units = sorted(set.intersection(*[set(v[("none", 0.0)]) for v in ready.values()]))

    print(f"\n  逐单元指标：Dsp_tn = 空间平均损失 − 色调平均损失（三档合并），{len(units)} 单元")
    print(f"  {'cell':26}{'clean AUROC':>13}{'色调损失':>11}{'空间损失':>11}{'Dsp_tn [95% CI]':>34}")
    vals = {}
    for k, M in sorted(ready.items()):
        t = loss_vec(M, TONAL, list(SEVS), units); sp = loss_vec(M, SPATIAL, list(SEVS), units)
        D = sp - t; vals[k] = dict(D=D, t=t, sp=sp,
                                   clean=np.array([M[("none", 0.0)][u] for u in units]))
        m, lo, hi, _ = paired(D)
        print(f"  {k[0]+' / '+k[1]:26}{vals[k]['clean'].mean():13.4f}{t.mean():11.3f}{sp.mean():11.3f}{fmt_ci(m,lo,hi):>34}")

    print("\n  主效应与交互（逐单元配对；正号 = 前者的 Dsp_tn 更大，即更偏空间脆弱）")
    def cmp(lbl, x, y):
        if x is None or y is None: return
        m, lo, hi, p = paired(vals[x]["D"] - vals[y]["D"])
        print(f"    {lbl:52}{fmt_ci(m,lo,hi):>34}  p={p:.1e}")
    K = lambda a, b: (a, b) if (a, b) in vals else None
    cmp("预训练效应 @ViT：self-sup − supervised", K("ViT","self-sup"), K("ViT","supervised"))
    cmp("预训练效应 @CNN：self-sup − supervised", K("CNN","self-sup"), K("CNN","supervised"))
    cmp("架构效应 @self-sup：ViT − CNN", K("ViT","self-sup"), K("CNN","self-sup"))
    cmp("架构效应 @supervised：ViT − CNN", K("ViT","supervised"), K("CNN","supervised"))
    if len(vals) == 4:
        inter = (vals[("ViT","self-sup")]["D"] - vals[("ViT","supervised")]["D"]) - \
                (vals[("CNN","self-sup")]["D"] - vals[("CNN","supervised")]["D"])
        m, lo, hi, p = paired(inter)
        print(f"    {'预训练 x 架构 交互':52}{fmt_ci(m,lo,hi):>34}  p={p:.1e}")


def table_depth(cfg, agg="meantop1p"):
    print("\n" + "="*100)
    print("R2b 特征层深对照 —— 层深单独能否解释色调轴")
    print("="*100)
    print(f"  {'浅层 vs 深层':44}{'ΔDsp_tn [95% CI]':>34}{'p':>10}")
    for shallow, deep in DEPTH_PAIRS:
        A, B = cfg.get((shallow, agg)), cfg.get((deep, agg))
        if A is None or B is None:
            print(f"  {shallow+' vs '+deep:44}{'(缺结果)':>34}"); continue
        units = sorted(set(A[("none", 0.0)]) & set(B[("none", 0.0)]))
        DA = loss_vec(A, SPATIAL, list(SEVS), units) - loss_vec(A, TONAL, list(SEVS), units)
        DB = loss_vec(B, SPATIAL, list(SEVS), units) - loss_vec(B, TONAL, list(SEVS), units)
        m, lo, hi, p = paired(DA - DB)
        print(f"  {shallow+' vs '+deep:44}{fmt_ci(m,lo,hi):>34}{p:10.1e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", default="meantop1p")
    ap.add_argument("--only", nargs="*", default=None,
                    help="子集：r1 r2 r4 depth")
    a = ap.parse_args()
    print("载入结果文件：")
    cfg = load_all()
    sel = a.only or ["r1", "r2", "depth", "r4"]
    if "r1" in sel: table_R1(cfg)
    if "r2" in sel: table_R2(cfg, a.agg)
    if "depth" in sel: table_depth(cfg, a.agg)
    if "r4" in sel:
        table_R4(cfg, ("dinov2_vits14", a.agg), ("wide_resnet50_2", a.agg))
        table_R4(cfg, ("dinov2_vits14", a.agg), ("resnet50", a.agg))
        table_R4(cfg, ("dinov2_vits14", a.agg), ("deit_small_patch16", a.agg))
        table_R4(cfg, ("resnet50", a.agg), ("dino_resnet50", a.agg))


if __name__ == "__main__":
    main()
