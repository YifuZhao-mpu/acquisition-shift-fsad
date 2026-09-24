#!/usr/bin/env python3
"""论文图表生成 —— 输出到 paper/figures/*.pdf。

Fig.1 剂量-响应小倍数：横轴是**实测物理幅度** ΔL(%)（而非无量纲 severity），
       六个面板 = 六种扰动，上排色调组、下排空间组；每面板两条线 =
       自监督 ViT vs 监督 CNN。这张图同时承载三件事：
         (a) 每种扰动的剂量-响应形状；
         (b) 两族表征在每个条件下的相对位置（即「排序反转」）；
         (c) 等 severity ≠ 等物理幅度（各面板横轴跨度差异巨大）。

配色：dataviz 参考调色板 slot 1/2，已用 validate_palette.js 在 all-pairs
模式下通过全部六项检查（CVD ΔE 24.7、normal ΔE 33.6、对比度 ≥3:1）。
标识不依赖颜色单独承载：两条线另有 marker 形状与直接标注。
"""
from __future__ import annotations
import os
import glob, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get("IADSHIFT_ROOT",
                    Path(__file__).resolve().parents[2]))   # experiment/s1/x.py -> 项目根
FIG = ROOT / "paper/figures"
TONAL = ["gamma", "exposure", "wb"]; SPATIAL = ["gradient", "shadow", "specular"]
SEVS = (0.33, 0.67, 1.0)
C_VIT, C_CNN = "#2a78d6", "#eb6834"          # dataviz slot 1 / slot 2（已验证）
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
RNG = np.random.default_rng(20260920)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 6.8, "axes.labelsize": 6.8, "axes.titlesize": 7.2,
    "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "legend.fontsize": 6.8,
    "axes.edgecolor": GRID, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.labelcolor": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def _unit_means(rows):
    d = defaultdict(lambda: defaultdict(list))
    for r in rows:
        d[(r["mode"], r["severity"])][(r["category"], r["defect"])].append(r["auroc"])
    return {k: {u: float(np.mean(v)) for u, v in m.items()} for k, m in d.items()}


def load(backbone, agg="meantop1p"):
    for f in glob.glob(str(ROOT / "reports/r2/xbb2_*.json")) + glob.glob(str(ROOT / "reports/xbb_*.json")):
        d = json.load(open(f)); m = d["meta"]
        if m["backbone"] == backbone and m.get("agg") == agg:
            return _unit_means(d["rows"])
    if backbone == "dinov2_vits14" and agg == "meantop1p":
        rows = []
        for f in sorted(glob.glob(str(ROOT / "reports/mvtec_illum_g*.json"))):
            rows += json.load(open(f))["rows"]
        if rows: return _unit_means(rows)
    return None


def band(vals, B=4000):
    """73 个单元均值的 bootstrap 95% 区间。"""
    v = np.asarray(vals); n = len(v)
    bs = v[RNG.integers(0, n, (B, n))].mean(1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


NICE_FIG = {"dinov2_vits14": "DINOv2 ViT-S/14", "wide_resnet50_2": "WideResNet-50-2",
            "resnet50": "ResNet-50 (sup.)", "dino_resnet50": "DINO ResNet-50",
            "deit_small_patch16": "DeiT-S/16"}


def fig_dose_response(vit="dinov2_vits14", cnn="wide_resnet50_2", agg="meantop1p",
                      out="fig_dose_response.pdf"):
    global NAME_VIT, NAME_CNN
    NAME_VIT, NAME_CNN = NICE_FIG.get(vit, vit), NICE_FIG.get(cnn, cnn)
    P = json.load(open(ROOT / "reports/r2/perturb_physical.json"))["desc"]
    A, B_ = load(vit, agg), load(cnn, agg)
    if A is None or B_ is None:
        print(f"  skip {out}: missing {vit if A is None else cnn}"); return False
    units = sorted(set(A[("none", 0.0)]) & set(B_[("none", 0.0)]))

    fig, axes = plt.subplots(2, 3, figsize=(7.16, 2.85), sharey=True)
    for row, group, gname in [(0, TONAL, "global tone"), (1, SPATIAL, "spatial illumination")]:
        for col, mode in enumerate(group):
            ax = axes[row, col]
            xs = [0.0] + [P[f"{mode}|{s}"]["dL_mean"] for s in SEVS]
            for M, c, mk, lab in [(A, C_VIT, "o", NAME_VIT), (B_, C_CNN, "s", NAME_CNN)]:
                ys, los, his = [], [], []
                for k in [("none", 0.0)] + [(mode, s) for s in SEVS]:
                    v = [M[k][u] for u in units]
                    ys.append(np.mean(v)); lo, hi = band(v); los.append(lo); his.append(hi)
                ax.fill_between(xs, los, his, color=c, alpha=0.16, linewidth=0)
                ax.plot(xs, ys, color=c, lw=1.6, marker=mk, ms=3.4, mew=0.8,
                        mfc="white", mec=c, label=lab, clip_on=False, zorder=3)
            ax.axhline(0.5, color=GRID, lw=0.8, ls=(0, (3, 2)), zorder=1)
            ax.set_title(rf"$\mathrm{{{mode}}}$", pad=3, color=INK)
            ax.set_ylim(0.45, 1.0); ax.set_xlim(left=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color=GRID, lw=0.45, alpha=0.7); ax.set_axisbelow(True)
            if col == 0:
                ax.set_ylabel("image AUROC", color=INK)
                ax.text(-0.42, 0.5, gname, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=7.4, color=INK2)
            if row == 1:
                ax.set_xlabel(r"measured $\Delta L$ (%)", color=INK)
    # 直接标注（不让识别只靠颜色）+ 一个图例
    ax0 = axes[0, 0]          # gamma 面板下半部整片留白，标注不会压线
    ax0.plot([0.06, 0.16], [0.30, 0.30], transform=ax0.transAxes, color=C_VIT,
             lw=1.6, marker="o", ms=3.4, mew=0.8, mfc="white", mec=C_VIT, clip_on=False)
    ax0.text(0.19, 0.30, NAME_VIT, transform=ax0.transAxes, va="center",
             color=C_VIT, fontsize=6.8, weight="bold")
    ax0.plot([0.06, 0.16], [0.16, 0.16], transform=ax0.transAxes, color=C_CNN,
             lw=1.6, marker="s", ms=3.4, mew=0.8, mfc="white", mec=C_CNN, clip_on=False)
    ax0.text(0.19, 0.16, NAME_CNN, transform=ax0.transAxes, va="center",
             color=C_CNN, fontsize=6.8, weight="bold")
    axes[1, 0].annotate("chance", xy=(0.03, 0.5), xycoords=("axes fraction", "data"),
                        ha="left", va="bottom", color=INK2, fontsize=6.2)
    fig.subplots_adjust(wspace=0.12, hspace=0.48)
    fig.savefig(FIG / out)
    plt.close(fig)
    print(f"  wrote {out}")
    return True


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    fig_dose_response()
    # 2x2 的 ResNet 臂就绪后追加一张
    fig_dose_response(vit="dinov2_vits14", cnn="resnet50",
                      out="fig_dose_response_rn50.pdf")


if __name__ == "__main__":
    main()
