#!/usr/bin/env python3
"""把 AeBAD-S 的真实测试域放到与合成扰动相同的物理坐标上 —— 论文 §IV-E。

动机：合成网格给出的分解是「监督 CNN 怕全局色调、自监督 ViT 怕空间光照」。
AeBAD 的 illumination 域上 CNN 反而垮得最惨，看似矛盾 —— 除非该真实域的
变化以**色调**成分为主。本脚本直接测量，把交叉检验变成可证伪的预测。

AeBAD 各域不是同一批拍摄、不逐像素配准，故只能比较**分布**而非配对差。
报告的是各域相对 same 域的分布位移：
  L_mean   平均亮度（0-255）
  L_std    图内亮度标准差（全局明暗结构的强度）
  chroma   CIELab a*b* 平面上离中性轴的平均距离（色度饱和度）
  ab_mean  a*、b* 的均值（色偏方向 —— 色温漂移的直接指标）
  hi_area  亮度 > 均值+2σ 的像素占比（高光面积）
  contrast 31x31 局部对比度的中位数（纹理可见性）
"""
from __future__ import annotations
import os
import json
from pathlib import Path
import numpy as np, cv2

ROOT = Path(os.environ.get("IADSHIFT_ROOT",
                    Path(__file__).resolve().parents[2]))   # experiment/s1/x.py -> 项目根
BASE = ROOT / "data/AeBAD_S_domains/single"
DOMS = ["same", "illumination", "view", "background"]
N_PER = 60


def stats(p):
    img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if max(h, w) > 768:
        s = 768 / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    a, b = lab[..., 1] - 128.0, lab[..., 2] - 128.0
    m = cv2.blur(L / 255.0, (31, 31)); m2 = cv2.blur((L / 255.0) ** 2, (31, 31))
    loc = np.sqrt(np.maximum(m2 - m * m, 0))
    thr = L.mean() + 2 * L.std()
    return dict(L_mean=float(L.mean()), L_std=float(L.std()),
                chroma=float(np.sqrt(a * a + b * b).mean()),
                a_mean=float(a.mean()), b_mean=float(b.mean()),
                hi_area=float((L > thr).mean() * 100),
                contrast=float(np.median(loc)))


def main():
    out = {}
    for dom in DOMS:
        ps = sorted((BASE / f"blade_{dom}/test/good").glob("*.png"))[:N_PER]
        acc = [stats(p) for p in ps]
        out[dom] = {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}
        out[dom]["n"] = len(ps)
    keys = ["L_mean", "L_std", "chroma", "a_mean", "b_mean", "hi_area", "contrast"]
    print(f"AeBAD-S normal test images, {N_PER} per domain\n")
    print(f"{'domain':16}" + "".join(f"{k:>10}" for k in keys))
    for dom in DOMS:
        print(f"{dom:16}" + "".join(f"{out[dom][k]:10.2f}" for k in keys))
    print(f"\n相对 same 域的位移：")
    print(f"{'domain':16}" + "".join(f"{k:>10}" for k in keys))
    for dom in DOMS[1:]:
        print(f"{dom:16}" + "".join(f"{out[dom][k]-out['same'][k]:+10.2f}" for k in keys))
    s = out["same"]
    dL = abs(out["illumination"]["L_mean"] - s["L_mean"]) / max(s["L_mean"], 1) * 100
    dC = np.hypot(out["illumination"]["a_mean"] - s["a_mean"],
                  out["illumination"]["b_mean"] - s["b_mean"])
    dH = out["illumination"]["hi_area"] - s["hi_area"]
    print(f"\n判读（illumination vs same）：")
    print(f"  相对平均亮度位移 ΔL = {dL:.1f}%   色偏 ΔC(ab) = {dC:.2f}"
          f"   高光面积变化 = {dH:+.2f} 个百分点")
    out["_summary"] = dict(illum_dL_pct=dL, illum_dC_ab=float(dC), illum_dhi_area=dH)
    p = ROOT / "reports/r2/aebad_physical.json"
    json.dump(out, open(p, "w"), indent=1)
    print(f"\n写入 {p}")


if __name__ == "__main__":
    main()
