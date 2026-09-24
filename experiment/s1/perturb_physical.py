#!/usr/bin/env python3
"""把 severity 参数映射到可测量的物理量 —— Round-2 审查 S1。

问题：severity 的 0.33/0.67/1.0 是参数幅度，工程读者无法判断自己产线
处在哪一档。exposure 与 wb 尚有物理含义（±stop、通道增益），但三个
空间模式没有。

本脚本在真实 MVTec 图像上直接量出每个 (mode, severity) 的可测描述符：
  dL_mean   平均相对亮度变化 |L'-L|/L 的均值（%）
  dL_p95    同上的 95 分位（%）——刻画最受影响区域
  area_10   相对亮度变化 >10% 的像素占比（%）——「受影响面积」
  area_25   相对亮度变化 >25% 的像素占比（%）
  sat_hi    进入上饱和（>250/255）的像素占比（%）——高光溢出
  contrast  局部对比度比值 C'/C（31x31 窗口内标准差之比的中位数）
            <1 表示局部纹理对比被压缩，缺陷可见性下降
这些量全部可由现场用一台相机与一块标准板测出，因此可以把论文的
severity 轴对接到产线实测。
"""
from __future__ import annotations
import os
import json, sys
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).parent))
import illum

ROOT = Path(os.environ.get("IADSHIFT_ROOT",
                    Path(__file__).resolve().parents[2]))   # experiment/s1/x.py -> 项目根
MV = ROOT / "data/mvtec_anomaly_detection"
SEVS = (0.33, 0.67, 1.0)


def local_std(x, k=31):
    m = cv2.blur(x, (k, k))
    m2 = cv2.blur(x * x, (k, k))
    return np.sqrt(np.maximum(m2 - m * m, 0))


def descriptors(img, out):
    a = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    b = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY).astype(np.float32)
    rel = np.abs(b - a) / np.maximum(a, 1.0)
    sa, sb = local_std(a / 255.0), local_std(b / 255.0)
    ok = sa > 0.01                                   # 只在有纹理处比对比度
    # 色度位移：CIELab 的 a*b* 平面欧氏距离。亮度描述符对 wb 这类
    # 「R/B 反向增益、亮度近似不变」的扰动系统性低估，必须并列报告。
    la = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    lb = cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(np.float32)
    dC = np.sqrt((la[..., 1] - lb[..., 1]) ** 2 + (la[..., 2] - lb[..., 2]) ** 2)
    return dict(dL_mean=float(rel.mean() * 100), dL_p95=float(np.percentile(rel, 95) * 100),
                area_10=float((rel > 0.10).mean() * 100), area_25=float((rel > 0.25).mean() * 100),
                sat_hi=float(((b > 250) & (a <= 250)).mean() * 100),
                contrast=float(np.median(sb[ok] / np.maximum(sa[ok], 1e-6))) if ok.any() else np.nan,
                dChroma=float(dC.mean()))


def main():
    cats = sorted(p.name for p in MV.iterdir() if p.is_dir())
    rng_imgs = []
    for c in cats:                                   # 每类取 4 张正常测试图
        ps = sorted((MV / c / "test/good").glob("*.png"))[:4]
        rng_imgs += [(c, p) for p in ps]
    print(f"采样 {len(rng_imgs)} 张图（{len(cats)} 类 x 4）")
    agg = {}
    for mode in illum.MODES:
        for sev in SEVS:
            acc = []
            for c, p in rng_imgs:
                img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]
                if max(h, w) > 1024:                 # 描述符对尺度不敏感，缩图加速
                    s = 1024 / max(h, w)
                    img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
                seed = int.from_bytes(f"{c}|{mode}|{p.name}".encode()[:4].ljust(4, b'0'), "little")
                out = illum.apply(img, mode, sev, np.random.default_rng(seed))
                acc.append(descriptors(img, out))
            agg[f"{mode}|{sev}"] = {k: float(np.nanmean([a[k] for a in acc])) for k in acc[0]}
            print(f"  {mode:9} sev={sev}  " +
                  "  ".join(f"{k}={agg[f'{mode}|{sev}'][k]:6.2f}" for k in acc[0]), flush=True)
    out_p = ROOT / "reports/r2/perturb_physical.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(n_images=len(rng_imgs), severities=list(SEVS), desc=agg), open(out_p, "w"), indent=1)
    print(f"写入 {out_p}")


if __name__ == "__main__":
    main()
