#!/usr/bin/env python3
"""把 M2AD 的 10 档真实光照放到与合成扰动**完全相同**的物理坐标上 —— 预注册 M3。

比 AeBAD 更强的一点：M2AD 的 10 档光照是同一台相机、同一视角、同一样本
逐档拍摄的，**逐像素配准**。因此可以直接调用 `perturb_physical.descriptors`
（合成网格用的同一个函数）做**配对**测量，而不是像 AeBAD 那样只能比分布。
这让 M2AD 的真实光照档与论文的 synthetic severity 轴可以并排放在一张图上。

tone_idx = 色调映射拟合优度 R^2（预注册修订 C）
  色调映射按定义是**逐像素的亮度函数** b = f(a)：把 b 对 a 做分箱中位数拟合，
  R^2 就是"这次变化有多大程度只是一条全局曲线"。空间光照让 b 依赖位置而非仅依赖 a，
  R^2 因此下降。该量无量纲、有界、对动态范围不敏感。

  为什么换掉上一版 1/(1+(ΔL_p95-ΔL)/ΔL)：M2AD 是黑底上的小样本
  （全图 97% 像素亮度 < 10），相对亮度变化 |b-a|/max(a,1) 的分母接近 0，
  ΔL 炸到 100–470%，p95 到 4667%；且方向性打光让同一表面从阴影(L=3)
  变到受光(L=100)，相对变化本身就没有上界。改用 R^2 后两个问题同时消失。

  前景限定：描述量只在 (a>12) | (b>12) 的像素上算。黑背景在各档之间完全相同，
  计入会把 R^2 虚假地推向 1（大量 (0,0) 点被完美预测）。

  在 18 个标签按构造已知的合成扰动单元上：色调组 [0.9915, 0.9977]，
  空间组 [0.3719, 0.9273]，完全分离（间隙 0.064），且在每个空间模式内随
  severity 单调下降。`--validate` 重跑该对照。
"""
from __future__ import annotations
import os
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).parent))
import importlib.util
_s = importlib.util.spec_from_file_location('pp', str(Path(__file__).parent / 'perturb_physical.py'))
pp = importlib.util.module_from_spec(_s); _s.loader.exec_module(pp)

ROOT = Path(os.environ.get("IADSHIFT_ROOT",
                    Path(__file__).resolve().parents[2]))   # experiment/s1/x.py -> 项目根
M2 = ROOT / "data/M2AD"
REF_ILLUM = "01"


def tone_r2(a, b, mask, nbins=64):
    """色调映射拟合优度：把 b 的亮度对 a 的亮度做分箱中位数拟合后的 R^2。"""
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)[mask]
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32)[mask]
    if len(ga) < 500: return float("nan")
    q = np.quantile(ga, np.linspace(0, 1, nbins + 1))
    idx = np.clip(np.digitize(ga, q[1:-1]), 0, nbins - 1)
    pred = np.empty_like(gb)
    for k in range(nbins):
        m = idx == k
        if m.any(): pred[m] = np.median(gb[m])
    return 1.0 - float(((gb - pred) ** 2).mean()) / max(float(gb.var()), 1e-9)


def foreground(a, b, thr=12):
    """并集前景：偏移后新出现的高光也要算进来，否则会低估空间成分。"""
    return ((cv2.cvtColor(a, cv2.COLOR_RGB2GRAY) > thr) |
            (cv2.cvtColor(b, cv2.COLOR_RGB2GRAY) > thr))


def load(p, maxside=1024):
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None: return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if max(h, w) > maxside:
        s = maxside / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def validate_index():
    """在 18 个**标签按构造已知**的合成扰动单元上检验 tone_idx 能否分开两组。
    exposure/gamma/wb 是逐像素同一条曲线；gradient/specular/shadow 把变化集中在部分视场。
    直接在 MVTec 正常图上重新施加扰动并算 R^2，不依赖任何缓存的中间产物。"""
    import importlib.util as _iu
    MV = ROOT / "data/mvtec_anomaly_detection"
    if not MV.is_dir():
        print(f"缺少 {MV}"); return False
    _i = _iu.spec_from_file_location('illum', str(Path(__file__).parent / 'illum.py'))
    illum = _iu.module_from_spec(_i); _i.loader.exec_module(illum)
    cats = sorted(p.name for p in MV.iterdir() if p.is_dir())
    imgs = [(c, p) for c in cats for p in sorted((MV / c / "test/good").glob("*.png"))[:3]]
    TONE = {"exposure", "gamma", "wb"}
    t, sp, cells = [], [], []
    for mode in illum.MODES:
        for sev in (0.33, 0.67, 1.0):
            acc = []
            for c, p in imgs:
                img = load(p, 768)
                if img is None: continue
                seed = int.from_bytes(f"{c}|{mode}|{p.name}".encode()[:4].ljust(4, b'0'), "little")
                out = illum.apply(img, mode, sev, np.random.default_rng(seed))
                acc.append(tone_r2(img, out, np.ones(img.shape[:2], bool)))
            v = float(np.nanmean(acc))
            cells.append(dict(mode=mode, severity=sev, tone_r2=v,
                              group="tone" if mode in TONE else "spatial"))
            (t if mode in TONE else sp).append(v)
    print(f"tone_idx 有效性检验（{len(t)} 个色调单元 / {len(sp)} 个空间单元）：")
    print(f"  色调组 [{min(t):.4f}, {max(t):.4f}]   空间组 [{min(sp):.4f}, {max(sp):.4f}]")
    ok = min(t) > max(sp)
    print(f"  完全分离: {ok}（间隙 {min(t) - max(sp):+.4f}）")
    # 落盘逐单元值：没有 MVTec 的读者也能核对这 18 个数与分离度的算术
    out = ROOT / "reports/r2/tone_index_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(index="tone-map R^2 on binned-median fit, foreground only",
                   cells=cells, tone=t, spatial=sp,
                   tone_range=[min(t), max(t)], spatial_range=[min(sp), max(sp)],
                   separated=bool(ok), gap=min(t) - max(sp)), open(out, "w"), indent=1)
    print(f"  逐单元值写入 {out}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="只跑指数有效性检验（用合成网格的已知标签），不读 M2AD")
    ap.add_argument("--views", nargs="+", default=["000", "090", "180", "270"])
    ap.add_argument("--specimens", type=int, default=3, help="每 (类别,视角) 取几个正常样本")
    ap.add_argument("--categories", nargs="+", default=None)
    ap.add_argument("--data-root", default=None, help="覆盖数据根目录（冒烟测试用）")
    ap.add_argument("--out", default=str(ROOT / "reports/r2/m2ad_physical.json"))
    a = ap.parse_args()

    global M2
    if a.validate:
        sys.exit(0 if validate_index() else 1)
    if a.data_root: M2 = Path(a.data_root)
    meta = json.load(open(M2 / "meta_unsupervised.json"))["test"]
    cats = a.categories or sorted(meta)
    # 索引：(cls, view, specimen) -> {illum: path}
    idx = defaultdict(dict)
    for cls in cats:
        for r in meta[cls]:
            if r["image_anomaly"] or r["view"] not in a.views: continue
            idx[(cls, r["view"], r["object_name"])][r["illumination"]] = r["img_path"]

    acc = defaultdict(list)          # illum -> [desc]
    per_cat = defaultdict(list)      # (cls, illum) -> [desc]
    groups = sorted(idx)
    used = defaultdict(int)
    for cls, view, spec in groups:
        if used[(cls, view)] >= a.specimens: continue
        d = idx[(cls, view, spec)]
        if REF_ILLUM not in d: continue
        base = load(M2 / "images" / d[REF_ILLUM])
        if base is None: continue
        used[(cls, view)] += 1
        for j, rel in sorted(d.items()):
            if j == REF_ILLUM: continue
            img = load(M2 / "images" / rel)
            if img is None or img.shape != base.shape: continue
            m = foreground(base, img)
            if m.sum() < 500: continue
            desc = pp.descriptors(base, img)          # 全图描述量，保留供参考
            desc["fg_frac"] = float(m.mean() * 100)
            desc["tone_idx"] = tone_r2(base, img, m)
            acc[j].append(desc); per_cat[(cls, j)].append(desc)

    if not acc:
        print("没有读到任何图像——数据可能还没解压完"); sys.exit(1)

    def mean_of(lst): return {k: float(np.nanmean([x[k] for x in lst])) for k in lst[0]}
    out = {j: mean_of(v) | {"n_pairs": len(v)} for j, v in sorted(acc.items())}

    keys = ["dL_mean", "dL_p95", "area_10", "dChroma", "contrast", "fg_frac", "tone_idx"]
    print(f"M2AD：{len(cats)} 类 x {len(a.views)} 视角 x {a.specimens} 样本，相对 illum={REF_ILLUM} 的配对测量\n")
    print(f"{'illum':7}" + "".join(f"{k:>10}" for k in keys) + f"{'n':>7}")
    for j in sorted(out):
        print(f"{j:7}" + "".join(f"{out[j][k]:10.2f}" for k in keys) + f"{out[j]['n_pairs']:7d}")
    ts = sorted(out, key=lambda j: out[j]["tone_idx"])
    print(f"\n色调主导排序（最空间化 → 最色调化）：{' < '.join(ts)}")

    res = dict(ref_illum=REF_ILLUM, views=a.views, specimens=a.specimens, categories=cats,
               by_illum=out,
               by_category={f"{c}|{j}": mean_of(v) for (c, j), v in sorted(per_cat.items())})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\n写入 {a.out}")


if __name__ == "__main__":
    main()
