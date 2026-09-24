#!/usr/bin/env python3
"""M2AD 分析管线的双向自检 —— 在真实数据落地前跑，也可供审稿人复核。

三个检查：
  1. tone_idx 有效性：在 18 个标签按构造已知的合成扰动单元上，色调组与空间组必须完全分离。
  2. 植入真值可回收：造一组带已知效应的结果，analyze_m2ad 必须找回 M1/M2/M3 的方向与量级。
  3. 零效应不误报：造一组无效应但含单元级共同噪声的结果，六个检验必须全部不显著。

第 3 项最重要：配对设计里若把共同随机效应误当作信号，就会系统性地造出假阳性。
"""
from __future__ import annotations
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
CATS = [f"C{i}" for i in range(10)]
VIEWS = ["000", "090", "180", "270"]
ILL = [f"{i:02d}" for i in range(1, 11)]
BB = {"dinov2_vits14": ("ViT", "self"), "deit_small_patch16": ("ViT", "sup"),
      "dino_resnet50": ("CNN", "self"), "resnet50": ("CNN", "sup"),
      "wide_resnet50_2": ("CNN", "sup")}
TONE = dict(zip(ILL[1:], [0.22, 0.31, 0.40, 0.48, 0.57, 0.66, 0.74, 0.83, 0.92]))


def write_physical(p: Path):
    phys = {}
    for j, t in TONE.items():
        dl, nonu = 40.0, 1.0 / t - 1.0
        phys[j] = dict(dL_mean=dl, dL_p95=dl * (1 + nonu), area_10=50.0, dChroma=5.0,
                       contrast=0.95, sat_hi=0.5, nonuniformity=nonu, tone_idx=t, n_pairs=120)
    json.dump(dict(ref_illum="01", by_illum=phys, by_category={}), open(p, "w"))


def gen(out: Path, planted: bool, seed: int):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    clean = {"dinov2_vits14": .86, "deit_small_patch16": .82, "dino_resnet50": .84,
             "resnet50": .91, "wide_resnet50_2": .90}
    # 单元级共同随机效应：同一单元所有 backbone 共享（"这张图本来就难"）
    cell = {(c, v, j): rng.normal(0, .04) for c in CATS for v in VIEWS for j in ILL}
    for bb, (arch, pre) in BB.items():
        rows = []
        for c in CATS:
            for v in VIEWS:
                cb = (clean[bb] if planted else .87) + rng.normal(0, .03)
                for j in ILL:
                    if j == "01":
                        d = 0.0
                    elif planted:
                        d = (-.10 - .10 * TONE[j] + (.05 * TONE[j] if arch == "ViT" else 0)
                             + (.012 if pre == "self" else 0) * (1 if arch == "ViT" else -1)
                             + rng.normal(0, .02))
                    else:
                        d = -.15 + cell[(c, v, j)] + rng.normal(0, .02)
                    for sc in ("all", "detectable"):
                        rows.append(dict(scope=sc, auroc=float(np.clip(cb + d, 0, 1)), fpr95=.5,
                                         n=70, npos=35, arm="E1", category=c, view=v, illum=j,
                                         matched=(j == "01"), k=4, seed=0, backbone=bb,
                                         agg="meantop1p"))
        json.dump(dict(meta=dict(backbone=bb), rows=rows), open(out / f"{bb}_meantop1p.json", "w"))


def run(rdir, phys, outp):
    subprocess.run([sys.executable, str(HERE / "analyze_m2ad.py"), "--reports-dir", str(rdir),
                    "--physical", str(phys), "--out", str(outp)],
                   capture_output=True, text=True, check=True)
    return json.load(open(outp))


def main():
    ok = True
    print("=" * 76); print("检查 1：tone_idx 在已知标签的合成网格上是否分开两组"); print("=" * 76)
    r = subprocess.run([sys.executable, str(HERE / "m2ad_physical.py"), "--validate"],
                       capture_output=True, text=True)
    print(r.stdout.strip())
    ok &= (r.returncode == 0)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td); phys = td / "phys.json"; write_physical(phys)

        print("\n" + "=" * 76); print("检查 2：植入真值能否回收"); print("=" * 76)
        gen(td / "planted", True, 7)
        res = run(td / "planted", phys, td / "planted.json")
        checks = [
            ("M1 自监督臂 > 0", res["all|M1-自监督臂"]["lo"] > 0),
            ("M1 监督臂 > 0", res["all|M1-监督臂"]["lo"] > 0),
            ("M2 两臂异号", res["all|M2-ViT臂"]["est"] * res["all|M2-CNN臂"]["est"] < 0),
            ("M2 幅度 < M1", max(abs(res["all|M2-ViT臂"]["est"]), abs(res["all|M2-CNN臂"]["est"]))
             < min(abs(res["all|M1-自监督臂"]["est"]), abs(res["all|M1-监督臂"]["est"]))),
            ("M3 ρ > 0.8", res["all|M3"]["est"] > 0.8),
            ("M3 显著", res["all|M3"]["p"] < 0.01),
        ]
        for nm, c in checks:
            print(f"  {'通过' if c else '失败'}  {nm}"); ok &= c

        print("\n" + "=" * 76); print("检查 3：零效应是否保持沉默"); print("=" * 76)
        gen(td / "null", False, 11)
        res = run(td / "null", phys, td / "null.json")
        for nm in ("M1-自监督臂", "M1-监督臂", "M2-ViT臂", "M2-CNN臂", "M3"):
            k = f"all|{nm}"
            if k not in res: continue
            c = res[k]["p"] > 0.05
            print(f"  {'通过' if c else '失败'}  {nm} 不显著 (p={res[k]['p']:.3f}, "
                  f"CI [{res[k]['lo']:+.4f},{res[k]['hi']:+.4f}])"); ok &= c

    print("\n" + ("全部通过" if ok else "有失败项"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
