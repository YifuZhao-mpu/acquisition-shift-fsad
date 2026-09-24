#!/usr/bin/env python3
"""复核前景遮罩形态学缺陷的影响量 —— 论文 §7.2。

上游 backbones.py:164-171 把 3x3 形态学核作用在 (Npatches,1) 列向量上：
第 166 行的 reshape 结果只赋给临时变量 m（用于中心裁剪启发式），mask 本身
从未变形。因此 dilate/close 退化为沿**光栅展平顺序**的一维操作。

本脚本对同一批 patch 特征分别按「缺陷实现」与「正确实现」生成遮罩，
量化 IoU、patch 判定分歧率，以及分歧是否依赖测试域。
"""
from __future__ import annotations
import os
import json, sys
from pathlib import Path
import numpy as np, cv2
from sklearn.decomposition import PCA

ROOT = Path(os.environ.get("IADSHIFT_ROOT",
                    Path(__file__).resolve().parents[2]))   # experiment/s1/x.py -> 项目根
sys.path.insert(0, str(Path(__file__).parent))
from fscache import Encoder

K = np.ones((3, 3), np.uint8)


def _pc(feats):
    return PCA(n_components=1, svd_solver="randomized").fit_transform(feats.astype(np.float32))


def _base(first_pc, grid, threshold=10, border=0.2):
    mask = first_pc > threshold
    m = mask.reshape(grid)[int(grid[0]*border):int(grid[0]*(1-border)),
                           int(grid[1]*border):int(grid[1]*(1-border))]
    if m.sum() <= m.size * 0.35:
        mask = -first_pc > threshold
    return mask


def mask_buggy(feats, grid):
    """上游原样：形态学作用在 (N,1) 列向量上。"""
    mask = _base(_pc(feats), grid)
    mask = cv2.dilate(mask.astype(np.uint8), K).astype(bool)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, K).astype(bool)
    return mask.squeeze()


def mask_fixed(feats, grid):
    """先 reshape 成 (H,W) 再做形态学，然后展平。"""
    mask = _base(_pc(feats), grid)
    m2 = mask.reshape(grid).astype(np.uint8)
    m2 = cv2.dilate(m2, K)
    m2 = cv2.morphologyEx(m2, cv2.MORPH_CLOSE, K)
    return m2.astype(bool).reshape(-1)


def main():
    enc = Encoder("dinov2_vits14", 448)
    groups = {
        "illumination / fracture": ROOT / "data/AeBAD_S_domains/single/blade_illumination/test/fracture",
        "same / good":             ROOT / "data/AeBAD_S_domains/single/blade_same/test/good",
        "view / fracture":         ROOT / "data/AeBAD_S_domains/single/blade_view/test/fracture",
        "background / good":       ROOT / "data/AeBAD_S_domains/single/blade_background/test/good",
    }
    out, allious, alldiff = {}, [], []
    print(f"{'group':26}{'n':>4}{'mean IoU':>10}{'min':>8}{'max':>8}{'% patches differ':>18}")
    for g, d in groups.items():
        ps = sorted(d.glob("*.png"))[:6]
        ious, diffs = [], []
        for p in ps:
            img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            t, grid = enc.model.prepare_image(img)
            f = enc.model.extract_features(t)
            a, b = mask_buggy(f, grid), mask_fixed(f, grid)
            inter = (a & b).sum(); union = (a | b).sum()
            ious.append(inter / max(union, 1)); diffs.append((a != b).mean())
        out[g] = dict(n=len(ps), iou_mean=float(np.mean(ious)), iou_min=float(np.min(ious)),
                      iou_max=float(np.max(ious)), pct_differ=float(np.mean(diffs) * 100))
        allious += ious; alldiff += diffs
        print(f"{g:26}{len(ps):4}{np.mean(ious):10.3f}{np.min(ious):8.3f}{np.max(ious):8.3f}"
              f"{np.mean(diffs)*100:17.2f}%")
    print(f"\n{'ALL':26}{len(allious):4}{np.mean(allious):10.3f}{np.min(allious):8.3f}"
          f"{np.max(allious):8.3f}{np.mean(alldiff)*100:17.2f}%")
    print("\n论文 §7.2 主张: 平均 IoU 0.836，7.89% 的 patch 判定不同，且分歧依赖于域")
    p = ROOT / "reports/r2/mask_bug_verify.json"
    json.dump(dict(per_group=out, overall=dict(n=len(allious), iou_mean=float(np.mean(allious)),
              pct_differ=float(np.mean(alldiff) * 100))), open(p, "w"), indent=1)
    print(f"写入 {p}")


if __name__ == "__main__":
    main()
