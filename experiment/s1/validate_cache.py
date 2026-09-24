#!/usr/bin/env python3
"""正确性闸门：fscache 的快速路径必须复现主扫描（AnomalyDINO 原始实现）的异常分数。

取主扫描一个已完成配置（single, k, seed, masking=off, rotation=on），
用 fscache 重算同一组测试图的分数，与 measurements CSV 逐图比对。
不通过则后续所有受控实验结论不可信。
"""
import os
import csv, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from fscache import Encoder, TestCache, per_ref_mindist, bank_scores

ROOT = Path(os.environ.get("IADSHIFT_ROOT",
                    Path(__file__).resolve().parents[2]))   # experiment/s1/x.py -> 项目根
AD = ROOT / "experiment/AnomalyDINO"
DEFECTS = ["ablation", "breakdown", "fracture", "groove"]


def main(proto="single", k=1, seed=0, domain="illumination", prep="agnostic", limit=None):
    data = ROOT / f"data/AeBAD_S_domains/{proto}"
    obj = data / f"blade_{domain}"

    # 主扫描的参考选取规则：sorted(listdir)[seed*k : (seed+1)*k]
    refs = sorted(p.name for p in (obj / "train/good").iterdir() if p.suffix == ".png")
    refs = refs[seed * k:(seed + 1) * k]
    print(f"配置 proto={proto} k={k} seed={seed} domain={domain} prep={prep}")
    print(f"参考图: {refs}")

    # 主扫描的分数
    csvp = AD / f"results_AeBAD_{proto}/dinov2_vits14_448/{k}-shot_preprocess={prep}/measurements_seed={seed}.csv"
    ref_scores = {}
    for r in csv.DictReader(open(csvp)):
        if r["Object"] == f"blade_{domain}":
            ref_scores[r["Sample"]] = float(r["Anomaly_Score"])
    print(f"主扫描该域样本数: {len(ref_scores)}")

    items = []
    for cls in ["good"] + DEFECTS:
        d = obj / "test" / cls
        if not d.is_dir(): continue
        for p in sorted(d.iterdir()):
            if p.suffix == ".png":
                key = f"{cls}/{p.name}"
                if key in ref_scores: items.append((key, p, 0 if cls == "good" else 1))
    if limit: items = items[:limit]
    print(f"本次核对样本数: {len(items)}")

    masking = (prep != "agnostic")          # 主扫描 agnostic ⇒ 无遮罩
    enc = Encoder()
    cache = TestCache(enc, items, masking=masking)
    maps = [per_ref_mindist(cache, enc.encode_reference(obj / "train/good" / rn,
                                                        masking_ref=False, rotation=True))
            for rn in refs]
    got = bank_scores(cache, maps)

    exp = np.array([ref_scores[k_] for k_ in cache.keys])
    diff = np.abs(got - exp)
    print()
    print(f"{'最大绝对差':>14s} = {diff.max():.6f}")
    print(f"{'平均绝对差':>14s} = {diff.mean():.6f}")
    print(f"{'相关系数':>14s} = {np.corrcoef(got, exp)[0,1]:.8f}")
    # CSV 只存 5 位小数，故 1e-5 量级差异属舍入
    ok = diff.max() < 2e-5
    print()
    print("✓ 通过：快速路径与主扫描一致（差异在 CSV 5 位小数舍入范围内）" if ok
          else f"✗ 不一致！最大差 {diff.max():.6f} 超出舍入容差")
    for i in np.argsort(-diff)[:4]:
        print(f"    {cache.keys[i]:28s} 快速={got[i]:.5f}  主扫描={exp[i]:.5f}  差={diff[i]:.2e}")
    return ok


if __name__ == "__main__":
    a = sys.argv[1:]
    ok = main(*(a[0:1] or ["single"]), int(a[1]) if len(a) > 1 else 1,
              int(a[2]) if len(a) > 2 else 0, *(a[3:4] or ["illumination"]))
    sys.exit(0 if ok else 1)
