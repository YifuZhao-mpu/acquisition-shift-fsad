#!/usr/bin/env python3
"""M2AD 上的真实视角×光照偏移实验 —— 检验 M2AD_PREREGISTRATION.md 的 M1–M4。

与主稿 MVTec / AeBAD 扫描共用同一引擎（fscache）、同一协议
（448px, masking=False, rotation=True, k=4, 1-NN, mean_top1p / max），
因此三个数据集的数字可直接并排比较。

设计（预注册锁定）：
  E1 纯光照：bank 与测试图**同视角**，bank 取 illumination=01，测试遍历 10 档。
  E2 纯视角：bank 固定在 (view=000, illumination=01)，测试取 illumination=01，遍历 12 视角。

缓存粒度 = 一个 (类别, 视角) 单元的 700 张测试图。对 ResNet l2+l3
（3136x1536/图）约 13.5 GB，单卡 32 GB 可容；整类别 8400 张则必 OOM。
"""
import os
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from fscache import Encoder, TestCache, per_ref_mindist, bank_scores
import importlib.util
_s = importlib.util.spec_from_file_location('an', str(Path(__file__).parent / 'analyze.py'))
an = importlib.util.module_from_spec(_s); _s.loader.exec_module(an)

ROOT = Path(os.environ.get("IADSHIFT_ROOT",
                    Path(__file__).resolve().parents[2]))   # experiment/s1/x.py -> 项目根
M2 = ROOT / "data/M2AD"          # 可被 --data-root 覆盖（冒烟测试用小规模替身数据集）
REF_ILLUM = "01"      # 预注册锁定的匹配档
REF_VIEW = "000"      # 预注册锁定的 E2 参考视角


def load_meta():
    d = json.load(open(M2 / "meta_unsupervised.json"))
    return d["train"], d["test"]


def pick_refs(train_rows, cls, view, illum, need):
    """参考集选取规则与上游一致：按 img_path 排序后取前 need 张。"""
    rs = sorted((r for r in train_rows[cls]
                 if r["view"] == view and r["illumination"] == illum),
                key=lambda r: r["img_path"])
    return [M2 / "images" / r["img_path"] for r in rs[:need]]


def auroc_rows(y, sc, det_ok, **kw):
    """同一批分数下两个标签口径各出一行：all / detectable。"""
    out = []
    for scope, m in (("all", np.ones(len(y), bool)), ("detectable", det_ok)):
        yy, ss = y[m], sc[m]
        if yy.sum() == 0 or (yy == 0).sum() == 0:
            continue
        out.append(dict(scope=scope, auroc=an.auroc(yy, ss),
                        fpr95=an.fpr_at_tpr(yy, ss), n=int(m.sum()),
                        npos=int(yy.sum()), **kw))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dinov2_vits14")
    ap.add_argument("--agg", default="meantop1p", choices=["meantop1p", "max"])
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=7)
    ap.add_argument("--categories", nargs="+", default=None)
    ap.add_argument("--views", nargs="+", default=None,
                    help="限定 E1 的视角集合（默认全部 12 个）。仅用于压缩算力，不改变假设。")
    ap.add_argument("--resolution", type=int, default=448)
    ap.add_argument("--no-e2", action="store_true", help="跳过视角偏移对照臂")
    ap.add_argument("--dump-scores", default=None)
    ap.add_argument("--data-root", default=None, help="覆盖数据根目录（冒烟测试用）")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.data_root:
        global M2
        M2 = Path(a.data_root)
    train_rows, test_rows = load_meta()
    cats = a.categories or sorted(test_rows)
    need = a.k * a.seeds
    t0 = time.time()
    enc = Encoder(a.backbone, a.resolution)
    rows, dump = [], ({} if a.dump_scores else None)

    for ci, cls in enumerate(cats):
        tr = test_rows[cls]
        views = a.views or sorted({r["view"] for r in tr})
        # E2 的参考集：整个类别只需一次
        r0_paths = pick_refs(train_rows, cls, REF_VIEW, REF_ILLUM, need)
        r0_feats = [enc.encode_reference(p, masking_ref=False, rotation=True) for p in r0_paths] \
            if not a.no_e2 else []

        for vi, view in enumerate(views):
            sub = sorted((r for r in tr if r["view"] == view), key=lambda r: r["img_path"])
            items = [(r["img_path"], M2 / "images" / r["img_path"], int(r["image_anomaly"]))
                     for r in sub]
            illum = np.array([r["illumination"] for r in sub])
            y = np.array([int(r["image_anomaly"]) for r in sub])
            # detectable 只在异常图上有定义；正常图恒为 True（它们不是"任务不可能"的对象）
            det_ok = np.array([(r["detectable"] is True) if r["image_anomaly"] else True
                               for r in sub])
            print(f"[{ci+1}/{len(cats)} {cls} | view {view} ({vi+1}/{len(views)})] "
                  f"测试图 {len(items)}，异常 {int(y.sum())}（可检出 {int((y & det_ok).sum())}）",
                  flush=True)

            cache = TestCache(enc, items, masking=False)
            rs_paths = pick_refs(train_rows, cls, view, REF_ILLUM, need)
            maps_self = [per_ref_mindist(cache, enc.encode_reference(p, masking_ref=False,
                                                                     rotation=True))
                         for p in rs_paths]
            maps_v0 = [per_ref_mindist(cache, f) for f in r0_feats]

            for seed in range(a.seeds):
                idx = list(range(seed * a.k, (seed + 1) * a.k))
                # ---- E1：同视角，光照档变化 ----
                if idx[-1] < len(maps_self):
                    sc = bank_scores(cache, [maps_self[i] for i in idx], agg=a.agg)
                    if dump is not None:
                        dump[f"E1/{cls}/{view}/seed{seed}"] = sc.astype(np.float32)
                    for j in sorted(set(illum)):
                        m = illum == j
                        rows += auroc_rows(y[m], sc[m], det_ok[m], arm="E1", category=cls,
                                           view=view, illum=j, matched=(j == REF_ILLUM),
                                           k=a.k, seed=seed, backbone=a.backbone, agg=a.agg)
                # ---- E2：同光照档，视角变化 ----
                if maps_v0 and idx[-1] < len(maps_v0):
                    sc0 = bank_scores(cache, [maps_v0[i] for i in idx], agg=a.agg)
                    m = illum == REF_ILLUM
                    if dump is not None:
                        dump[f"E2/{cls}/{view}/seed{seed}"] = sc0[m].astype(np.float32)
                    rows += auroc_rows(y[m], sc0[m], det_ok[m], arm="E2", category=cls,
                                       view=view, illum=REF_ILLUM, matched=(view == REF_VIEW),
                                       k=a.k, seed=seed, backbone=a.backbone, agg=a.agg)

            if dump is not None:
                dump[f"keys/{cls}/{view}"] = np.array([r["img_path"] for r in sub])
                dump[f"illum/{cls}/{view}"] = illum
                dump[f"y/{cls}/{view}"] = y.astype(np.int8)
                dump[f"det/{cls}/{view}"] = det_ok.astype(np.int8)
            del cache, maps_self, maps_v0
            import torch as _t; _t.cuda.empty_cache()
        print(f"  {cls} 完成，累计 {len(rows)} 行，用时 {time.time()-t0:.0f}s", flush=True)

    meta = dict(backbone=a.backbone, agg=a.agg, k=a.k, seeds=a.seeds,
                resolution=a.resolution, masking=False, rotation=True,
                categories=cats, views=a.views, ref_illum=REF_ILLUM, ref_view=REF_VIEW,
                e2=not a.no_e2, engine="fscache", dataset="M2AD",
                wall_seconds=round(time.time() - t0, 1),
                gpu=__import__("torch").cuda.get_device_name(0))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(meta=meta, rows=rows), open(a.out, "w"), indent=1)
    print(f"写入 {a.out}（{len(rows)} 行）")
    if dump is not None:
        dump["__meta__"] = np.array([json.dumps(meta)])
        Path(a.dump_scores).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.dump_scores, **dump)
        print(f"写入逐图分数 {a.dump_scores}")


if __name__ == "__main__":
    main()
