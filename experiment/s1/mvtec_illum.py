#!/usr/bin/env python3
"""MVTec AD 上的受控光照扰动实验 —— 检验预注册的「签名 × 扰动」交互（见 PREREGISTRATION.md）。

设计要点：
  - 扰动**只施加于测试图**（正常与异常一视同仁），参考图保持标称条件
    → 对应真实部署：标称条件下建库，之后照明漂移。
  - severity=0 严格恒等于原图，构成同一流程内的零扰动对照
    （排除"扰动管线本身引入差异"的质疑）。
  - 参考图每类别只编码一次，跨全部 (mode, severity) 复用。
  - 逐图扰动的随机性由 (category, mode, severity, image) 派生的确定性种子控制，
    保证可复现，且同一图在不同 severity 下用同一随机实例（高光位置等不跳变）。
"""
import os
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from fscache import Encoder, TestCache, per_ref_mindist, bank_scores
import illum
import importlib.util
_s = importlib.util.spec_from_file_location('an', str(Path(__file__).parent / 'analyze.py'))
an = importlib.util.module_from_spec(_s); _s.loader.exec_module(an)

ROOT = Path(os.environ.get("IADSHIFT_ROOT",
                    Path(__file__).resolve().parents[2]))   # experiment/s1/x.py -> 项目根
MV = ROOT / "data/mvtec_anomaly_detection"


def seed_for(cat, mode, img):
    """扰动随机实例只依赖 (类别, 模式, 图像)，**不依赖 severity**
    → 同一图在不同 severity 下是同一扰动的强度序列，可做剂量-反应分析。"""
    h = hashlib.sha256(f"{cat}|{mode}|{img}".encode()).digest()
    return int.from_bytes(h[:4], "little")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--severities", nargs="+", type=float, default=[0.0, 0.33, 0.67, 1.0])
    ap.add_argument("--modes", nargs="+", default=illum.MODES)
    ap.add_argument("--categories", nargs="+", default=None)
    ap.add_argument("--backbone", default="dinov2_vits14")
    ap.add_argument("--agg", default="meantop1p", choices=["meantop1p", "max"],
                    help="图像级聚合：meantop1p=AnomalyDINO，max=PatchCore")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-scores", default=None,
                    help="额外落盘逐图异常分数（npz）。Round-2 审查 R10/R11 需要："
                         "固定工作点 TPR 与测试样本重采样区间都要原始分数，不能只存 AUROC。")
    a = ap.parse_args()

    cats = a.categories or sorted(p.name for p in MV.iterdir() if p.is_dir())
    t_start = time.time()
    enc = Encoder(a.backbone, 448)
    rows = []
    dump = {} if a.dump_scores else None

    for ci, cat in enumerate(cats):
        cdir = MV / cat
        # ---- 测试清单 ----
        items, cls_of = [], []
        for d in sorted((cdir / "test").iterdir()):
            if not d.is_dir(): continue
            for p in sorted(d.iterdir()):
                if p.suffix.lower() == ".png":
                    items.append((f"{d.name}/{p.name}", p, 0 if d.name == "good" else 1))
                    cls_of.append(d.name)
        cls_of = np.array(cls_of)
        # ---- 参考图：只编码一次 ----
        refs = sorted((cdir / "train/good").glob("*.png"))[: a.k * a.seeds]
        ref_feats = [enc.encode_reference(p, masking_ref=False, rotation=True) for p in refs]
        print(f"[{ci+1}/{len(cats)}] {cat}: test={len(items)} refs={len(refs)}", flush=True)

        # ---- 条件：severity=0 只跑一次（六种模式恒等） ----
        conds = [("none", 0.0)] + [(m, s) for m in a.modes for s in a.severities if s > 0]
        if dump is not None:
            dump[f"{cat}/keys"] = np.array([it[0] for it in items])
            dump[f"{cat}/cls"] = cls_of
            dump[f"{cat}/conds"] = np.array([f"{m}|{s}" for m, s in conds])
            cat_scores = np.full((len(conds), a.seeds, len(items)), np.nan, dtype=np.float32)
        for ic, (mode, sev) in enumerate(conds):
            if mode == "none":
                tf = None
            else:
                def tf(im, key, _m=mode, _s=sev, _c=cat):
                    return illum.apply(im, _m, _s, np.random.default_rng(seed_for(_c, _m, key)))
            cache = TestCache(enc, items, masking=False, transform=tf)  # noqa: F841
            y = cache.labels
            maps = [per_ref_mindist(cache, rf) for rf in ref_feats]
            for seed in range(a.seeds):
                idx = list(range(seed * a.k, (seed + 1) * a.k))
                if idx[-1] >= len(maps): continue
                sc = bank_scores(cache, [maps[i] for i in idx], agg=a.agg)
                if dump is not None:
                    cat_scores[ic, seed] = sc
                for dfc in sorted(set(cls_of) - {"good"}):
                    m = (cls_of == "good") | (cls_of == dfc)
                    if (y[m] == 1).sum() == 0: continue
                    rows.append(dict(category=cat, defect=dfc, mode=mode, severity=sev,
                                     backbone=a.backbone, agg=a.agg,
                                     seed=seed, auroc=an.auroc(y[m], sc[m]),
                                     fpr95=an.fpr_at_tpr(y[m], sc[m]), n=int(m.sum())))
        if dump is not None:
            dump[f"{cat}/scores"] = cat_scores
            del cache
        print(f"    {cat} 完成，累计 {len(rows)} 行", flush=True)

    meta = dict(k=a.k, seeds=a.seeds, severities=a.severities, modes=a.modes,
                backbone=a.backbone, agg=a.agg, categories=cats, masking=False, rotation=True,
                wall_seconds=round(time.time() - t_start, 1))
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
