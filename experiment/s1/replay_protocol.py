#!/usr/bin/env python3
"""用 fscache 引擎复现主扫描的 protocol × k × seed × domain 网格。

用途：遮罩实现修复后，原来的 256 个 masked 单元全部作废，需要重跑。
旧路径（逐 seed 调 run_anomalydino.py）约 8 小时；本脚本利用
「参考图编码一次、bank 取逐元素最小值」把同一网格压到几十分钟。

参考选取规则与上游严格一致：sorted(listdir(train/good))[seed*k:(seed+1)*k]
（build_aebad_domains.py 已用数字前缀把该顺序固定并可审计）。
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
DOMAINS = ["same", "illumination", "view", "background"]
DEFECTS = ["ablation", "breakdown", "fracture", "groove"]


def list_png(d): return sorted(p for p in d.iterdir() if p.suffix == ".png") if d.is_dir() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", nargs="+", default=["single", "mixed"])
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--backbone", default="dinov2_vits14")
    ap.add_argument("--resolution", type=int, default=448)
    ap.add_argument("--masking", action="store_true")
    ap.add_argument("--agg", default="meantop1p", choices=["meantop1p", "max"],
                    help="图像级聚合：meantop1p=AnomalyDINO，max=PatchCore。"
                         "Round-2 审查 R3 需要在 AeBAD 真实域上跑与 MVTec 相同的 5 个配置。")
    ap.add_argument("--dump-scores", default=None, help="逐图分数落盘（npz），供固定工作点 TPR 与测试样本 bootstrap 使用")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t_start = time.time()
    enc = Encoder(a.backbone, a.resolution)
    rows = []
    dump = {} if a.dump_scores else None
    for proto in a.protocols:
        root = ROOT / f"data/AeBAD_S_domains/{proto}"
        refdir = root / "blade_same/train/good"     # 四域参考集已验证字节一致
        refs = list_png(refdir)
        need = max(a.ks) * a.seeds
        nref = min(need, len(refs))

        # [Round-2 修复] 原实现把四个域 1639 张测试图的特征一次性常驻 GPU。
        # 对 DINOv2（1024x384）只有 2.6 GB，但对 ResNet l2+l3（3136x1536）是
        # 31.6 GB —— 32 GB 卡必 OOM。改为**逐域**建缓存：最大域 689 张约 13 GB。
        # 参考集、打分与评价逻辑完全不变，故结果与整批处理逐位一致。
        for dom in DOMAINS:
            items, cls_of = [], []
            for cls in ["good"] + DEFECTS:
                for p_ in list_png(root / f"blade_{dom}/test/{cls}"):
                    items.append((f"{dom}|{cls}|{p_.name}", p_, 0 if cls == "good" else 1))
                    cls_of.append(cls)
            cls_of = np.array(cls_of)
            print(f"[{proto}/{dom}] 测试图 {len(items)} 张，编码中…", flush=True)
            cache = TestCache(enc, items, masking=a.masking)
            y = cache.labels
            if dump is not None:
                dump[f"{proto}/{dom}/keys"] = np.array([it[0] for it in items])
                dump[f"{proto}/{dom}/cls"] = cls_of

            print(f"[{proto}/{dom}] 参考图编码 {nref} 张…", flush=True)
            maps = {}
            for i, rp in enumerate(refs[:need]):
                maps[i] = per_ref_mindist(cache, enc.encode_reference(rp, masking_ref=False,
                                                                     rotation=True))
                if (i + 1) % 16 == 0: print(f"    {i+1}/{nref}", flush=True)

            for k in a.ks:
                for seed in range(a.seeds):
                    idx = list(range(seed * k, (seed + 1) * k))
                    if idx[-1] >= len(maps): continue
                    sc = bank_scores(cache, [maps[i] for i in idx], agg=a.agg)
                    if dump is not None:
                        dump[f"{proto}/{dom}/k{k}/seed{seed}"] = sc.astype(np.float32)
                    rows.append(dict(protocol=proto, k=k, seed=seed, domain=dom, scope="all",
                                     backbone=a.backbone, agg=a.agg,
                                     auroc=an.auroc(y, sc), fpr95=an.fpr_at_tpr(y, sc),
                                     n=int(len(y))))
                    for dfc in DEFECTS:
                        mm = (cls_of == "good") | (cls_of == dfc)
                        if (y[mm] == 1).sum() == 0: continue
                        rows.append(dict(protocol=proto, k=k, seed=seed, domain=dom, scope=dfc,
                                         backbone=a.backbone, agg=a.agg,
                                         auroc=an.auroc(y[mm], sc[mm]),
                                         fpr95=an.fpr_at_tpr(y[mm], sc[mm]), n=int(mm.sum())))
            del cache, maps
            import torch as _t; _t.cuda.empty_cache()
            print(f"[{proto}/{dom}] 完成，累计 {len(rows)} 行", flush=True)

    meta = dict(backbone=a.backbone, resolution=a.resolution, masking=a.masking, agg=a.agg,
                protocols=a.protocols, ks=a.ks, seeds=a.seeds, engine="fscache",
                wall_seconds=round(time.time() - t_start, 1),
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
