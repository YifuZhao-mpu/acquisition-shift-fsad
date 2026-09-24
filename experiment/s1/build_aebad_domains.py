#!/usr/bin/env python3
"""
把 AeBAD_S 重组为 AnomalyDINO custom-dataset 可读的「域切片」布局。

科学目的（S1）：在固定 shot 预算 k 下，把「测试域」作为唯一自变量，
测量 same / illumination / view / background 四种条件各自造成多少退化。

两个关键控制：
  1. 四个域对象共用**逐字节相同**的 train/good 参考集 —— 因为 AnomalyDINO 用
     sorted(os.listdir())[seed*k:(seed+1)*k] 取参考，相同内容+相同 seed ⇒ 相同参考图。
     这样域间差异不会被参考采样污染。
  2. 参考顺序由显式数字前缀 (000_, 001_, ...) 固定，使 seed 分块完全可控可审计，
     不依赖原始文件名的偶然排序。

两种支撑集协议（这是 S1 的核心对照）：
  - single: 参考全部来自**单一**拍摄条件（部署现实：现场拍 k 张）。
            → 支撑集无法覆盖光照变化。
  - mixed : 参考在三种条件间轮转（round-robin）。
            → 支撑集覆盖了光照变化。
  两者之差 = 「支撑集多样性能否解释光照退化」的直接测量。
"""
import argparse, os, shutil, json, hashlib
from pathlib import Path

DOMAINS = ["same", "illumination", "view", "background"]
DEFECTS = ["ablation", "breakdown", "fracture", "groove"]
TRAIN_CONDS = ["background", "illumination", "view"]   # AeBAD train/good 下的三个条件


def list_pngs(d: Path):
    """只取真实 png，显式排除 macOS 元数据（._* / .DS_Store）。"""
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() == ".png"
                  and not p.name.startswith("._") and p.name != ".DS_Store")


def build_ref_order(src: Path, protocol: str, primary: str):
    """返回参考图的有序列表 (Path, 原始条件)。顺序即 seed 分块顺序。"""
    per_cond = {c: list_pngs(src / "train" / "good" / c) for c in TRAIN_CONDS}
    if protocol == "single":
        return [(p, primary) for p in per_cond[primary]]
    elif protocol == "mixed":
        # round-robin，使任意长度的前缀都尽量条件均衡
        out, idx = [], 0
        while any(idx < len(per_cond[c]) for c in TRAIN_CONDS):
            for c in TRAIN_CONDS:
                if idx < len(per_cond[c]):
                    out.append((per_cond[c][idx], c))
            idx += 1
        return out
    raise ValueError(protocol)


def link(src: Path, dst: Path, copy: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(os.path.abspath(src), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="AeBAD_S 根目录")
    ap.add_argument("--out", required=True, help="输出根目录")
    ap.add_argument("--protocol", choices=["single", "mixed"], required=True)
    ap.add_argument("--primary", default="background",
                    help="single 协议下参考图取自哪个条件（默认 background，样本最多）")
    ap.add_argument("--max-refs", type=int, default=64,
                    help="写入 train/good 的参考图数量上限（需 >= max_k * num_seeds）")
    ap.add_argument("--copy", action="store_true", help="复制而非符号链接")
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out) / a.protocol
    ref_order = build_ref_order(src, a.protocol, a.primary)[: a.max_refs]
    if not ref_order:
        raise SystemExit("没有找到参考图")

    manifest = {"protocol": a.protocol, "primary": a.primary,
                "n_refs": len(ref_order), "refs": [], "domains": {}}

    # ---- 为每个域建一个 object，train/good 内容完全一致 ----
    for dom in DOMAINS:
        obj = out / f"blade_{dom}"
        # 参考集：显式数字前缀锁定顺序
        for i, (p, cond) in enumerate(ref_order):
            link(p, obj / "train" / "good" / f"{i:03d}_{cond}_{p.name}", a.copy)
        # 测试：正常
        n_good = 0
        for p in list_pngs(src / "test" / "good" / dom):
            link(p, obj / "test" / "good" / p.name, a.copy); n_good += 1
        # 测试：各缺陷 + GT 掩码
        n_def = {}
        for dfc in DEFECTS:
            cnt = 0
            for p in list_pngs(src / "test" / dfc / dom):
                link(p, obj / "test" / dfc / p.name, a.copy); cnt += 1
            for p in list_pngs(src / "ground_truth" / dfc / dom):
                link(p, obj / "ground_truth" / dfc / p.name, a.copy)
            n_def[dfc] = cnt
        manifest["domains"][dom] = {"n_good": n_good, "n_defect": n_def,
                                    "n_total": n_good + sum(n_def.values())}

    manifest["refs"] = [{"idx": i, "cond": c, "file": p.name} for i, (p, c) in enumerate(ref_order)]
    # 参考集指纹：用于事后证明四个域确实用了同一组参考
    h = hashlib.sha256("".join(f"{i:03d}_{c}_{p.name}" for i, (p, c) in enumerate(ref_order)).encode())
    manifest["ref_set_sha256"] = h.hexdigest()

    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[{a.protocol}] 参考集 {len(ref_order)} 张  sha256={h.hexdigest()[:16]}")
    print(f"  参考条件分布: " + str({c: sum(1 for _, cc in ref_order if cc == c) for c in TRAIN_CONDS}))
    for dom in DOMAINS:
        d = manifest["domains"][dom]
        print(f"  blade_{dom:12s} good={d['n_good']:4d}  defect={sum(d['n_defect'].values()):4d}  total={d['n_total']:4d}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
