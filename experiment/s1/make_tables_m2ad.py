#!/usr/bin/env python3
"""M2AD 的论文表格与行内宏 —— 单独成模块，避免再去动 750 行的 make_tables.py。

产出：
  tab_m2ad.tex        主文紧凑表：各表示的匹配档 / 偏移档 / Δ
  tab_m2ad_full.tex   补充：逐光照档 × 逐表示，两种标签口径
  tab_m2ad_phys.tex   补充：10 档光照的物理描述量与色调主导指数
  facts_m2ad.tex      行内宏
全部在结果缺失时写 "%% pending"，与既有表格的行为一致。
"""
from __future__ import annotations
import os
import glob, json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("IADSHIFT_ROOT",
                    Path(__file__).resolve().parents[2]))   # experiment/s1/x.py -> 项目根
OUT = ROOT / "paper/tables"
RNG = np.random.default_rng(3)
REF = "01"
ORDER = ["dinov2_vits14", "deit_small_patch16", "dino_resnet50", "resnet50", "wide_resnet50_2"]
NICE = {"dinov2_vits14": "DINOv2 ViT-S/14", "deit_small_patch16": "DeiT-S/16",
        "dino_resnet50": "DINO ResNet-50", "resnet50": "ResNet-50 (sup.)",
        "wide_resnet50_2": "WideResNet-50-2"}


def load(agg="meantop1p"):
    """-> A[bb][scope][(cat,view,illum)] = 逐 seed 平均 AUROC（只取 E1 臂）"""
    acc = defaultdict(list)
    for f in sorted(glob.glob(str(ROOT / f"reports/m2ad/*_{agg}.json"))):
        for r in json.load(open(f))["rows"]:
            if r["arm"] != "E1": continue
            acc[(r["backbone"], r["scope"], (r["category"], r["view"], r["illum"]))].append(r["auroc"])
    A = defaultdict(lambda: defaultdict(dict))
    for (bb, sc, key), v in acc.items():
        A[bb][sc][key] = float(np.mean(v))
    return A


def units(A, scope, bbs):
    if not bbs: return []
    u = set.intersection(*[set(A[b][scope]) for b in bbs])
    return sorted(k for k in u if k[2] != REF
                  and all((k[0], k[1], REF) in A[b][scope] for b in bbs))


def dvec(A, bb, scope, us):
    return np.array([A[bb][scope][u] - A[bb][scope][(u[0], u[1], REF)] for u in us])


def pboot(d, B=10000):
    d = np.asarray(d, float)
    if len(d) == 0: return (np.nan,) * 3
    bs = d[RNG.integers(0, len(d), (B, len(d)))].mean(1)
    return d.mean(), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def tab_m2ad(A, agg="meantop1p"):
    bbs = [b for b in ORDER if b in A and "all" in A[b]]
    us = units(A, "all", bbs)
    if len(bbs) < 2 or not us:
        (OUT / "tab_m2ad.tex").write_text("%% pending: M2AD\n"); return False
    ncat = len({u[0] for u in us}); nview = len({u[1] for u in us})
    L = [r"\begin{table}[t]",
         rf"\caption{{M2AD real view--illumination shift ($k=4$, 4 support draws, no masking). "
         rf"Bank and test share the viewpoint; only the illumination setting differs, so the "
         rf"contrast isolates illumination. Paired over {len(us)} (category, view, illumination) "
         rf"units from {ncat} part families $\times$ {nview} views $\times$ 9 shifted settings. "
         rf"$\Delta$ is the paired change from the matched setting; brackets give the "
         rf"bootstrap 95\% interval.}}",
         r"\label{tab:m2ad}\centering\footnotesize",
         r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{@{}lrrl@{}}", r"\toprule",
         r"Representation & matched & shifted & $\Delta$ [95\% CI]\\", r"\midrule"]
    for b in bbs:
        base = np.mean([A[b]["all"][(c, v, REF)] for (c, v, _) in us])
        m, lo, hi = pboot(dvec(A, b, "all", us))
        L.append(rf"{NICE.get(b,b)} & {base:.3f} & {base+m:.3f} & "
                 rf"${m:+.3f}$ $[{lo:+.3f},{hi:+.3f}]$\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "tab_m2ad.tex").write_text("\n".join(L) + "\n")
    return len(bbs)


def tab_m2ad_full(A, agg="meantop1p"):
    bbs = [b for b in ORDER if b in A and "all" in A[b]]
    if len(bbs) < 2:
        (OUT / "tab_m2ad_full.tex").write_text("%% pending: M2AD\n"); return False
    phys_f = ROOT / "reports/r2/m2ad_physical.json"
    phys = json.load(open(phys_f))["by_illum"] if phys_f.exists() else {}
    L = [r"\begin{table*}[t]",
         r"\caption{M2AD per illumination setting. Each cell is the paired change $\Delta$ in "
         r"image AUROC from the matched setting \texttt{I01}, averaged over part families, "
         r"views and support draws. \emph{tone} is the tone-dominance index of "
         r"Table~\ref{M-tab:m2adphys} of the main paper. \emph{arch} is the mean of the two "
         r"architecture contrasts "
         r"(ViT $-$ CNN within each pretraining level); positive means the ViTs lose less.}",
         r"\label{tab:m2adfull}\centering\footnotesize",
         r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}llr" + "r" * len(bbs) + r"r@{}}", r"\toprule",
         r"Scope & I & tone & " + " & ".join(
             NICE.get(b, b).replace("ResNet-50", "RN50").replace("WideRN50-2", "WRN")
             for b in bbs) + r" & arch\\", r"\midrule"]
    for scope in ("all", "detectable"):
        if not all(scope in A[b] for b in bbs): continue
        first = True
        for j in [f"{i:02d}" for i in range(2, 11)]:
            us = [u for u in units(A, scope, bbs) if u[2] == j]
            if not us: continue
            cells = [f"${dvec(A,b,scope,us).mean():+.3f}$" for b in bbs]
            arch = ""
            if all(b in A for b in ORDER[:4]):
                a = ((dvec(A, ORDER[0], scope, us) - dvec(A, ORDER[2], scope, us)) +
                     (dvec(A, ORDER[1], scope, us) - dvec(A, ORDER[3], scope, us))) / 2
                arch = f"${a.mean():+.3f}$"
            tag = (scope if first else "")
            tone = f"{phys[j]['tone_idx']:.2f}" if j in phys else "---"
            L.append(f"{tag} & \\texttt{{I{j}}} & {tone} & " + " & ".join(cells)
                     + f" & {arch}" + r"\\")
            first = False
        L.append(r"\addlinespace[2pt]")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (OUT / "tab_m2ad_full.tex").write_text("\n".join(L) + "\n")
    return True


def tab_m2ad_phys():
    f = ROOT / "reports/r2/m2ad_physical.json"
    if not f.exists():
        (OUT / "tab_m2ad_phys.tex").write_text("%% pending: m2ad_physical\n"); return False
    P = json.load(open(f))["by_illum"]
    L = [r"\begin{table}[!t]",
         r"\caption{Physical descriptors of M2AD's nine shifted illumination settings, "
         r"measured pairwise against the matched setting \texttt{I01} on the same specimen "
         r"and viewpoint, with the same function used for the synthetic grid "
         r"(Supplementary Table~\ref{S-tab:physical}). The tone-dominance index "
         r"$1/(1+(\Delta L_{p95}-\Delta L)/\Delta L)$ separates the synthetic tone and spatial "
         r"families with no overlap (Supplementary Sec.~\ref{S-sup:m2ad}).}",
         r"\label{tab:m2adphys}\centering\footnotesize",
         r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
         r"Setting & $\Delta L$ (\%) & $\Delta L_{p95}$ (\%) & $A_{>10}$ (\%) & $\Delta C$ & tone\\",
         r"\midrule"]
    for j in sorted(P, key=lambda j: -P[j]["tone_idx"]):
        v = P[j]
        L.append(rf"\texttt{{I{j}}} & {v['dL_mean']:.1f} & {v['dL_p95']:.1f} & "
                 rf"{v['area_10']:.1f} & {v['dChroma']:.1f} & {v['tone_idx']:.2f}\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "tab_m2ad_phys.tex").write_text("\n".join(L) + "\n")
    return len(P)


def tab_m2ad_agg():
    """两种图像级聚合口径的并排比较 —— 预注册 §(e) 承诺的稳健性检查。
    mean_top1p 是预注册指定的主口径；max 是 PatchCore 口径。"""
    rows, have = [], []
    for agg in ("meantop1p", "max"):
        f = ROOT / f"reports/r2/m2ad_tests_{agg}.json"
        if not f.exists(): continue
        T = json.load(open(f)); have.append(agg)
        for nm, key in (("M1 arch., self-sup.", "M1-自监督臂"),
                        ("M1 arch., sup.", "M1-监督臂"),
                        ("M2 pretrain., ViT", "M2-ViT臂"),
                        ("M2 pretrain., CNN", "M2-CNN臂"),
                        ("M3 tone $\\rho$", "M3")):
            k = f"all|{key}"
            if k in T: rows.append((agg, nm, T[k]))
    if len(have) < 2:
        (OUT / "tab_m2ad_agg.tex").write_text("%% pending: 需要两种聚合口径\n"); return False
    names = [n for a, n, _ in rows if a == have[0]]
    L = [r"\begin{table}[t]",
         r"\caption{M2AD preregistered tests under both image-level aggregations. "
         r"\texttt{mean\_top1p} is the aggregation the preregistration designates as "
         r"primary; \texttt{max} is the PatchCore-style alternative, run as the "
         r"robustness check the preregistration also commits to. Brackets give the "
         r"bootstrap 95\% interval; $\checkmark$ marks a test that survives "
         r"Holm--Bonferroni over the preregistered family of six.}",
         r"\label{tab:m2adagg}\centering\scriptsize",
         r"\setlength{\tabcolsep}{2pt}",
         r"\begin{tabular}{@{}lll@{}}", r"\toprule",
         r"Test & \texttt{mean\_top1p} & \texttt{max}\\", r"\midrule"]
    d = {(a, n): v for a, n, v in rows}
    # Holm 由 analyze 脚本判定，这里按 p 与阈值重算以标注
    for agg in have:
        ps = sorted(((n, d[(agg, n)]["p"]) for n in names if (agg, n) in d), key=lambda x: x[1])
        blocked = False
        for i, (n, pv) in enumerate(ps):
            ok = (pv <= 0.05 / (len(ps) - i)) and not blocked
            if not ok: blocked = True
            d[(agg, n)]["holm"] = ok
    for n in names:
        cells = []
        for agg in have:
            v = d.get((agg, n))
            if v is None: cells.append("---"); continue
            mark = r"\,$\checkmark$" if v.get("holm") else ""
            cells.append(f"${v['est']:+.3f}\\,[{v['lo']:+.3f},{v['hi']:+.3f}]${mark}")
        L.append(f"{n} & " + " & ".join(cells) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "tab_m2ad_agg.tex").write_text("\n".join(L) + "\n")
    return len(names)


def facts_m2ad(A, agg="meantop1p"):
    M = {}
    bbs = [b for b in ORDER if b in A and "all" in A[b]]
    us = units(A, "all", bbs) if bbs else []
    if us:
        M["MtwoUnits"] = str(len(us))
        M["MtwoCats"] = str(len({u[0] for u in us}))
        M["MtwoViews"] = str(len({u[1] for u in us}))
        short = {"dinov2_vits14": "VitS", "deit_small_patch16": "VitSup",
                 "dino_resnet50": "CnnS", "resnet50": "CnnSup", "wide_resnet50_2": "Wrn"}
        for b in bbs:
            base = np.mean([A[b]["all"][(c, v, REF)] for (c, v, _) in us])
            m, lo, hi = pboot(dvec(A, b, "all", us))
            M[f"Mtwo{short[b]}Match"] = f"{base:.3f}"
            M[f"Mtwo{short[b]}Shift"] = f"{base+m:.3f}"
            M[f"Mtwo{short[b]}D"] = f"{m:+.3f}"
        if all(b in A for b in ORDER[:4]):
            for nm, a_, b_ in (("ArchS", ORDER[0], ORDER[2]), ("ArchSup", ORDER[1], ORDER[3]),
                               ("PreVit", ORDER[0], ORDER[1]), ("PreCnn", ORDER[2], ORDER[3])):
                m, lo, hi = pboot(dvec(A, a_, "all", us) - dvec(A, b_, "all", us))
                M[f"Mtwo{nm}"] = f"{m:+.3f}"
                M[f"Mtwo{nm}CI"] = f"[{lo:+.3f},\\,{hi:+.3f}]"
        # 标签有效性：可检出子集与全集的退化差
        if all("detectable" in A[b] for b in bbs):
            ud = units(A, "detectable", bbs)
            common = sorted(set(us) & set(ud))
            if common:
                da = np.mean([dvec(A, b, "all", common).mean() for b in bbs])
                dd = np.mean([dvec(A, b, "detectable", common).mean() for b in bbs])
                M["MtwoDegAll"] = f"{da:+.3f}"
                M["MtwoDegDet"] = f"{dd:+.3f}"
                M["MtwoImposs"] = f"{da - dd:+.3f}"
                M["MtwoImpossPct"] = f"{abs(da - dd) / max(abs(da), 1e-9) * 100:.0f}"
    tm = ROOT / "reports/r2/m2ad_tests_max.json"
    if tm.exists():
        T = json.load(open(tm))
        for nm, key in (("MaxArchS", "M1-自监督臂"), ("MaxArchSup", "M1-监督臂")):
            k = f"all|{key}"
            if k in T:
                M[f"Mtwo{nm}"] = f"{T[k]['est']:+.3f}"
                M[f"Mtwo{nm}CI"] = f"[{T[k]['lo']:+.3f},\\,{T[k]['hi']:+.3f}]"
        if "all|M3" in T:
            M["MtwoMaxRho"] = f"{T['all|M3']['est']:+.2f}"
            M["MtwoMaxRhoCI"] = f"[{T['all|M3']['lo']:+.2f},\\,{T['all|M3']['hi']:+.2f}]"
            M["MtwoMaxRhoP"] = f"{T['all|M3']['p']:.3f}"
    t = ROOT / f"reports/r2/m2ad_tests_{agg}.json"
    if t.exists():
        T = json.load(open(t))
        if "all|M3" in T:
            M["MtwoRho"] = f"{T['all|M3']['est']:+.2f}"
            M["MtwoRhoCI"] = f"[{T['all|M3']['lo']:+.2f},\\,{T['all|M3']['hi']:+.2f}]"
            M["MtwoRhoP"] = f"{T['all|M3']['p']:.3f}"
    p = ROOT / "reports/r2/m2ad_physical.json"
    if p.exists():
        P = json.load(open(p))["by_illum"]
        ts = sorted(P, key=lambda j: P[j]["tone_idx"])
        M["MtwoToneLo"] = f"{P[ts[0]]['tone_idx']:.2f}"
        M["MtwoToneHi"] = f"{P[ts[-1]]['tone_idx']:.2f}"
        M["MtwoToneLoI"] = ts[0]
        M["MtwoToneHiI"] = ts[-1]
        # 最空间化档与最色调化档上的架构对比 —— M3 的两个端点
        if us and all(b in A for b in ORDER[:4]):
            for tag, j in (("Lo", ts[0]), ("Hi", ts[-1])):
                uj = [u for u in us if u[2] == j]
                if not uj: continue
                a = ((dvec(A, ORDER[0], "all", uj) - dvec(A, ORDER[2], "all", uj)) +
                     (dvec(A, ORDER[1], "all", uj) - dvec(A, ORDER[3], "all", uj))) / 2
                M[f"MtwoArchAt{tag}"] = f"{a.mean():+.4f}"
    bad = [k for k in M if not k.isalpha()]
    assert not bad, f"LaTeX 宏名只能含字母，违规: {bad}"   # \fFoo9 会被解析成 \fFoo 加字面 9
    (OUT / "facts_m2ad.tex").write_text(
        "\n".join(rf"\newcommand{{\f{k}}}{{{v}}}" for k, v in sorted(M.items())) + "\n")
    return len(M)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    A = load()
    print(f"  载入 backbone: {', '.join(NICE.get(b,b) for b in sorted(A))}" if A else "  无 M2AD 结果")
    print(f"  tab_m2ad.tex         ({tab_m2ad(A) or 'pending'})")
    print(f"  tab_m2ad_full.tex    ({'ok' if tab_m2ad_full(A) else 'pending'})")
    print(f"  tab_m2ad_phys.tex    ({tab_m2ad_phys() or 'pending'})")
    print(f"  tab_m2ad_agg.tex     ({tab_m2ad_agg() or 'pending'})")
    print(f"  facts_m2ad.tex       ({facts_m2ad(A)} macros)")


if __name__ == "__main__":
    main()
