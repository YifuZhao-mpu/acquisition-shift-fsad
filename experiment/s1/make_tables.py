#!/usr/bin/env python3
r"""从结果文件生成论文的 LaTeX 表格 —— 论文里不出现手抄的数字。\n\n输出到 paper/tables/*.tex，由 main.tex 用 \input 引入。\n每次新配置跑完后重跑本脚本即可刷新全部表格。\n"""
from __future__ import annotations
import os
import glob, json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(os.environ.get("IADSHIFT_ROOT",
                    Path(__file__).resolve().parents[2]))   # experiment/s1/x.py -> 项目根
OUT = ROOT / "paper/tables"
TONAL = ["exposure", "gamma", "wb"]; SPATIAL = ["gradient", "specular", "shadow"]
SEVS = (0.33, 0.67, 1.0)
RNG = np.random.default_rng(20260920)

NICE = {"dinov2_vits14": r"DINOv2 ViT-S/14", "dinov2_vitb14": r"DINOv2 ViT-B/14",
        "deit_small_patch16": r"DeiT-S/16", "resnet50": r"ResNet-50 (sup.)",
        "dino_resnet50": r"DINO ResNet-50", "wide_resnet50_2": r"WideResNet-50-2",
        "resnet50_l3l4": r"ResNet-50 (l3l4)", "dino_resnet50_l3l4": r"DINO RN-50 (l3l4)",
        "dinov2_vits14_b6": r"DINOv2 ViT-S/14 (b6)"}
MODE_NICE = {"none": r"clean", "exposure": r"\textsc{exposure}", "gamma": r"\textsc{gamma}",
             "wb": r"\textsc{wb}", "gradient": r"\textsc{gradient}",
             "specular": r"\textsc{specular}", "shadow": r"\textsc{shadow}"}


def _unit_means(rows):
    d = defaultdict(lambda: defaultdict(list))
    for r in rows:
        d[(r["mode"], r["severity"])][(r["category"], r["defect"])].append(r["auroc"])
    return {k: {u: float(np.mean(v)) for u, v in m.items()} for k, m in d.items()}


def load_all():
    cfg = {}
    for f in sorted(glob.glob(str(ROOT / "reports/xbb_*.json"))) + \
             sorted(glob.glob(str(ROOT / "reports/r2/xbb2_*.json"))):
        d = json.load(open(f)); m = d["meta"]
        k = (m["backbone"], m["agg"])
        cfg[k] = _unit_means(d["rows"])
        if m.get("partial"):
            print(f"  [PARTIAL] {k[0]}/{k[1]}: {len(m['categories'])}/15 categories")
    rows = []
    for f in sorted(glob.glob(str(ROOT / "reports/mvtec_illum_g*.json"))):
        rows += json.load(open(f))["rows"]
    if rows and ("dinov2_vits14", "meantop1p") not in cfg:
        cfg[("dinov2_vits14", "meantop1p")] = _unit_means(rows)
    return cfg


def paired(d, B=10000):
    d = np.asarray(d, float); n = len(d)
    bs = d[RNG.integers(0, n, (B, n))].mean(1)
    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return d.mean(), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float(p)


def loss(M, modes, sevs, units):
    b = M[("none", 0.0)]
    return np.array([np.mean([b[u] - M[(m, s)][u] for m in modes for s in sevs]) for u in units])


def stars(p):
    return r"$^{***}$" if p < 1e-3 else (r"$^{**}$" if p < 1e-2 else (r"$^{*}$" if p < .05 else ""))


# ───────────────── Table 1: severity-matched ratio ─────────────────
def tab_ratio(cfg, agg="meantop1p"):
    keys = [k for k in cfg if k[1] == agg]
    order = ["dinov2_vitb14", "dinov2_vits14", "deit_small_patch16",
             "dino_resnet50", "resnet50", "wide_resnet50_2",
             "dinov2_vits14_b6", "resnet50_l3l4", "dino_resnet50_l3l4"]
    keys = [(b, agg) for b in order if (b, agg) in cfg]
    L = [r"\begin{table*}[t]",
         r"\caption{Degradation of image AUROC by perturbation group, at \emph{matched} severity.",
         r"$T$ = mean loss over the three tone perturbations, $S$ = over the three spatial ones.",
         r"A double dissociation would require $S/T<1$ for one representation family; no cell",
         r"in the table is below 1. All losses are relative to each unit's own unperturbed",
         r"baseline, averaged over 73 (category, defect) units and 8 support draws.}",
         r"\label{tab:ratio}\centering\small",
         r"\begin{tabular}{@{}l" + "ccc" * 3 + r"@{}}", r"\toprule",
         r"& \multicolumn{3}{c}{severity $0.33$} & \multicolumn{3}{c}{severity $0.67$}"
         r" & \multicolumn{3}{c}{severity $1.0$}\\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
         r"Representation & $T$ & $S$ & $S/T$ & $T$ & $S$ & $S/T$ & $T$ & $S$ & $S/T$\\",
         r"\midrule"]
    for k in keys:
        M = cfg[k]; u = sorted(M[("none", 0.0)]); cells = []
        for s in SEVS:
            t = loss(M, TONAL, [s], u).mean(); sp = loss(M, SPATIAL, [s], u).mean()
            cells += [f"{t:.3f}", f"{sp:.3f}", f"{sp/t:.1f}" if t > 1e-6 else "---"]
        L.append(NICE.get(k[0], k[0]) + " & " + " & ".join(cells) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (OUT / "tab_ratio.tex").write_text("\n".join(L) + "\n")
    return len(keys)


# ───────────────── Table 2: paired interaction ─────────────────
def tab_interaction(cfg, agg="meantop1p"):
    order = ["dinov2_vits14", "deit_small_patch16", "dino_resnet50", "resnet50",
             "dinov2_vitb14", "wide_resnet50_2"]
    keys = [(b, agg) for b in order if (b, agg) in cfg]
    L = [r"\begin{table}[t]",
         r"\caption{Paired interaction contrast $D_u = S_u - T_u$ per (category, defect) unit,",
         r"with 10{,}000-resample bootstrap 95\% CI and exact Wilcoxon signed-rank $p$.",
         r"A double dissociation requires $D<0$ for one family and $D>0$ for the other;",
         r"every entry is positive. $^{*}p<.05$, $^{**}p<.01$, $^{***}p<.001$.}",
         r"\label{tab:interaction}\centering\small",
         r"\begin{tabular}{@{}llr@{}}", r"\toprule",
         r"Representation & Severity & $D$ [95\% CI]\\", r"\midrule"]
    for k in keys:
        M = cfg[k]; u = sorted(M[("none", 0.0)])
        for i, s in enumerate(SEVS):
            D = loss(M, SPATIAL, [s], u) - loss(M, TONAL, [s], u)
            m, lo, hi, p = paired(D)
            name = NICE.get(k[0], k[0]) if i == 0 else ""
            L.append(f"{name} & {s} & ${m:+.3f}$ $[{lo:+.3f},{hi:+.3f}]${stars(p)}" + r"\\")
        L.append(r"\addlinespace[2pt]")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "tab_interaction.tex").write_text("\n".join(L) + "\n")


# ───────────────── Table 3: all 19 conditions, backbone contrast ─────────────────
def tab_conditions(cfg, a, b, agg="meantop1p", fname="tab_conditions.tex"):
    A, B = cfg.get((a, agg)), cfg.get((b, agg))
    if A is None or B is None:
        (OUT / fname).write_text("%% pending: " + f"{a} vs {b}\n"); return False
    u = sorted(set(A[("none", 0.0)]) & set(B[("none", 0.0)]))
    lab = "tab:" + fname[4:-4].replace("_", "")   # tab_conditions_cnn.tex -> tab:conditionscnn
    L = [r"\begin{table*}[t]",
         rf"\caption{{All 19 conditions: $\Delta = \AUROC(\text{{{NICE.get(a,a)}}}) - "
         rf"\AUROC(\text{{{NICE.get(b,b)}}})$, paired over {len(u)} (category, defect) units.",
         (r"Rows marked $\dagger$ are those a selective report would naturally pick; the "
          r"unmarked rows reverse the story at the strong end. "
          if lab == "tab:conditions" else ""),
         r"$^{*}p<.05$, $^{**}p<.01$, $^{***}p<.001$ (exact Wilcoxon).}",
         rf"\label{{{lab}}}\centering\small",
         r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
         rf"Condition & Sev. & {NICE.get(a,a)} & {NICE.get(b,b)} & $\Delta$ [95\% CI]\\",
         r"\midrule"]
    L = [x for x in L if x != ""]   # 空串会在 \caption{} 里变成空行 = \par，LaTeX 报错
    SEL = {("none", 0.0), ("gamma", 1.0), ("exposure", 1.0),
           ("specular", 0.33), ("shadow", 0.67), ("gradient", 0.67)}
    conds = [("none", 0.0)] + [(m, s) for m in TONAL + SPATIAL for s in SEVS]
    prev = None
    for c in conds:
        x = np.array([A[c][t] for t in u]); y = np.array([B[c][t] for t in u])
        m, lo, hi, p = paired(x - y)
        nm = MODE_NICE[c[0]] if c[0] != prev else ""
        prev = c[0]
        dag = r"$\dagger$" if (c in SEL and lab == "tab:conditions") else ""
        sev = "---" if c[0] == "none" else f"{c[1]}"
        L.append(f"{nm}{dag} & {sev} & {x.mean():.4f} & {y.mean():.4f} & "
                 f"${m:+.4f}$ $[{lo:+.3f},{hi:+.3f}]${stars(p)}" + r"\\")
        if c[1] == 1.0: L.append(r"\addlinespace[2pt]")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (OUT / fname).write_text("\n".join(L) + "\n")
    return True


# ───────────────── Table 4: 2x2 identification ─────────────────
GRID = {("ViT", "self-sup."): "dinov2_vits14", ("ViT", "supervised"): "deit_small_patch16",
        ("CNN", "self-sup."): "dino_resnet50", ("CNN", "supervised"): "resnet50"}


def tab_2x2(cfg, agg="meantop1p"):
    """R2：2x2 识别设计（单栏紧凑版，T-II 10 页限制）。"""
    have = {k: cfg.get((v, agg)) for k, v in GRID.items()}
    if any(v is None for v in have.values()):
        miss = [GRID[k] for k, v in have.items() if v is None]
        (OUT / "tab_2x2.tex").write_text("%% pending backbones: " + ", ".join(miss) + "\n")
        return False
    u = sorted(set.intersection(*[set(v[("none", 0.0)]) for v in have.values()]))
    MIN_UNITS = 20          # 单元太少时配对 CI 无意义，宁可标 pending
    if len(u) < MIN_UNITS:
        (OUT / "tab_2x2.tex").write_text(
            f"%% pending: only {len(u)} units common to all four cells (need >= {MIN_UNITS})\n")
        return False
    partial = len(u) < 73
    val = {}
    for k, M in have.items():
        t = loss(M, TONAL, list(SEVS), u); sp = loss(M, SPATIAL, list(SEVS), u)
        val[k] = dict(D=sp - t, t=t.mean(), sp=sp.mean(),
                      clean=np.mean([M[("none", 0.0)][x] for x in u]))
    note = (rf" Computed on the {len(u)} units available for all four cells."
            if partial else "")
    L = [r"\begin{table}[t]",
         r"\caption{$2\times2$ identification. $D = S - T$ per unit (three severities",
         r"pooled); larger $D$ means more spatially than tonally vulnerable. Each contrast",
         r"in the lower block moves exactly one factor; the CNN arm holds the network graph",
         r"fixed and swaps only the weights. $\dagger$ marks a contrast whose bootstrap",
         r"interval excludes zero." + note + r"}",
         r"\label{tab:2x2}\centering\footnotesize",
         r"\setlength{\tabcolsep}{3.5pt}",
         r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
         r"Arch. & Pretraining & clean & $T$ & $S$ & $D$\\", r"\midrule"]
    for k in [("ViT", "self-sup."), ("ViT", "supervised"),
              ("CNN", "self-sup."), ("CNN", "supervised")]:
        v = val[k]
        L.append(f"{k[0]} & {k[1]} & {v['clean']:.3f} & {v['t']:.3f} & {v['sp']:.3f} & "
                 f"${v['D'].mean():+.3f}$" + r"\\")
    L += [r"\midrule",
          r"\multicolumn{6}{@{}l}{\emph{Contrasts in $D$} (paired over " + str(len(u))
          + r" units, bootstrap 95\% CI)}\\"]
    def row(lbl, x, y):
        m, lo, hi, p = paired(val[x]["D"] - val[y]["D"])
        mark = "" if lo < 0 < hi else r"$^{\dagger}$"   # 与题注的 bootstrap CI 一致
        L.append(r"\multicolumn{3}{@{}l}{" + lbl + r"} & \multicolumn{3}{r}{"
                 + f"${m:+.3f}$ $[{lo:+.3f},{hi:+.3f}]${mark}" + r"}\\")
    row(r"\quad pretraining, within ViT", ("ViT", "self-sup."), ("ViT", "supervised"))
    row(r"\quad pretraining, within CNN", ("CNN", "self-sup."), ("CNN", "supervised"))
    row(r"\quad architecture, self-sup.", ("ViT", "self-sup."), ("CNN", "self-sup."))
    row(r"\quad architecture, supervised", ("ViT", "supervised"), ("CNN", "supervised"))
    inter = (val[("ViT", "self-sup.")]["D"] - val[("ViT", "supervised")]["D"]) - \
            (val[("CNN", "self-sup.")]["D"] - val[("CNN", "supervised")]["D"])
    m, lo, hi, p = paired(inter)
    mark = "" if lo < 0 < hi else r"$^{\dagger}$"
    L.append(r"\multicolumn{3}{@{}l}{\quad pretraining $\times$ architecture} & "
             r"\multicolumn{3}{r}{" + f"${m:+.3f}$ $[{lo:+.3f},{hi:+.3f}]${mark}" + r"}\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "tab_2x2.tex").write_text("\n".join(L) + "\n")
    return True


def tab_physical(cfg, agg="meantop1p"):
    """S1 + R1 的加强版：把 severity 换算成可测物理量，并在**物理量配平**处比较两族。\n\n    这比按检测器效应量配平更可取：后者用被测量本身定义自变量，是循环的。\n    """
    pf = ROOT / "reports/r2/perturb_physical.json"
    if not pf.exists():
        (OUT / "tab_physical.tex").write_text("%% pending: perturb_physical.json\n"); return False
    P = json.load(open(pf))["desc"]
    L = [r"\begin{table*}[t]",
         r"\caption{Physical descriptors of each perturbation, measured on real MVTec images",
         r"(15 categories $\times$ 4 normal test images). $\Delta L$ is relative luminance change;",
         r"$A_{>10}$ the fraction of pixels changed by more than 10\%; $\Delta C$ the mean CIELab",
         r"chroma displacement; $c'/c$ the median ratio of local (31\,px) contrast.",
         r"The last column is the resulting image-AUROC loss for DINOv2 ViT-S/14.",
         r"\textsc{wb} is the strongest \emph{chromatic} perturbation in the grid",
         rf"($\Delta C = {P['wb|1.0']['dChroma']:.0f}$ at severity 1, five to thirty times any other)",
         r"while being the weakest in luminance, so its null is informative about chroma",
         r"rather than an artefact of a weak perturbation.}",
         r"\label{tab:physical}\centering\small",
         r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
         r"Perturbation & Sev. & $\Delta L$ (\%) & $\Delta L_{p95}$ (\%) & $A_{>10}$ (\%) &"
         r" $\Delta C$ & $c'/c$ & AUROC loss\\", r"\midrule"]
    V = cfg.get(("dinov2_vits14", agg))
    prev = None
    for mode in TONAL + SPATIAL:
        for s in SEVS:
            d = P.get(f"{mode}|{s}")
            if d is None: continue
            lo = ""
            if V is not None:
                u = sorted(V[("none", 0.0)])
                lo = f"{np.mean([V[('none',0.0)][x] - V[(mode,s)][x] for x in u]):.3f}"
            nm = MODE_NICE[mode] if mode != prev else ""; prev = mode
            L.append(f"{nm} & {s} & {d['dL_mean']:.1f} & {d['dL_p95']:.1f} & {d['area_10']:.0f} & "
                     f"{d.get('dChroma', float('nan')):.1f} & {d['contrast']:.2f} & {lo}" + r"\\")
        L.append(r"\addlinespace[2pt]")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (OUT / "tab_physical.tex").write_text("\n".join(L) + "\n")
    return True


def tab_physmatch(cfg, agg="meantop1p", targets=(15.0, 25.0)):
    """在**相同物理幅度**($\\Delta L$)处比较两族的 AUROC 损失。\n\n    每个 mode 的 (dL_mean, loss) 三点加上原点 (0,0) 做单调线性插值；\n    只在该 mode 的实测范围内取值，绝不外推（wb 因此被排除并如实注明）。\n    """
    pf = ROOT / "reports/r2/perturb_physical.json"
    if not pf.exists():
        (OUT / "tab_physmatch.tex").write_text("%% pending\n"); return False
    P = json.load(open(pf))["desc"]
    order = ["dinov2_vits14", "deit_small_patch16", "dino_resnet50", "resnet50",
             "dinov2_vitb14", "wide_resnet50_2"]
    keys = [(b, agg) for b in order if (b, agg) in cfg]
    if not keys:
        (OUT / "tab_physmatch.tex").write_text("%% pending\n"); return False

    def interp(mode, losses, tgt):
        xs = [0.0] + [P[f"{mode}|{s}"]["dL_mean"] for s in SEVS]
        ys = [0.0] + losses
        if tgt > xs[-1]: return None                    # 不外推
        return float(np.interp(tgt, xs, ys))

    L = [r"\begin{table}[t]",
         r"\caption{Loss at \emph{matched physical magnitude}. Each perturbation's",
         r"dose--response is interpolated on the measured $\Delta L$ axis (Table~\ref{tab:physical})",
         r"and read off at a common $\Delta L$; no extrapolation is performed, so \textsc{wb}",
         r"(maximum $\Delta L\approx9\%$) is excluded and marked. Matching on a physical",
         r"quantity avoids the circularity of matching on the detector's own response.",
         r"We report $S-T$ rather than $S/T$ because $T$ is near zero for some",
         r"representations at these magnitudes, which makes a ratio unstable.}",
         r"\label{tab:physmatch}\centering\footnotesize",
         r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{@{}l" + "rrr" * len(targets) + r"@{}}", r"\toprule"]
    hdr = " & ".join(rf"\multicolumn{{3}}{{c}}{{$\Delta L={t:.0f}\%$}}" for t in targets)
    L += [r"Representation & " + hdr + r"\\",
          " ".join(rf"\cmidrule(lr){{{2+3*i}-{4+3*i}}}" for i in range(len(targets))),
          r" & " + " & ".join([r"$T$ & $S$ & $S-T$"] * len(targets)) + r"\\", r"\midrule"]
    for k in keys:
        M = cfg[k]; u = sorted(M[("none", 0.0)]); cells = []
        for tgt in targets:
            gt, gs = [], []
            for mode in TONAL:
                if mode == "wb": continue
                ls = [float(np.mean([M[("none", 0.0)][x] - M[(mode, s)][x] for x in u])) for s in SEVS]
                v = interp(mode, ls, tgt)
                if v is not None: gt.append(v)
            for mode in SPATIAL:
                ls = [float(np.mean([M[("none", 0.0)][x] - M[(mode, s)][x] for x in u])) for s in SEVS]
                v = interp(mode, ls, tgt)
                if v is not None: gs.append(v)
            if gt and gs:
                t, sp = float(np.mean(gt)), float(np.mean(gs))
                cells += [f"{t:.3f}", f"{sp:.3f}", f"{sp-t:+.3f}"]
            else:
                cells += ["---", "---", "---"]
        L.append(NICE.get(k[0], k[0]) + " & " + " & ".join(cells) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "tab_physmatch.tex").write_text("\n".join(L) + "\n")
    return True


# ───────────── R12：来源池等量化前后的受控支撑来源实验 ─────────────
def tab_sourcepool():
    f = ROOT / "reports/r2/source_equalpool_compare.json"
    if not f.exists():
        (OUT / "tab_sourcepool.tex").write_text("%% pending\n"); return False
    C = json.load(open(f))
    DOMS = ["same", "illumination", "view", "background"]
    L = [r"\begin{table}[t]",
         r"\caption{Controlled support-source experiment on AeBAD-S ($k=6$, 32 randomised",
         r"replicates). $G_d$ is the within-bank diversity contrast; 6B/6I/6V are pure-source",
         r"banks and MIX the balanced $2{+}2{+}2$ mixture. The lower block repeats the whole",
         r"experiment with all three source pools randomly subsampled to the size of the",
         r"smallest ($N=62$), removing the pool-size confound. Both the null diversity effect",
         r"and the counter-intuitive ordering survive.}",
         r"\label{tab:sourcepool}\centering\footnotesize",
         r"\setlength{\tabcolsep}{2.2pt}",
         r"\begin{tabular}{@{}l r@{\,}c rrrr@{}}",
         r"\toprule",
         r"Domain & $G_d$ & 95\% CI & 6B & 6I & 6V & MIX\\",
         r"\midrule",
         r"\multicolumn{7}{@{}l}{\emph{Original pools} (265 / 62 / 194)}\\"]
    for d in DOMS:
        v = C["orig"][d]
        L.append("\\quad \\texttt{%s} & $%+.3f$ & --- & %.3f & %.3f & %.3f & %.3f\\\\"
                 % (d, v["G"], v["6B"], v["6I"], v["6V"], v["MIX"]))
    L.append(r"\addlinespace[3pt]")
    L.append(r"\multicolumn{7}{@{}l}{\emph{Pools equalised to $N=62$}}\\")
    for d in DOMS:
        v = C["equal"][d]
        L.append("\\quad \\texttt{%s} & $%+.3f$ & \\scriptsize$[%+.3f,%+.3f]$ & %.3f & %.3f & %.3f & %.3f\\\\"
                 % (d, v["G"], v["lo"], v["hi"], v["6B"], v["6I"], v["6V"], v["MIX"]))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "tab_sourcepool.tex").write_text("\n".join(L) + "\n")
    return True


# ───────────── 行内数字宏：正文里不出现手抄数字 ─────────────
def facts(cfg, agg="meantop1p"):
    M = {}
    def put(name, val): M[name] = val
    V, C = cfg.get(("dinov2_vits14", agg)), cfg.get(("wide_resnet50_2", agg))
    if V and C:
        u = sorted(set(V[("none", 0.0)]) & set(C[("none", 0.0)]))
        put("NUnits", str(len(u)))
        for lbl, c in [("Clean", ("none", 0.0)), ("GammaHi", ("gamma", 1.0)),
                       ("ExpoHi", ("exposure", 1.0)), ("SpecLo", ("specular", 0.33)),
                       ("SpecHi", ("specular", 1.0)), ("ShadMid", ("shadow", 0.67)),
                       ("GradHi", ("gradient", 1.0)), ("GradMid", ("gradient", 0.67))]:
            x = np.array([V[c][t] for t in u]); y = np.array([C[c][t] for t in u])
            m, lo, hi, pv = paired(x - y)
            put(f"Vit{lbl}", f"{x.mean():.4f}"); put(f"Cnn{lbl}", f"{y.mean():.4f}")
            put(f"D{lbl}", f"{m:+.4f}"); put(f"D{lbl}CI", f"[{lo:+.4f},\\,{hi:+.4f}]")
        d = {t: V[("none", 0.0)][t] - C[("none", 0.0)][t] for t in u}
        tot = sum(d.values())
        big = [t for t in u if t[0] in ("capsule", "grid", "toothbrush")]
        put("CleanConcPct", f"{sum(d[t] for t in big)/tot*100:.0f}")
        put("CleanConcN", str(len(big)))
        put("CleanCnnWins", str(sum(1 for t in u if d[t] < 0)))
        bycat = defaultdict(list)
        for t in u: bycat[t[0]].append(V[("specular", 0.33)][t] - C[("specular", 0.33)][t])
        put("SpecLoCatCnn", str(sum(1 for v in bycat.values() if np.mean(v) < 0)))
        put("SpecLoCatN", str(len(bycat)))
    for bb, nm in [("dinov2_vits14", "Vit"), ("wide_resnet50_2", "Cnn")]:
        Mx = cfg.get((bb, agg))
        if not Mx: continue
        u = sorted(Mx[("none", 0.0)])
        t1 = loss(Mx, TONAL, [1.0], u).mean()
        sp_mixed = np.mean([Mx[("none", 0.0)][x] - Mx[c][x]
                            for c in [("specular", 0.33), ("shadow", 0.67), ("gradient", 0.67)]
                            for x in u])
        put(f"{nm}Unmatched", f"{sp_mixed/t1:.2f}")
        put(f"{nm}MatchedLo", f"{loss(Mx,SPATIAL,[0.33],u).mean()/loss(Mx,TONAL,[0.33],u).mean():.1f}")
        put(f"{nm}MatchedHi", f"{loss(Mx,SPATIAL,[1.0],u).mean()/loss(Mx,TONAL,[1.0],u).mean():.1f}")

    pf = ROOT / "reports/r2/perturb_physical.json"
    if pf.exists():
        P = json.load(open(pf))["desc"]
        put("WbChromaHi", f"{P['wb|1.0']['dChroma']:.0f}")
        put("WbDLHi", f"{P['wb|1.0']['dL_mean']:.0f}")
        put("ExpoDLHi", f"{P['exposure|1.0']['dL_mean']:.0f}")
        put("GammaDLHi", f"{P['gamma|1.0']['dL_mean']:.0f}")
        put("ShadDLHi", f"{P['shadow|1.0']['dL_mean']:.0f}")
        put("SpecPHi", f"{P['specular|1.0']['dL_p95']:.0f}")
        put("SpecPLo", f"{P['specular|0.33']['dL_p95']:.0f}")
        put("SpecDLLo", f"{P['specular|0.33']['dL_mean']:.0f}")
        put("SpecDLMid", f"{P['specular|0.67']['dL_mean']:.0f}")
        put("SpecDLHi", f"{P['specular|1.0']['dL_mean']:.0f}")
        put("GammaContrast", f"{P['gamma|1.0']['contrast']:.2f}")

    cf = ROOT / "reports/r2/c3_power.json"
    if cf.exists():
        C3 = json.load(open(cf))
        put("CThreeNneg", str(C3["n_illum"][0])); put("CThreeNpos", str(C3["n_illum"][1]))
        put("CThreeCIhm", f"{C3['hm_ci_width_illum']:.3f}")
        put("CThreeCImc", f"{C3['mc_ci_width_illum']:.3f}")
        put("CThreeP", f"{C3['mc_p']:.2f}")
        put("CThreeMedian", f"{C3['mc_median_gap']:.3f}")
        put("CThreePct", f"{C3['gap_pct_of_ci']:.0f}")
        put("CThreeSEdiff", f"{C3['hm_se_diff']:.3f}")

    pv = ROOT / "reports/r2/prereg_verify.json"
    if pv.exists():
        Vp = json.load(open(pv))
        put("PregUnits", str(Vp["n_units"])); put("PregExcl", str(Vp["counts"]["X"]))
        put("PregNS", str(Vp["counts"]["S"])); put("PregNA", str(Vp["counts"]["A"]))
        put("PregNG", str(Vp["counts"]["G"]))
        NUM = {"P1": "POne", "P2": "PTwo", "P3": "PThree"}
        SEV = {"0.33": "Lo", "0.67": "Mid", "1.0": "Hi"}
        for t in Vp["tests"]:
            a_, b_ = t["name"].split("@")
            put(f"Preg{NUM[a_]}{SEV[b_]}", f"{t['diff']:+.3f}")
            put(f"Preg{NUM[a_]}{SEV[b_]}P", f"{t['p_holm']:.2f}")
        det = {t["name"]: t["detail"] for t in Vp["tests"]}
        for k_, lbl in [("P1@0.33", "Lo"), ("P1@1.0", "Hi")]:
            nums = re.findall(r"=([0-9.]+)", det[k_])
            put(f"PregSloss{lbl}", nums[0]); put(f"PregGloss{lbl}", nums[1])

    mf = ROOT / "reports/r2/mask_bug_verify.json"
    if mf.exists():
        Mk = json.load(open(mf))
        put("MaskIoU", f"{Mk['overall']['iou_mean']:.3f}")
        put("MaskPct", f"{Mk['overall']['pct_differ']:.2f}")
        put("MaskN", str(Mk['overall']['n']))
        g = Mk["per_group"]
        put("MaskIllumLo", f"{g['illumination / fracture']['iou_min']:.2f}")
        put("MaskIllumHi", f"{g['illumination / fracture']['iou_max']:.2f}")
        put("MaskSameLo", f"{g['same / good']['iou_min']:.2f}")
        put("MaskSameHi", f"{g['same / good']['iou_max']:.2f}")

    MV = ROOT / "data/mvtec_anomaly_detection"
    if MV.exists():
        g = {d.name: len(list((d / "test/good").glob("*.png"))) for d in MV.iterdir() if d.is_dir()}
        half = {k: v // 2 for k, v in g.items()}
        put("CalNmin", str(min(g.values()))); put("CalNmax", str(max(g.values())))
        put("CalHalfMin", str(min(half.values()))); put("CalHalfMax", str(max(half.values())))
        put("CalFloorBest", f"{100/max(half.values()):.1f}")
        put("CalFloorWorst", f"{100/min(half.values()):.1f}")
        put("CalTotal", str(sum(g.values())))

    hf = ROOT / "reports/r2/aebad_hygiene.json"
    if hf.exists():
        H = json.load(open(hf))
        put("AeAD", str(H["payload_appledouble"])); put("AeDS", str(H["payload_dsstore"]))
        put("AeADall", str(H["archive_appledouble"]))
        r_ = H["leading_invalid_run"]
        put("AeRunBg", str(r_.get("background", 0))); put("AeRunView", str(r_.get("view", 0)))
        put("AeRunIll", str(r_.get("illumination", 0)))
        put("AeSeedsK", str(r_.get("background", 0) // 4))

    sf = ROOT / "reports/r2/source_equalpool_compare.json"
    if sf.exists():
        Sc = json.load(open(sf))
        for tag, key in [("Orig", "orig"), ("Eq", "equal")]:
            v = Sc[key]["illumination"]
            put(f"Src{tag}SixB", f"{v['6B']:.3f}"); put(f"Src{tag}SixI", f"{v['6I']:.3f}")
            put(f"Src{tag}SixV", f"{v['6V']:.3f}"); put(f"Src{tag}Mix", f"{v['MIX']:.3f}")
            put(f"Src{tag}G", f"{v['G']:+.4f}")
        put("SrcEqGciLo", f"{Sc['equal']['illumination']['lo']:+.4f}")
        put("SrcEqGciHi", f"{Sc['equal']['illumination']['hi']:+.4f}")

    # 实际完成的表征数（摘要与正文引用，避免写死）
    full = [k for k in cfg if len(cfg[k].get(("none", 0.0), {})) >= 73]
    put("NRepr", str(len({k[0] for k in full})))
    put("NConfig", str(len(full)))
    WORD = {1:"one",2:"two",3:"three",4:"four",5:"five",6:"six",7:"seven",8:"eight",
            9:"nine",10:"ten",11:"eleven",12:"twelve"}
    put("NReprWord", WORD.get(len({k[0] for k in full}), str(len({k[0] for k in full}))))
    ef = ROOT / "reports/r2/engine_equiv.json"
    if ef.exists():
        E = json.load(open(ef))
        r_ = E.get("reference", {})
        put("EngUnits", str(r_.get("units", ""))); put("EngCorr", f"{r_.get('corr', 0):.8f}")
        _m = r_.get("max_abs_diff", 0.0)
        _e = int(np.floor(np.log10(_m))) if _m > 0 else 0
        put("EngMaxDiff", r"%.2f\times 10^{%d}" % (_m / 10 ** _e, _e))
        dist = r_.get("distribution", {})
        put("EngDist", ", ".join(f"{c} at ${v}$" for v, c in
                                 sorted(dist.items(), key=lambda x: float(x[0]))))
        m_ = E.get("batched_mvtec")
        if m_:
            put("EngBatchRows", f"{m_['rows']:,}".replace(",", "{,}"))
        b_ = E.get("batched_and_perdomain")
        if b_:
            put("EngBatchUnits", str(b_["units"]))
            put("EngBatchNonzero", str(b_["n_nonzero"]))
    ap_ = ROOT / "reports/r2/aebad_physical.json"
    if ap_.exists():
        AP = json.load(open(ap_))
        put("AeIllumDL", f"{AP['_summary']['illum_dL_pct']:.0f}")
        put("AeIllumDC", f"{AP['_summary']['illum_dC_ab']:.1f}")
        put("AeIllumHi", f"{AP['_summary']['illum_dhi_area']:+.1f}")
        put("AeLsame", f"{AP['same']['L_mean']:.0f}")
        put("AeLillum", f"{AP['illumination']['L_mean']:.0f}")
        put("AeBgDC", f"{np.hypot(AP['background']['a_mean']-AP['same']['a_mean'], AP['background']['b_mean']-AP['same']['b_mean']):.1f}")
    uf = ROOT / "reports/r2/uncertainty.json"
    if uf.exists():
        U = json.load(open(uf))
        for lbl, tag in [("clean", "Clean"), ("exposure@1.0", "Expo"),
                         ("specular@0.67", "Spec"), ("shadow@1.0", "Shad")]:
            if lbl not in U: continue
            u_ = U[lbl]
            put(f"Unc{tag}Sup", f"{u_['med_support']:.3f}")
            put(f"Unc{tag}Test", f"{u_['med_test']:.3f}")
            put(f"Unc{tag}Ratio", f"{u_['med_ratio']:.1f}")
            put(f"Unc{tag}N", str(u_["n"]))
            put(f"Unc{tag}Wider", str(u_["n_test_wider"]))
        rr = [U[k]["med_ratio"] for k in U]
        put("UncRatioLo", f"{min(rr):.1f}"); put("UncRatioHi", f"{max(rr):.1f}")
        tot = sum(U[k]["n"] for k in U); wid = sum(U[k]["n_test_wider"] for k in U)
        put("UncWiderPct", f"{wid/tot*100:.0f}")

    # AeBAD 真实域 2x2 分解宏
    import collections as _cc
    _rng = np.random.default_rng(7)
    _ae = {}
    for bb in ["dinov2_vits14", "deit_small_patch16", "dino_resnet50", "resnet50",
               "wide_resnet50_2"]:
        f2 = ROOT / f"reports/r2/aebad_{bb}_meantop1p.json"
        if not f2.exists(): continue
        d2 = json.load(open(f2)); by = _cc.defaultdict(dict)
        for r2 in d2["rows"]:
            if r2["scope"] == "all": by[r2["seed"]][r2["domain"]] = r2["auroc"]
        sd = sorted(by); bs_ = np.array([by[s2]["same"] for s2 in sd])
        _ae[bb] = {"same": float(bs_.mean())}
        for dom in ["illumination", "view", "background"]:
            v2 = np.array([by[s2][dom] for s2 in sd]); dd = v2 - bs_
            _ae[bb][dom] = float(dd.mean()); _ae[bb][dom + "_abs"] = float(v2.mean())
    if len(_ae) >= 4:
        SHORT = {"dinov2_vits14": "VitS", "deit_small_patch16": "VitSup",
                 "dino_resnet50": "CnnS", "resnet50": "CnnSup", "wide_resnet50_2": "Wrn"}
        for bb, tag in SHORT.items():
            if bb not in _ae: continue
            put(f"Ae{tag}Same", f"{_ae[bb]['same']:.3f}")
            put(f"Ae{tag}Illum", f"{_ae[bb]['illumination']:+.3f}")
            put(f"Ae{tag}IllumAbs", f"{_ae[bb]['illumination_abs']:.3f}")
        need = ["dinov2_vits14", "deit_small_patch16", "dino_resnet50", "resnet50"]
        if all(b2 in _ae for b2 in need):
            gI = lambda b2: _ae[b2]["illumination"]
            put("AePreVit", f"{gI('dinov2_vits14') - gI('deit_small_patch16'):+.3f}")
            put("AePreCnn", f"{gI('dino_resnet50') - gI('resnet50'):+.3f}")
            put("AeArchS", f"{gI('dinov2_vits14') - gI('dino_resnet50'):+.3f}")
            put("AeArchSup", f"{gI('deit_small_patch16') - gI('resnet50'):+.3f}")

    import glob as _g2
    _w = {}
    for f3 in _g2.glob(str(ROOT / "reports/r2/aebad_*_meantop1p.json")):
        m3 = json.load(open(f3))["meta"]
        if m3.get("wall_seconds"): _w[m3["backbone"]] = m3["wall_seconds"]
    if _w:
        put("EngWallMin", f"{min(_w.values())/60:.1f}")
        put("EngWallMax", f"{max(_w.values())/60:.0f}")
        put("EngWallN", str(len(_w)))
    # 2x2 按轴分解宏（色调轴 / 空间轴各自的四个对照）
    _G = {("ViT", "ssl"): "dinov2_vits14", ("ViT", "sup"): "deit_small_patch16",
          ("CNN", "ssl"): "dino_resnet50", ("CNN", "sup"): "resnet50"}
    _hv = {k: cfg[(v, agg)] for k, v in _G.items() if (v, agg) in cfg}
    if len(_hv) == 4:
        _u = sorted(set.intersection(*[set(v[("none", 0.0)]) for v in _hv.values()]))
        put("TwoByTwoUnits", str(len(_u)))
        for axis, modes, tag in [("T", TONAL, "Ton"), ("S", SPATIAL, "Spa")]:
            _L = {k: loss(_hv[k], modes, list(SEVS), _u) for k in _hv}
            for lbl, a2, b2 in [("PreVit", ("ViT", "ssl"), ("ViT", "sup")),
                                ("PreCnn", ("CNN", "ssl"), ("CNN", "sup")),
                                ("ArchS", ("ViT", "ssl"), ("CNN", "ssl")),
                                ("ArchSup", ("ViT", "sup"), ("CNN", "sup"))]:
                m4, lo4, hi4, p4 = paired(_L[a2] - _L[b2])
                put(f"X{tag}{lbl}", f"{m4:+.3f}")
                put(f"X{tag}{lbl}CI", f"[{lo4:+.3f},\\,{hi4:+.3f}]")
                put(f"X{tag}{lbl}Sig", "" if lo4 < 0 < hi4 else stars(p4))
            for k4, tg in [(("ViT", "ssl"), "VitS"), (("ViT", "sup"), "VitSup"),
                           (("CNN", "ssl"), "CnnS"), (("CNN", "sup"), "CnnSup")]:
                put(f"X{tag}{tg}", f"{_L[k4].mean():.3f}")

    # D 对照的宏（供正文按倍数描述，避免写死「显著/不显著」）
    if len(_hv) == 4:
        _D = {k: loss(_hv[k], SPATIAL, list(SEVS), _u) - loss(_hv[k], TONAL, list(SEVS), _u)
              for k in _hv}
        for lbl, a3, b3 in [("DPreVit", ("ViT", "ssl"), ("ViT", "sup")),
                            ("DPreCnn", ("CNN", "ssl"), ("CNN", "sup"))]:
            m5, lo5, hi5, _ = paired(_D[a3] - _D[b3])
            put(lbl, f"{m5:+.3f}"); put(lbl + "CI", f"[{lo5:+.3f},\\,{hi5:+.3f}]")
        _r = abs(paired(_D[("ViT","ssl")] - _D[("ViT","sup")])[0]) / \
             max(abs(paired(_D[("CNN","ssl")] - _D[("CNN","sup")])[0]), 1e-9)
        put("DPreRatio", f"{_r:.0f}")

    # 层深对照宏
    for a4, b4, tg in [("dinov2_vits14", "dinov2_vits14_b6", "Vit"),
                       ("resnet50", "resnet50_l3l4", "CnnSup"),
                       ("dino_resnet50", "dino_resnet50_l3l4", "CnnS")]:
        A4, B4 = cfg.get((a4, agg)), cfg.get((b4, agg))
        if A4 is None or B4 is None: continue
        u4 = sorted(set(A4[("none", 0.0)]) & set(B4[("none", 0.0)]))
        if len(u4) < 20: continue
        for ax, modes in [("T", TONAL), ("S", SPATIAL)]:
            sh = loss(A4, modes, list(SEVS), u4).mean()
            dp = loss(B4, modes, list(SEVS), u4).mean()
            put(f"Dep{tg}{ax}sh", f"{sh:.3f}"); put(f"Dep{tg}{ax}dp", f"{dp:.3f}")
            put(f"Dep{tg}{ax}d", f"{dp - sh:+.3f}")
        put(f"Dep{tg}Csh", f"{np.mean([A4[('none',0.0)][x] for x in u4]):.3f}")
        put(f"Dep{tg}Cdp", f"{np.mean([B4[('none',0.0)][x] for x in u4]):.3f}")
        put(f"Dep{tg}Units", str(len(u4)))

    cf2 = ROOT / "reports/r2/cost_bench.json"
    if cf2.exists():
        R2 = json.load(open(cf2))["results"]
        for bb, tg in [("deit_small_patch16", "Deit"), ("dinov2_vits14", "Vit"),
                       ("dinov2_vits14_b6", "VitMid"), ("resnet50_l3l4", "CnnDeep"),
                       ("resnet50", "CnnShal"), ("dinov2_vitb14", "VitB")]:
            if bb in R2:
                put(f"Cost{tg}", f"{R2[bb]['total_ms']:.0f}")
                put(f"Cost{tg}Match", f"{R2[bb]['match_ms']:.0f}")
                put(f"Cost{tg}Vram", f"{R2[bb]['vram_mib']:.0f}")
    bad = [k for k in M if not k.isalpha()]
    assert not bad, f"LaTeX 宏名只能含字母，违规: {bad}"   # \fFoo9 会被解析成 \fFoo 加字面 9
    lines = [r"%% AUTO-GENERATED by experiment/s1/make_tables.py — do not edit by hand.",
             r"%% Every inline number in the manuscript comes from here."]
    for k, v in sorted(M.items()):
        lines.append(r"\newcommand{\f" + k + r"}{" + v + r"}")
    (OUT / "facts.tex").write_text("\n".join(lines) + "\n")
    return len(M)


# ───────────── R10 + S8：部署口径（沿用阈值 vs 漂移后重标定） ─────────────
def tab_deploy(backbone="dinov2_vits14", agg="meantop1p", tgt=0.10):
    """install-then-drift 与 recalibrate 两条臂合并成一张单栏表。"""
    import importlib.util as _iu
    spec = _iu.spec_from_file_location("ars", Path(__file__).parent / "analyze_r2_scores.py")
    ars = _iu.module_from_spec(spec); spec.loader.exec_module(ars)
    f = ROOT / f"reports/r2/scores_{backbone}_{agg}.npz"
    if not f.exists():
        (OUT / "tab_deploy.tex").write_text("%% pending: " + f.name + "\n"); return False
    _, data = ars.load_scores(f)
    rec = ars.recalibration_gain(data, fpr_targets=(tgt,))
    base = np.array(rec[("none", 0.0, tgt)])
    L = [r"\begin{table}[t]",
         rf"\caption{{Install-then-drift evaluation, {NICE.get(backbone, backbone)}. The",
         rf"threshold is set on nominal-condition normal parts at a target false-alarm rate",
         rf"of {tgt:.0%}, then held (\emph{{keep}}) or re-estimated on normal parts imaged",
         r"under the drifted condition (\emph{re-cal}). A shift whose detection rate",
         r"returns to baseline after re-calibration is a \emph{threshold} problem; one",
         r"whose detection rate stays depressed is a \emph{representation} problem.}",
         r"\label{tab:deployfull}\centering\footnotesize",
         r"\setlength{\tabcolsep}{3.5pt}",
         r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
         r"& & \multicolumn{2}{c}{keep $\tau$} & \multicolumn{2}{c}{re-cal $\tau$} & irrecov.\\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r"Perturbation & Sev. & FPR & TPR & FPR & TPR & $\Delta$TPR\\", r"\midrule",
         "nominal & --- & %.3f & %.3f & %.3f & %.3f & ---\\\\"
         % (base[:, 0].mean(), base[:, 1].mean(), base[:, 2].mean(), base[:, 3].mean()),
         r"\addlinespace[2pt]"]
    prev = None
    for m in ["gamma", "exposure", "wb"] + SPATIAL:
        for sv in SEVS:
            k = (m, sv, tgt)
            if k not in rec: continue
            v = np.array(rec[k])
            nm = MODE_NICE[m] if m != prev else ""; prev = m
            L.append("%s & %s & %.3f & %.3f & %.3f & %.3f & %.3f\\\\"
                     % (nm, sv, v[:, 0].mean(), v[:, 1].mean(), v[:, 2].mean(), v[:, 3].mean(),
                        base[:, 3].mean() - v[:, 3].mean()))
        L.append(r"\addlinespace[2pt]")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "tab_deploy.tex").write_text("\n".join(L) + "\n")
    b = base[:, 3].mean()
    GROUPS = [("Mono", ["gamma"]), ("Clip", ["exposure", "wb"]), ("Spat", SPATIAL)]

    # 主文用的 3 行摘要表（按可恢复性分组）；完整 19 行表进补充材料
    S = [r"\begin{table}[t]",
         r"\caption{Install-then-drift summary, DINOv2 ViT-S/14, threshold set on",
         rf"nominal-condition normal parts at a {int(tgt*100)}\% target. \emph{{keep}} holds the",
         r"install-time threshold; \emph{re-cal} re-estimates it on normal parts imaged under",
         r"the drifted condition. The last column is the detection rate that re-calibration",
         r"cannot recover, relative to the nominal baseline. Recoverability is set by the",
         r"kind of shift, not its magnitude: the group with the \emph{largest} luminance",
         r"change is the one that recovers completely. Supplementary Table~\ref{S-tab:deployfull} gives all 19",
         r"conditions.}",
         r"\label{tab:deploy}\centering\footnotesize",
         r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
         r"& \multicolumn{2}{c}{FPR} & TPR & irrecov.\\",
         r"\cmidrule(lr){2-3}",
         r"Shift group & keep & re-cal & re-cal & $\Delta$TPR\\", r"\midrule",
         "nominal & %.3f & %.3f & %.3f & ---\\\\" % (base[:, 0].mean(), base[:, 0].mean(), b),
         r"\addlinespace[2pt]"]
    NAMEG = {"Mono": r"monotone tone (\textsc{gamma})",
             "Clip": r"tone w/ clipping or chroma",
             "Spat": r"spatial illumination"}
    for gname, modes in GROUPS:
        vs = [np.array(rec[(m, sv, tgt)]) for m in modes for sv in SEVS if (m, sv, tgt) in rec]
        if not vs: continue
        S.append("%s & %.3f & %.3f & %.3f & %.3f\\\\" % (
            NAMEG[gname], np.mean([v[:, 0].mean() for v in vs]),
            np.mean([v[:, 2].mean() for v in vs]), np.mean([v[:, 3].mean() for v in vs]),
            np.mean([b - v[:, 3].mean() for v in vs])))
    S += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "tab_deploy_summary.tex").write_text("\n".join(S) + "\n")


    # 分组小结宏：按「可恢复性」而非按色调/空间分组
    macros = []
    for gname, modes in GROUPS:
        vs = [np.array(rec[(m, sv, tgt)]) for m in modes for sv in SEVS if (m, sv, tgt) in rec]
        if not vs: continue
        irr = [b - v[:, 3].mean() for v in vs]
        macros.append((f"Dep{gname}Keep", f"{np.mean([v[:,0].mean() for v in vs]):.3f}"))
        macros.append((f"Dep{gname}Recal", f"{np.mean([v[:,2].mean() for v in vs]):.3f}"))
        macros.append((f"Dep{gname}Irr", f"{np.mean(irr):.3f}"))
        macros.append((f"Dep{gname}IrrMax", f"{max(irr):.3f}"))
    for nm, key in [("GammaHi", ("gamma", 1.0)), ("ExpoHi", ("exposure", 1.0)),
                    ("ShadLo", ("shadow", 0.33)), ("SpecHi", ("specular", 1.0)),
                    ("GradHi", ("gradient", 1.0)), ("WbHi", ("wb", 1.0))]:
        k2 = (key[0], key[1], tgt)
        if k2 in rec:
            v = np.array(rec[k2])
            macros.append((f"DepIrr{nm}", f"{b - v[:,3].mean():.3f}"))
            macros.append((f"DepKeepFpr{nm}", f"{v[:,0].mean():.3f}"))
    macros.append(("DepBaseTpr", f"{b:.3f}"))
    macros.append(("DepBaseFpr", f"{base[:,0].mean():.3f}"))
    macros.append(("DepTarget", str(int(tgt * 100))))
    (OUT / "facts_deploy.tex").write_text(
        "\n".join(r"\newcommand{\f" + k + r"}{" + v + r"}" for k, v in macros) + "\n")
    return True


# ───────────── R3：AeBAD 真实域跨表征对照 ─────────────
def tab_aebad(agg="meantop1p"):
    import collections as _c
    rng = np.random.default_rng(7)
    order = ["dinov2_vits14", "deit_small_patch16", "dino_resnet50", "resnet50", "wide_resnet50_2"]
    rows_out, have = [], []
    for bb in order:
        f = ROOT / f"reports/r2/aebad_{bb}_{agg}.json"
        if not f.exists(): continue
        d = json.load(open(f))
        by = _c.defaultdict(dict)
        for r in d["rows"]:
            if r["scope"] == "all": by[r["seed"]][r["domain"]] = r["auroc"]
        seeds = sorted(by); DOMS = ["same", "illumination", "view", "background"]
        base = np.array([by[s]["same"] for s in seeds])
        cells = [f"{base.mean():.3f}"]
        for dom in DOMS[1:]:
            v = np.array([by[s][dom] for s in seeds]); dd = v - base
            bs = dd[rng.integers(0, len(dd), (10000, len(dd)))].mean(1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            star = "" if lo < 0 < hi else r"$^{*}$"
            cells.append(f"{v.mean():.3f} (${dd.mean():+.3f}${star})")
        rows_out.append((NICE.get(bb, bb), cells)); have.append(bb)
    if not rows_out:
        (OUT / "tab_aebad.tex").write_text("%% pending\n"); return False
    L = [r"\begin{table}[t]",
         r"\caption{AeBAD-S, real acquisition shift ($k=4$, 8 support draws, no masking).",
         r"Image AUROC per test domain; parentheses give the paired change from the matched",
         r"\texttt{same} domain, starred when the bootstrap 95\% interval excludes zero.}",
         r"\label{tab:aebad}\centering\scriptsize",
         r"\setlength{\tabcolsep}{2pt}",
         r"\begin{tabular}{@{}lrlll@{}}", r"\toprule",
         r"Representation & same & illum. & view & backgr.\\", r"\midrule"]
    for nm, cells in rows_out:
        L.append(nm + " & " + " & ".join(cells) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "tab_aebad.tex").write_text("\n".join(L) + "\n")
    return len(rows_out)


def tab_aebad_physical():
    f = ROOT / "reports/r2/aebad_physical.json"
    if not f.exists():
        (OUT / "tab_aebad_physical.tex").write_text("%% pending\n"); return False
    A = json.load(open(f))
    DOMS = ["same", "illumination", "view", "background"]
    KEYS = [("L_mean", r"$\bar{L}$"), ("L_std", r"$\sigma_L$"), ("chroma", r"chroma"),
            ("a_mean", r"$\bar{a}^*$"), ("b_mean", r"$\bar{b}^*$"),
            ("hi_area", r"hi.\ area \%"), ("contrast", r"$c_{31}$")]
    L = [r"\begin{table}[t]",
         r"\caption{Physical composition of the AeBAD-S test domains, measured on 60 normal",
         r"images per domain with the same descriptors used for the synthetic perturbations.",
         r"The \texttt{illumination} domain is dominated by a global luminance change",
         r"($\bar{L}$ falls by more than half) with only a modest chroma shift, which is the",
         r"axis on which the supervised CNN is weakest in the synthetic grid.}",
         r"\label{tab:aebadphys}\centering\footnotesize",
         r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}l" + "r" * len(KEYS) + r"@{}}", r"\toprule",
         r"Domain & " + " & ".join(lbl for _, lbl in KEYS) + r"\\", r"\midrule"]
    for d in DOMS:
        L.append(r"\texttt{" + d + "} & " + \
                 " & ".join(f"{A[d][k]:.2f}" for k, _ in KEYS) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "tab_aebad_physical.tex").write_text("\n".join(L) + "\n")
    return True


# ───────────── R2b：特征层深对照 ─────────────
DEPTH_PAIRS = [("resnet50", "resnet50_l3l4", r"ResNet-50 (sup.)"),
               ("dino_resnet50", "dino_resnet50_l3l4", r"DINO ResNet-50"),
               ("dinov2_vits14", "dinov2_vits14_b6", r"DINOv2 ViT-S/14")]


def tab_depth(cfg, agg="meantop1p"):
    """同一组权重、只改读出层深：检验色调敏感性是否由层深解释。"""
    rows = []
    for shallow, deep, nm in DEPTH_PAIRS:
        A, B = cfg.get((shallow, agg)), cfg.get((deep, agg))
        if A is None or B is None: continue
        u = sorted(set(A[("none", 0.0)]) & set(B[("none", 0.0)]))
        if len(u) < 20: continue
        r = dict(name=nm, n=len(u))
        for tag, modes in [("T", TONAL), ("S", SPATIAL)]:
            la = loss(A, modes, list(SEVS), u); lb = loss(B, modes, list(SEVS), u)
            m, lo, hi, pv = paired(lb - la)      # 深层 − 浅层
            r[tag] = (la.mean(), lb.mean(), m, lo, hi)
        r["cleanA"] = np.mean([A[("none", 0.0)][x] for x in u])
        r["cleanB"] = np.mean([B[("none", 0.0)][x] for x in u])
        rows.append(r)
    if not rows:
        (OUT / "tab_depth.tex").write_text("%% pending\n"); return False
    L = [r"\begin{table}[t]",
         r"\caption{Feature-depth control: the same weights read at a deeper site.",
         r"For the CNNs, \texttt{layer3}+\texttt{layer4} instead of",
         r"\texttt{layer2}+\texttt{layer3}; for the ViT, an intermediate block instead of",
         r"the last. $\Delta$ is (deeper $-$ shallower) loss, paired per unit with a",
         r"bootstrap 95\% CI; negative means the deeper read loses less. If feature depth",
         r"drove tone sensitivity, $\Delta T$ would be strongly negative for the CNNs.}",
         r"\label{tab:depth}\centering\footnotesize",
         r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
         r"Weights & clean (shal./deep) & $\Delta T$ & $\Delta S$\\", r"\midrule"]
    for r in rows:
        def cell(t):
            _, _, m, lo, hi = r[t]
            mark = "" if lo < 0 < hi else r"$^{\dagger}$"
            return f"${m:+.3f}$ $[{lo:+.3f},{hi:+.3f}]${mark}"
        L.append(f"{r['name']} & {r['cleanA']:.3f}\\,/\\,{r['cleanB']:.3f} & "
                 f"{cell('T')} & {cell('S')}" + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "tab_depth.tex").write_text("\n".join(L) + "\n")
    return len(rows)


# ───────────── S2：推理开销 ─────────────
def tab_cost():
    f = ROOT / "reports/r2/cost_bench.json"
    if not f.exists():
        (OUT / "tab_cost.tex").write_text("%% pending\n"); return False
    C = json.load(open(f)); R = C["results"]
    order = ["deit_small_patch16", "dinov2_vits14", "dinov2_vits14_b6", "resnet50_l3l4",
             "dino_resnet50", "resnet50", "wide_resnet50_2", "dinov2_vitb14"]
    L = [r"\begin{table}[t]",
         rf"\caption{{Inference cost at {C['resolution']}\,px, batch 1, fp32, on one",
         rf"{C['gpu']}, for a $k={C['k']}$ bank with {C['rotations']}-fold rotation",
         r"augmentation. \emph{match} is the 1-NN search against the bank. The",
         r"PatchCore-style mid-level CNN configurations cost roughly three times the",
         r"latency of the ViTs, and the cost is in the search rather than the backbone:",
         r"their $56\times56$ patch grid is four times denser.}",
         r"\label{tab:cost}\centering\footnotesize",
         r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
         r"Representation & $N\times D$ & encode & match & total & VRAM\\",
         r" & & (ms) & (ms) & (ms) & (MiB)\\", r"\midrule"]
    for bb in order:
        if bb not in R: continue
        v = R[bb]
        L.append(f"{NICE.get(bb, bb)} & ${v['N']}{{\\times}}{v['D']}$ & {v['encode_ms']:.1f} & "
                 f"{v['match_ms']:.1f} & {v['total_ms']:.1f} & {v['vram_mib']:.0f}" + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "tab_cost.tex").write_text("\n".join(L) + "\n")
    return len([b for b in order if b in R])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_all()
    print("configs:", ", ".join(f"{k[0]}/{k[1]}" for k in sorted(cfg)))
    print(f"  tab_ratio.tex        ({tab_ratio(cfg)} rows)")
    tab_interaction(cfg); print("  tab_interaction.tex")
    ok1 = tab_conditions(cfg, "dinov2_vits14", "wide_resnet50_2", fname="tab_conditions.tex")
    print(f"  tab_conditions.tex   ({'ok' if ok1 else 'pending'})")
    ok2 = tab_conditions(cfg, "resnet50", "dino_resnet50", fname="tab_conditions_cnn.tex")
    print(f"  tab_conditions_cnn.tex ({'ok' if ok2 else 'pending'})")
    print(f"  tab_2x2.tex          ({'ok' if tab_2x2(cfg) else 'pending'})")
    nd = tab_depth(cfg); print(f"  tab_depth.tex        ({nd if nd else 'pending'} rows)")
    nc = tab_cost(); print(f"  tab_cost.tex         ({nc if nc else 'pending'} rows)")
    print(f"  tab_sourcepool.tex   ({'ok' if tab_sourcepool() else 'pending'})")
    print(f"  tab_physical.tex     ({'ok' if tab_physical(cfg) else 'pending'})")
    print(f"  tab_physmatch.tex    ({'ok' if tab_physmatch(cfg) else 'pending'})")
    print(f"  tab_deploy.tex       ({'ok' if tab_deploy() else 'pending'})")
    n_ae = tab_aebad(); print(f"  tab_aebad.tex        ({n_ae if n_ae else 'pending'} rows)")
    print(f"  tab_aebad_physical.tex ({'ok' if tab_aebad_physical() else 'pending'})")
    print(f"  facts.tex            ({facts(cfg)} macros)")


if __name__ == "__main__":
    main()
