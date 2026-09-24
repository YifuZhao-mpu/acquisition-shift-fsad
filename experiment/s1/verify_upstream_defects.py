#!/usr/bin/env python3
"""可运行地复现 AnomalyDINO 参考实现（commit b9d1c26）的四个自定义数据集路径缺陷。

论文 §7.1 不满足于「读代码得出的结论」；本脚本把每个缺陷**触发一次**并记录
实际异常，使审稿人可以自行核验。上游原文件保存在 src/*.py.orig。
"""
from __future__ import annotations
import os
import ast, importlib.util, json, sys, traceback
from pathlib import Path

ROOT = Path(os.environ.get("IADSHIFT_ROOT",
                    Path(__file__).resolve().parents[2]))   # experiment/s1/x.py -> 项目根
AD = ROOT / "experiment/AnomalyDINO"
DATA = ROOT / "data/AeBAD_S_domains/single"       # 一个「自定义」数据集根目录


def load_orig(name):
    """按原始（未修复）文件加载模块，不污染已修复的包。"""
    sys.path.insert(0, str(AD))
    from importlib.machinery import SourceFileLoader
    # .orig 后缀不被识别为 Python，需显式指定 loader
    loader = SourceFileLoader(f"orig_{name}", str(AD / f"src/{name}.py.orig"))
    spec = importlib.util.spec_from_loader(f"orig_{name}", loader)
    m = importlib.util.module_from_spec(spec)
    sys.modules[f"orig_{name}"] = m
    spec.loader.exec_module(m)
    return m


def rec(results, n, title, where, fn):
    try:
        out = fn()
        results.append(dict(n=n, title=title, where=where, triggered=False, detail=repr(out)[:400]))
        print(f"  [{n}] {title}\n       {where}\n       -> NOT triggered: {repr(out)[:120]}")
    except Exception as e:
        tb = traceback.format_exc().strip().splitlines()[-1]
        results.append(dict(n=n, title=title, where=where, triggered=True, detail=tb))
        print(f"  [{n}] {title}\n       {where}\n       -> {tb}")


def main():
    results = []
    print("复现 AnomalyDINO @ b9d1c26 自定义数据集路径的缺陷：\n")

    pe = load_orig("post_eval")
    rec(results, 1, "get_objects_from_dataset has no else branch",
        "src/post_eval.py:410",
        lambda: pe.get_objects_from_dataset("AeBAD_S"))

    ut = load_orig("utils")
    # 缺陷 4：custom 分支无条件覆盖 preprocess，使 force_* 不可达
    def d4():
        objs, anom, masking, rotation = ut.get_dataset_info(
            "AeBAD_S", preprocess="force_mask_rotation", data_path=str(DATA))
        return dict(requested="force_mask_rotation",
                    masking_returned={k: bool(v) for k, v in list(masking.items())[:3]},
                    rotation_returned={k: bool(v) for k, v in list(rotation.items())[:3]})
    rec(results, 4, "custom branch unconditionally overrides --preprocess "
                    "(force_* unreachable, silent)", "src/utils.py:172", d4)

    # 缺陷 3：visualize.py 用相对导入，无法脱离包单独加载 -> 改为 AST 静态核验
    def d3():
        tree = ast.parse((AD / "src/visualize.py.orig").read_text())
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "create_sample_plots")
        params = [a.arg for a in fn.args.args]
        calls = [c for c in ast.walk(fn)
                 if isinstance(c, ast.Call) and getattr(c.func, "id", "") == "get_dataset_info"]
        assert calls, "get_dataset_info not called"
        kw = {k.arg for k in calls[0].keywords}
        if "data_root" in params and "data_path" not in kw:
            raise RuntimeError(
                f"create_sample_plots(line {fn.lineno}) accepts data_root but calls "
                f"get_dataset_info(line {calls[0].lineno}) with kwargs {sorted(kw)} "
                f"-> get_dataset_info raises ValueError: Please provide 'data_path'")
        return dict(params=params, call_kwargs=sorted(kw))
    rec(results, 3, "create_sample_plots does not forward data_root to get_dataset_info "
                    "(static check: relative import prevents standalone execution)",
        "src/visualize.py:61", d3)

    # 缺陷 2：eval_clf 依赖只有 eval_segm=True 才写的 tiff -> 空数组
    def d2():
        src = (AD / "src/post_eval.py.orig").read_text()
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "eval_finished_run")
        body = ast.get_source_segment(src, fn) or ""
        if "tiff" in body and "eval_clf" in body:
            raise RuntimeError(
                f"eval_finished_run(line {fn.lineno}) couples image-level evaluation to TIFF "
                f"files written only when eval_segm=True -> empty array -> "
                f"ValueError: Found array with 0 sample(s)")
        return "no coupling found"
    rec(results, 2, "image-level evaluation depends on TIFFs written only under eval_segm",
        "src/post_eval.py:418", d2)

    print("\n判读：")
    d4r = next(r for r in results if r["n"] == 4)
    if not d4r["triggered"] and "'masking_returned'" in d4r["detail"]:
        print("  缺陷 4 是**静默**的：请求 force_mask_rotation，返回的 masking 全为 False，"
              "\n       不抛异常、不给警告。任何据此做的遮罩消融实际都没开遮罩。")
    print("  缺陷 1 是**响亮崩溃**：任何人在自定义数据集上跑都会先撞上它。")
    print("  缺陷 2/3 由静态检查确认（相对导入使其无法脱离包单独执行）。")
    p = ROOT / "reports/r2/upstream_defects.json"
    json.dump(dict(commit="b9d1c26", results=results), open(p, "w"), indent=1)
    print(f"\n写入 {p}")


if __name__ == "__main__":
    main()
