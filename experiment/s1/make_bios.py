#!/usr/bin/env python3
"""由 author_bios.md 生成 IEEE 作者传记节（paper/sections/11_biographies.tex）。

源文本逐字取自 CICDWOA 论文的传记页，本脚本只做**可审计的规范化**：
姓名与正文作者列表对齐、弯引号转 LaTeX、机构名统一、明显拼写错订正。
每一处改动都打印出来，不做静默修改。
"""
from __future__ import annotations
import json, re, sys
import os
from pathlib import Path

ROOT = Path(os.environ.get("IADSHIFT_ROOT", Path(__file__).resolve().parents[2]))
OUT = ROOT / "paper/sections/11_biographies.tex"

# 与主文 \author 块完全一致的顺序与写法
ORDER = [
    ("Yifu Zhao",   "Yifu_Zhao",   None),
    ("Xiaofan Zou", "Xiaofan_Zou", None),
    ("Yanxiao Li",  "Yanxiao_Li",  None),
    ("Junhao Wei",  "Junhao_Wei",  None),
    ("Sio-Kei Im",  "Sio-Kei_Im",  None),
    ("Yapeng Wang", "Yapeng_Wang", "Member, IEEE"),
    ("Xu Yang",     "Xu_Yang",     None),
]

# 源文写法 -> 作者列表写法（仅书写变体，非改名）
NAME_FIX = {
    "Zou Xiaofan": "Xiaofan Zou",
    "Hao Chen Li": "Haochen Li",
    "Li Yanshao":  "Yanxiao Li",       # 源文拼写与作者列表不一致，见 README 说明
}
# Sio-Kei Im 的源文是行政履历（率团访问、爱国教育、互联网论坛），
# 与 IEEE 传记的体例（学位、职位、研究方向）不符，且是八段里最长的一段。
# 这里截到与学术身份相关的事实主干；被删部分列在下方，便于作者复核。
TRIM = {
    "Sio-Kei Im": (
        "is the President of Macao Polytechnic University, "
        "dedicated to promoting the development of higher education and the "
        "integration of industry, academia, and research. In 2022, he successfully "
        "led the institution to be renamed Macao Polytechnic University and "
        "established the ``Mr. He Chuan Scholarship'' to foster technological talent. "
        "He engages with the National Natural Science Foundation of China on the "
        "cultivation of scientific and technological talent, and promotes regional "
        "cooperation and the construction of a smart city in Macao."
    ),
}

TEXT_FIX = [
    ("Macau Polytechnic University", "Macao Polytechnic University"),
    ("nature language processing", "natural language processing"),
    (" (Member, IEEE, https://fca.mpu.edu.mo/profile/yapengwang)", ""),
]


def to_latex(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "`")
    s = re.sub(r"'([^']{2,40})'", r"``\1''", s)      # 成对单引号 -> LaTeX 双引号
    s = s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
    return s


def main():
    src = ROOT / "paper/author_bios.md"
    if not src.exists():
        print(f"缺少 {src}"); sys.exit(1)
    blocks = re.split(r"\n## \d+ · ", src.read_text())[1:]
    bios = {}
    for b in blocks:
        name, body = b.split("\n", 1)
        body = re.sub(r"^照片：.*$", "", body, flags=re.M).split("---")[0]
        bios[name.strip()] = " ".join(body.split())

    changes, L = [], [
        "%% 由 experiment/s1/make_bios.py 生成，勿手改。",
        "%% 传记原文取自作者提供的 author_bios.md；规范化项见该脚本顶部说明。",
        "",
    ]
    for disp, photo, membership in ORDER:
        raw = bios.get(disp)
        if raw is None:
            print(f"  警告：{disp} 在 author_bios.md 中无条目"); continue
        t = raw
        for a, b in NAME_FIX.items():
            if t.startswith(a):
                t = b + t[len(a):]; changes.append(f"{disp}: 开头姓名 '{a}' -> '{b}'")
        for a, b in TEXT_FIX:
            if a in t:
                t = t.replace(a, b); changes.append(f"{disp}: '{a.strip()}' -> '{b.strip() or '(删除)'}'")
        # IEEEtran 会把 {姓名} 参数以大写排成起头，正文再写一遍姓名就成了
        # "XU YANG Xu Yang received..."。按 IEEE 体例正文直接从动词起。
        for pre in (disp, "Professor " + disp):
            if t.startswith(pre + " "):
                t = t[len(pre) + 1:]
                t = t[0].lower() + t[1:] if t[:1].isupper() and t.split()[0] not in ("He", "She", "In", "Since", "During") else t
                changes.append(f"{disp}: 正文去掉重复的姓名起头")
                break
        t = to_latex(t)
        if disp in TRIM:
            changes.append(f"{disp}: 传记由 {len(t.split())} 词截到 "
                           f"{len(TRIM[disp].split())} 词（删去率团访问、爱国教育、"
                           f"互联网论坛等与学术身份无关的段落）")
            t = TRIM[disp]
        head = (f"\\begin{{IEEEbiography}}[{{\\includegraphics[width=1in,height=1.25in,"
                f"clip,keepaspectratio]{{photos/{photo}.jpg}}}}]{{{disp}}}")
        if membership:
            head = (f"\\begin{{IEEEbiography}}[{{\\includegraphics[width=1in,height=1.25in,"
                    f"clip,keepaspectratio]{{photos/{photo}.jpg}}}}]"
                    f"{{{disp}}} \\IEEEmembership{{{membership}}}")
        L += [head, t, "\\end{IEEEbiography}", ""]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L))
    print(f"写入 {OUT}（{len(ORDER)} 段，{sum(len(bios.get(d,'').split()) for d,_,_ in ORDER)} 词）")
    print("\n规范化改动：")
    for c in changes: print("  -", c)


if __name__ == "__main__":
    main()
