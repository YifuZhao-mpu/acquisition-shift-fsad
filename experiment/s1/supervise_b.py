#!/usr/bin/env python3
"""方案 B 接力调度（Python 实现）。

为什么不用 shell + pgrep：`pgrep -f "run_sweep.sh single cuda:1"` 会匹配到
任何命令行中含该字符串的进程——包括正在查看它的 shell 本身——从而误杀无关进程。
本实现锁定启动时确定的确切 PID，不做任何模式匹配。

阶段1  等 k=4 两协议各 8/8 → 终止指定 PID（阻止其串行开跑 k=8）
阶段2  等 k=1,2 两协议各 8/8（GPU0/GPU2 收工）
阶段3  k=8 的 16 个 seed 拆到 4 块 GPU，seed 集合互不相交
"""
import os, re, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path(os.environ.get("IADSHIFT_ROOT",
                    Path(__file__).resolve().parents[2]))   # experiment/s1/x.py -> 项目根
AD   = ROOT / "experiment/AnomalyDINO"
S1   = ROOT / "experiment/s1"
LOG  = S1 / "logs"
PREP = "force_mask_rotation"
KILL_PIDS = [int(x) for x in sys.argv[1:]] or []


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def done_count(proto, k):
    d = AD / f"results_AeBAD_{proto}/dinov2_vits14_448/{k}-shot_preprocess={PREP}"
    n = 0
    if d.is_dir():
        for f in d.glob("measurements_seed=*.csv"):
            try:
                with open(f) as fh: n += (sum(1 for _ in fh) >= 1640)
            except OSError: pass
    return n


def wait_for(pairs, label):
    log(f"等待 {label}")
    while True:
        cnt = {(p, k): done_count(p, k) for p, k in pairs}
        if all(v >= 8 for v in cnt.values()):
            log(f"{label} 已完成: " + ", ".join(f"{p}k{k}={v}" for (p, k), v in cnt.items()))
            return
        time.sleep(60)


def children(pid):
    out = []
    for p in os.listdir("/proc"):
        if not p.isdigit(): continue
        try:
            for l in open(f"/proc/{p}/status"):
                if l.startswith("PPid:"):
                    if int(l.split()[1]) == pid: out.append(int(p))
                    break
        except OSError: pass
    return out


def kill_tree(pid):
    """先杀子（python），再杀父（run_sweep.sh）。只按 PID，不按名字。"""
    for c in children(pid):
        try: os.kill(c, signal.SIGTERM); log(f"  终止子进程 {c}")
        except ProcessLookupError: pass
    try: os.kill(pid, signal.SIGTERM); log(f"  终止 {pid}")
    except ProcessLookupError: log(f"  {pid} 已不存在")


def main():
    wait_for([("single", 4), ("mixed", 4)], "阶段1：k=4")
    log(f"终止 k=8 串行通道 PID={KILL_PIDS}（阻止其独占式跑 k=8）")
    for p in KILL_PIDS: kill_tree(p)
    time.sleep(8)

    wait_for([("single", 1), ("mixed", 1), ("single", 2), ("mixed", 2)], "阶段2：k=1,2")

    log("阶段3：k=8 四路并行，seed 互不相交")
    lanes = [("single", "cuda:0", "0 1 2 3"), ("single", "cuda:1", "4 5 6 7"),
             ("mixed",  "cuda:2", "0 1 2 3"), ("mixed",  "cuda:3", "4 5 6 7")]
    procs = []
    for proto, gpu, seeds in lanes:
        out = open(LOG / f"k8_{proto}_{gpu.replace(':','')}.out", "w")
        procs.append(subprocess.Popen(
            ["bash", str(S1 / "run_k8.sh"), proto, gpu] + seeds.split(),
            stdout=out, stderr=subprocess.STDOUT))
        log(f"  启动 {proto} {gpu} seeds={seeds}")
    for p in procs: p.wait()

    tot = sum(done_count(p, k) for p in ("single", "mixed") for k in (1, 2, 4, 8))
    (LOG / "MASK_ALL_DONE").write_text(f"{time.strftime('%H:%M:%S')} total={tot}/64\n")
    log(f"全部完成 total={tot}/64")


if __name__ == "__main__":
    main()
