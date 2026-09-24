#!/usr/bin/env python3
"""推理开销基准 —— Round-2 审查 S2。

§10 让实务者在不同 backbone 间选型，而产线有节拍约束（常见 50–200 ms/件）。
没有延迟与显存数字，该建议无法执行。

测量：448px、batch=1、单卡 V100、fp32，含 (a) 特征提取；(b) k=4 bank 的
1-NN 匹配（含 8 倍旋转增强的参考 patch 数）。必须在**空闲 GPU** 上跑。
"""
from __future__ import annotations
import os
import json, sys, time
from pathlib import Path
import numpy as np, torch, cv2

ROOT = Path(os.environ.get("IADSHIFT_ROOT",
                    Path(__file__).resolve().parents[2]))   # experiment/s1/x.py -> 项目根
sys.path.insert(0, str(Path(__file__).parent))
from fscache import Encoder

BACKBONES = ["dinov2_vits14", "dinov2_vitb14", "deit_small_patch16",
             "resnet50", "dino_resnet50", "wide_resnet50_2",
             "resnet50_l3l4", "dinov2_vits14_b6"]
K = 4; ROT = 8


def main():
    img = cv2.cvtColor(cv2.imread(str(ROOT / "data/mvtec_anomaly_detection/bottle/test/good/000.png")),
                       cv2.COLOR_BGR2RGB)
    out = {}
    print(f"{'backbone':22}{'N x D':>14}{'encode':>10}{'match':>9}{'total':>9}{'peak VRAM':>11}")
    for bb in BACKBONES:
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            enc = Encoder(bb, 448)
            for _ in range(3): enc.encode(None, masking=False, img=img)       # warm-up
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(20): f, _, _ = enc.encode(None, masking=False, img=img)
            torch.cuda.synchronize(); t_enc = (time.time() - t0) / 20

            N, D = f.shape
            bank = torch.randn(K * ROT * N, D, device="cuda")
            bank /= bank.norm(dim=1, keepdim=True)
            ft = torch.from_numpy(f).cuda()
            torch.cuda.synchronize(); t0 = time.time()
            with torch.inference_mode():
                for _ in range(20):
                    best = None
                    for i in range(0, bank.shape[0], 4096):
                        mx = (ft @ bank[i:i + 4096].T).max(dim=1).values
                        best = mx if best is None else torch.maximum(best, mx)
            torch.cuda.synchronize(); t_m = (time.time() - t0) / 20
            vram = torch.cuda.max_memory_allocated() / 2**20
            out[bb] = dict(N=int(N), D=int(D), encode_ms=t_enc * 1e3,
                           match_ms=t_m * 1e3, total_ms=(t_enc + t_m) * 1e3, vram_mib=vram)
            print(f"{bb:22}{f'{N}x{D}':>14}{t_enc*1e3:9.1f}m{t_m*1e3:8.1f}m{(t_enc+t_m)*1e3:8.1f}m{vram:10.0f}M")
            del enc, bank, ft; torch.cuda.empty_cache()
        except Exception as e:
            print(f"{bb:22}  FAILED: {type(e).__name__}: {e}")
    p = ROOT / "reports/r2/cost_bench.json"
    json.dump(dict(resolution=448, batch=1, k=K, rotations=ROT, gpu=torch.cuda.get_device_name(0),
                   dtype="fp32", results=out), open(p, "w"), indent=1)
    print(f"写入 {p}")


if __name__ == "__main__":
    main()
