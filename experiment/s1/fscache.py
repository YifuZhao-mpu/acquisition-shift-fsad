#!/usr/bin/env python3
"""特征缓存 + bank 组装引擎 —— 支持多检测器 / 多 backbone / 多数据集的受控支撑实验。

核心效率洞察（源自 Round-1 审稿）：
  k_neighbors=1 时，一个 bank 的逐 patch 最近邻距离 =
      min over 其成员参考图 ( 该参考图单独给出的逐 patch 最小距离 )
  因此把「每张参考图的逐 test-patch 最小距离图」算一次并缓存，
  任意 bank 只需在其成员子集上取逐元素最小值 —— 组装近乎免费。

  一个 replicate 抽 18 张参考（6B+6I+6V），缓存 1639×18×1024 float ≈ 120 MB。
  由此 4 种 bank（6B / 6I / 6V / 2+2+2）共享同一批缓存。

严格复用 AnomalyDINO 自身的 prepare_image / extract_features /
compute_background_mask / augment_image / mean_top1p，保证与主扫描可比。
"""
from __future__ import annotations
import os, sys, hashlib
from pathlib import Path
import numpy as np
import torch

AD = Path(os.environ.get("IADSHIFT_ROOT",
                        Path(__file__).resolve().parents[2])) / "experiment/AnomalyDINO"
sys.path.insert(0, str(AD))
import cv2
from src.utils import augment_image
from src.post_eval import mean_top1p
sys.path.insert(0, str(Path(__file__).parent))
from backbones_ext import get_model_ext as get_model

# 批处理只在已验证「逐位相同」的形状上启用（见 per_ref_mindist 文档字符串）。
# ResNet l2+l3 的 3136 patch/图 超过该阈值，故自动退回逐图路径。
BATCH_MAX_PATCHES = 2048


# ───────────────────────── 特征提取 ─────────────────────────
class Encoder:
    def __init__(self, model_name="dinov2_vits14", resolution=448, device="cuda"):
        self.model = get_model(model_name, device, smaller_edge_size=resolution)
        self.device = device
        self.model_name, self.resolution = model_name, resolution

    def encode(self, path, masking: bool, img=None):
        """-> (features[N_keep, D] float32 已L2归一化, keep_mask[N_all] bool, grid)
        img 非 None 时直接用该 RGB uint8 数组（用于已施加扰动的图像），忽略 path。"""
        if img is None:
            img = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        t, grid = self.model.prepare_image(img)
        f = self.model.extract_features(t)
        m = self.model.compute_background_mask(f, grid, threshold=10, masking_type=masking)
        fk = f[m].astype(np.float32)
        fk /= (np.linalg.norm(fk, axis=1, keepdims=True) + 1e-12)   # 对应 faiss.normalize_L2
        return fk, np.asarray(m).reshape(-1), grid

    def encode_reference(self, path, masking_ref: bool, rotation: bool):
        """参考图：按需做旋转增强，拼接全部增强视图的 patch。"""
        img = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        views = augment_image(img) if rotation else [img]
        chunks = []
        for v in views:
            t, grid = self.model.prepare_image(v)
            f = self.model.extract_features(t)
            m = self.model.compute_background_mask(f, grid, threshold=10, masking_type=masking_ref)
            fk = f[m].astype(np.float32)
            fk /= (np.linalg.norm(fk, axis=1, keepdims=True) + 1e-12)
            chunks.append(fk)
        return np.concatenate(chunks, axis=0)


# ───────────────────────── 测试集缓存 ─────────────────────────
class TestCache:
    """一次性编码整个测试集，常驻 GPU。"""
    def __init__(self, enc: Encoder, items, masking: bool, device="cuda", transform=None):
        """items: [(key, path, label)]  label: 0 正常 / 1 异常
        transform: 可选 callable(rgb_uint8, key) -> rgb_uint8，在编码前施加（用于受控扰动）。"""
        self.keys, self.labels, self.feats, self.masks, self.npatch = [], [], [], [], []
        for key, path, lab in items:
            im = None
            if transform is not None:
                im = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                im = transform(im, key)
            fk, m, grid = enc.encode(path, masking, img=im)
            self.keys.append(key); self.labels.append(lab)
            self.feats.append(torch.from_numpy(fk).to(device))
            self.masks.append(m); self.npatch.append(len(m))
        self.labels = np.asarray(self.labels)
        self.device = device

    def __len__(self): return len(self.keys)


def per_ref_mindist(cache: TestCache, ref_feats: np.ndarray, chunk=4096,
                    patch_budget=32768) -> list[np.ndarray]:
    """单张参考图 → 每张测试图的逐 patch 最小余弦距离（仅前景 patch）。

    余弦距离 = 1 - <a,b>（两侧均已 L2 归一化），对应 AnomalyDINO 的 L2_normalized/2。

    [Round-2 优化] 把若干张测试图拼成一个批次再做 matmul。矩阵乘法逐行独立、
    max 与 maximum 均为精确运算，故该改动在**数学上**恒等。

    但**浮点上不一定**：cuBLAS 会按矩阵形状挑选不同的 GEMM 算法，M 维变化可能
    改变 K 维归约的求和顺序。实测在 (3136,1536)x(1536,25088) 这一 ResNet
    l2+l3 形状上，批处理与逐图结果相差 3.7e-4（随机高斯输入；L2 归一化特征上
    约 1e-5）；而在 DINOv2-S(1024x384)、DeiT-S(784x384)、ResNet l3+l4(784x3072)
    这些形状上**逐位相同**（max|Δ|=0）。

    因此只在每图 patch 数 <= BATCH_MAX_PATCHES 时启用批处理 —— 该阈值把所有
    已验证逐位相同的形状纳入、把 ResNet l2+l3 排除在外，保证全部结果与
    逐图参考路径在浮点层面完全一致。
    """
    R = torch.from_numpy(ref_feats).to(cache.device)          # (Nr, D)
    out = [None] * len(cache.feats)
    npatch0 = cache.feats[0].shape[0] if cache.feats else 0
    if npatch0 > BATCH_MAX_PATCHES:        # 大形状：走逐图路径，保证逐位一致
        patch_budget = 0
    with torch.inference_mode():
        i0 = 0
        while i0 < len(cache.feats):
            i1, tot = i0, 0
            while i1 < len(cache.feats) and (tot == 0 or tot + cache.feats[i1].shape[0] <= patch_budget):
                tot += cache.feats[i1].shape[0]; i1 += 1
            F = torch.cat(cache.feats[i0:i1], dim=0) if i1 - i0 > 1 else cache.feats[i0]
            best = None
            for i in range(0, R.shape[0], chunk):
                mx = (F @ R[i:i + chunk].T).max(dim=1).values
                best = mx if best is None else torch.maximum(best, mx)
            d = (1.0 - best).float().cpu().numpy()
            off = 0
            for j in range(i0, i1):
                n = cache.feats[j].shape[0]
                out[j] = d[off:off + n]; off += n
            i0 = i1
    return out


def bank_scores(cache: TestCache, ref_maps: list[list[np.ndarray]], agg: str = "meantop1p") -> np.ndarray:
    """ref_maps: 该 bank 每个成员参考图的 per_ref_mindist 结果。
    bank 距离 = 成员逐元素最小值；再按 AnomalyDINO 方式散回全 patch 网格并取 mean_top1p。"""
    n = len(cache)
    scores = np.empty(n, dtype=np.float64)
    for j in range(n):
        d = ref_maps[0][j]
        for r in ref_maps[1:]:
            d = np.minimum(d, r[j])
        full = np.zeros(cache.npatch[j], dtype=np.float64)     # 被遮罩 patch 记 0（与上游一致）
        full[cache.masks[j]] = d
        # meantop1p = AnomalyDINO（top-1% patch 距离均值，即 0.99 分位的经验尾部风险值）
        # max       = PatchCore 的图像级打分（最大 patch 距离）
        scores[j] = float(full.max()) if agg == "max" else mean_top1p(full)
    return scores
