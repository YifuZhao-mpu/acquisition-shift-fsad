#!/usr/bin/env python3
"""受控光照扰动 —— 为 MVTec AD / VisA 提供 AeBAD 所缺的**因果轴**。

AeBAD 的 illumination 域是真实但混杂的：换灯的同时工件位置、背景也可能变。
本模块只改变成像的光照项，其余像素内容完全不动，因此
「光照 → 性能退化」这条因果链可被单独隔离，且严重度可连续参数化。

物理动机（非任意的图像增强）：
  exposure  乘性增益            —— 快门/光圈变化
  gamma     非线性色调响应      —— 相机 tone curve / 显示 gamma
  wb        通道增益            —— 色温漂移（灯具老化、环境光串入）
  gradient  空间线性照度梯度    —— 光源偏置 / 距离平方反比
  specular  高斯高光斑          —— 镜面反射随几何/光源角度移动
  shadow    高斯压暗斑          —— 遮挡投影

关键约定：扰动**只施加于测试图**，参考图保持标称条件 —— 对应真实部署：
产线用标称条件下拍的 k 张正常件建库，之后照明发生漂移。
"""
from __future__ import annotations
import numpy as np

MODES = ["exposure", "gamma", "wb", "gradient", "specular", "shadow"]


def _f(img):  return img.astype(np.float32) / 255.0
def _u(x):    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def apply(img: np.ndarray, mode: str, severity: float, rng: np.random.Generator | None = None):
    """img: HxWx3 RGB uint8; severity in [0,1]（0 = 恒等变换）。"""
    assert mode in MODES, mode
    rng = rng or np.random.default_rng(0)
    x = _f(img); H, W = x.shape[:2]

    if mode == "exposure":                      # ±(0 .. 1.2) stop
        stops = 1.2 * severity * (1 if rng.random() < .5 else -1)
        return _u(x * (2.0 ** stops))

    if mode == "gamma":                         # gamma in [1/2.2, 2.2]
        g = 2.2 ** (severity * (1 if rng.random() < .5 else -1))
        return _u(np.power(x, g))

    if mode == "wb":                            # 色温：R/B 反向增益
        d = 0.45 * severity
        gain = np.array([1 + d, 1.0, 1 - d], np.float32) if rng.random() < .5 \
               else np.array([1 - d, 1.0, 1 + d], np.float32)
        return _u(x * gain)

    if mode == "gradient":                      # 线性照度梯度，沿随机方向
        th = rng.uniform(0, 2 * np.pi)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        u = (np.cos(th) * (xx / W - .5) + np.sin(th) * (yy / H - .5)) * 2.0    # [-1,1]
        return _u(x * (1.0 + 0.85 * severity * u)[..., None])

    if mode == "specular":                      # 高斯高光斑（加性，模拟镜面反射）
        cy, cx = rng.uniform(.25, .75, 2) * [H, W]
        sig = 0.16 * min(H, W)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        g = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sig ** 2)))
        return _u(x + (0.85 * severity) * g[..., None])

    if mode == "shadow":                        # 高斯压暗斑（乘性，模拟投影）
        cy, cx = rng.uniform(.25, .75, 2) * [H, W]
        sig = 0.22 * min(H, W)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        g = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sig ** 2)))
        return _u(x * (1.0 - 0.75 * severity * g)[..., None])


def severity_grid(n=5):
    """0 必须在内：severity=0 是恒等变换，构成同一流程内的零扰动对照。"""
    return list(np.linspace(0.0, 1.0, n))
