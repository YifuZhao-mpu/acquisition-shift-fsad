#!/usr/bin/env python3
"""额外 backbone：PatchCore 风格的 WideResNet-50 特征。

目的：本项目全部结论此前都基于 DINOv2（自监督 ViT）。
加入一个**监督预训练的 CNN**，其归纳偏置与训练范式都不同，
才能回答「这些结论是否是 DINOv2 特异的」。

PatchCore 的做法：取 layer2 + layer3 的特征，layer3 上采样到 layer2 分辨率后拼接，
再做 3x3 平均池化（局部感受野聚合）。本实现遵循该配置。
接口与 AnomalyDINO 的 wrapper 对齐（prepare_image / extract_features /
compute_background_mask），以便直接接入 fscache。
"""
import re
import numpy as np
import torch, torch.nn.functional as F
import torchvision
from torchvision import transforms


class WideResNetWrapper:
    def __init__(self, model_name="wide_resnet50_2", device="cuda", smaller_edge_size=448):
        self.device = device
        self.smaller_edge_size = smaller_edge_size
        net = getattr(torchvision.models, model_name)(weights="IMAGENET1K_V1")
        net.eval().to(device)
        self.net = net
        self._feats = {}
        net.layer2.register_forward_hook(self._hook("l2"))
        net.layer3.register_forward_hook(self._hook("l3"))
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _hook(self, name):
        def fn(_m, _i, o): self._feats[name] = o
        return fn

    def prepare_image(self, img):
        """img: HxWx3 RGB uint8 或 路径已在外部读好的数组。-> (tensor, grid_size)"""
        import cv2
        h, w = img.shape[:2]
        s = self.smaller_edge_size / min(h, w)
        nh, nw = int(round(h * s)), int(round(w * s))
        nh, nw = nh - nh % 32, nw - nw % 32          # 对齐到 stride 32
        im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        t = self.tf(im).unsqueeze(0).to(self.device)
        return t, (nh // 8, nw // 8)                  # layer2 stride = 8

    def extract_features(self, t):
        with torch.inference_mode():
            self.net(t)
            l2, l3 = self._feats["l2"], self._feats["l3"]
            l3 = F.interpolate(l3, size=l2.shape[-2:], mode="bilinear", align_corners=False)
            f = torch.cat([l2, l3], dim=1)            # (1, C, H, W)
            f = F.avg_pool2d(f, 3, 1, 1)              # PatchCore 的 3x3 局部聚合
            f = f.squeeze(0).permute(1, 2, 0).reshape(-1, f.shape[1])
        return f.float().cpu().numpy()

    def compute_background_mask(self, feats, grid_size, threshold=10, masking_type=False,
                                kernel_size=3, border=0.2):
        """CNN 特征不做 DINOv2 那种 PCA 前景估计；本项目 CNN 实验一律 masking=False。"""
        if masking_type:
            raise NotImplementedError("WideResNet 路径不支持 PCA 前景遮罩；请用 masking=False")
        return np.ones(feats.shape[0], dtype=bool)


def get_model_ext(model_name, device="cuda", smaller_edge_size=448):
    # --- 既有路径，行为逐字节不变（wide_resnet50_2 的既有结果必须可复现）---
    if model_name.startswith("wide_resnet"):
        return WideResNetWrapper(model_name, device, smaller_edge_size)

    # --- Round-2 新增：2x2 识别设计 + 层深对照 ---
    if model_name.startswith("resnet") or model_name.startswith("dino_resnet"):
        pre = "dino" if model_name.startswith("dino_") else "supervised"
        name = model_name[len("dino_"):] if pre == "dino" else model_name
        layers = "l3l4" if name.endswith("_l3l4") else "l2l3"
        arch = name[:-len("_l3l4")] if layers == "l3l4" else name
        return ResNetWrapper(arch, pre, layers, device, smaller_edge_size)

    if model_name in TimmViTWrapper._ALIAS:
        return TimmViTWrapper(model_name, device, smaller_edge_size)

    m = re.match(r"^(dinov2_\w+?)_b(\d+)$", model_name)
    if m:
        return DINOv2LayerWrapper(m.group(1), int(m.group(2)), device, smaller_edge_size)

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "AnomalyDINO"))
    from src.backbones import get_model
    return get_model(model_name, device, smaller_edge_size)


# ══════════════════════════════════════════════════════════════════════════
# Round-2 审查新增：把「预训练目标」「架构」「特征层深度」三个因素拆开
#
# 原实现只有 DINOv2 ViT（自监督）vs WideResNet-50-2（监督），两者同时差了
# 预训练目标、架构、层深、维度、patch 粒度 —— 无法把差异归因到任一因素。
# 下面补齐一个 2×2 识别设计，外加一条层深对照：
#
#                  self-supervised            supervised
#   ViT            dinov2_vits14              deit_small_patch16
#   CNN            dino_resnet50              resnet50
#
#   层深对照       resnet50_l3l4 / dino_resnet50_l3l4   （vs 默认 l2+l3）
#                  dinov2_vits14_b6 / _b9                （vs 默认末层）
#
# 关键：CNN 两臂用**完全相同的架构**（torchvision resnet50），只换权重；
# ViT 两臂都是 ViT-Small、384 维。这样每一次比较只动一个因素。
# ══════════════════════════════════════════════════════════════════════════

_DINO_RN50_URL = "https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth"


class ResNetWrapper:
    """torchvision ResNet 系，PatchCore 风格的多层拼接 + 3x3 局部聚合。

    pretrain: 'supervised' = IMAGENET1K_V2 权重； 'dino' = DINO 自监督权重（同一架构）
    layers:   'l2l3'（PatchCore 默认，中层）或 'l3l4'（更深，用于层深对照）
    """

    def __init__(self, arch="resnet50", pretrain="supervised", layers="l2l3",
                 device="cuda", smaller_edge_size=448):
        import torchvision
        self.device, self.smaller_edge_size = device, smaller_edge_size
        self.layers, self.pretrain, self.arch = layers, pretrain, arch

        if pretrain == "dino":
            assert arch == "resnet50", "DINO 自监督权重只提供 ResNet-50"
            net = torchvision.models.resnet50(weights=None)
            sd = torch.hub.load_state_dict_from_url(_DINO_RN50_URL, map_location="cpu")
            missing, unexpected = net.load_state_dict(sd, strict=False)
            # DINO checkpoint 不含 fc（自监督无分类头）——这是预期内的唯一缺项
            assert set(missing) <= {"fc.weight", "fc.bias"}, f"unexpected missing keys: {missing}"
            assert not unexpected, f"unexpected keys: {unexpected}"
        else:
            net = getattr(torchvision.models, arch)(weights="IMAGENET1K_V2"
                                                    if arch == "resnet50" else "IMAGENET1K_V1")
        net.eval().to(device)
        self.net = net
        self._feats = {}
        a, b = ("layer2", "layer3") if layers == "l2l3" else ("layer3", "layer4")
        getattr(net, a).register_forward_hook(self._hook("a"))
        getattr(net, b).register_forward_hook(self._hook("b"))
        self._stride = 8 if layers == "l2l3" else 16
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _hook(self, name):
        def fn(_m, _i, o): self._feats[name] = o
        return fn

    def prepare_image(self, img):
        import cv2
        h, w = img.shape[:2]
        s = self.smaller_edge_size / min(h, w)
        nh, nw = int(round(h * s)), int(round(w * s))
        nh, nw = nh - nh % 32, nw - nw % 32
        im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        t = self.tf(im).unsqueeze(0).to(self.device)
        return t, (nh // self._stride, nw // self._stride)

    def extract_features(self, t):
        with torch.inference_mode():
            self.net(t)
            fa, fb = self._feats["a"], self._feats["b"]
            fb = F.interpolate(fb, size=fa.shape[-2:], mode="bilinear", align_corners=False)
            f = torch.cat([fa, fb], dim=1)
            f = F.avg_pool2d(f, 3, 1, 1)
            f = f.squeeze(0).permute(1, 2, 0).reshape(-1, f.shape[1])
        return f.float().cpu().numpy()

    def compute_background_mask(self, feats, grid_size, threshold=10, masking_type=False,
                                kernel_size=3, border=0.2):
        if masking_type:
            raise NotImplementedError("ResNet 路径不支持 PCA 前景遮罩；请用 masking=False")
        return np.ones(feats.shape[0], dtype=bool)


class TimmViTWrapper:
    """timm ViT 系（监督预训练），接口对齐 DINOv2Wrapper。

    用于 2×2 的「监督 ViT」臂。DeiT-S/16 与 DINOv2-S/14 同为 ViT-Small、384 维，
    差别集中在预训练目标与 patch 尺寸（16 vs 14）。
    """

    _ALIAS = {
        "deit_small_patch16": "deit_small_patch16_224",
        "vit_small_patch16":  "vit_small_patch16_224.augreg_in1k",
    }

    def __init__(self, model_name, device="cuda", smaller_edge_size=448):
        import timm
        self.device, self.smaller_edge_size = device, smaller_edge_size
        self.model_name = model_name
        tname = self._ALIAS[model_name]
        self.net = timm.create_model(tname, pretrained=True, num_classes=0,
                                     img_size=smaller_edge_size).eval().to(device)
        self.patch = self.net.patch_embed.patch_size[0]
        self.n_prefix = getattr(self.net, "num_prefix_tokens", 1)
        cfg = timm.data.resolve_data_config({}, model=self.net)
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg["mean"], std=cfg["std"]),
        ])

    def prepare_image(self, img):
        import cv2
        h, w = img.shape[:2]
        s = self.smaller_edge_size / min(h, w)
        nh, nw = int(round(h * s)), int(round(w * s))
        nh, nw = nh - nh % self.patch, nw - nw % self.patch
        # timm 的 pos-embed 按创建时的 img_size 插值；此处统一到正方形，避免二次插值歧义
        nh = nw = self.smaller_edge_size
        im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        t = self.tf(im).unsqueeze(0).to(self.device)
        return t, (nh // self.patch, nw // self.patch)

    def extract_features(self, t):
        with torch.inference_mode():
            tok = self.net.forward_features(t)          # (1, n_prefix + N, D)
        return tok[0, self.n_prefix:, :].float().cpu().numpy()

    def compute_background_mask(self, feats, grid_size, threshold=10, masking_type=False,
                                kernel_size=3, border=0.2):
        if masking_type:
            raise NotImplementedError("timm ViT 路径未实现 PCA 前景遮罩；请用 masking=False")
        return np.ones(feats.shape[0], dtype=bool)


class DINOv2LayerWrapper:
    """DINOv2 的**中间层** patch token —— 层深对照。

    R2 席指出：DINOv2 取末层 patch token，而 PatchCore 风格 CNN 取 layer2+layer3
    的中层特征。PatchCore 刻意避开末层，因为末层过度语义化。因此原对照里
    「层深」与「预训练目标」完全混淆。本 wrapper 取指定 block 的输出。
    """

    def __init__(self, base_name, block, device="cuda", smaller_edge_size=448):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "AnomalyDINO"))
        from src.backbones import DINOv2Wrapper
        self._w = DINOv2Wrapper(base_name, device, smaller_edge_size)
        self.block = block
        self.device, self.smaller_edge_size = device, smaller_edge_size

    def prepare_image(self, img):
        return self._w.prepare_image(img)

    def extract_features(self, image_tensor):
        with torch.inference_mode():
            batch = image_tensor.unsqueeze(0).to(self.device)
            tok = self._w.model.get_intermediate_layers(batch, n=[self.block])[0]
        return tok.squeeze(0).float().cpu().numpy()

    def compute_background_mask(self, *a, **kw):
        return self._w.compute_background_mask(*a, **kw)
