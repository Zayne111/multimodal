import torch
from cmm import FeatureFusion
from torch.nn import BatchNorm2d

modal_features = torch.randn(2, 4, 3, 512, 512)

# 定义 FeatureFusion 的参数
dim = 64
num_modals = 4
reduction = 2
num_heads = 4
feature_fusion = FeatureFusion(dim=dim, num_modals=num_modals, reduction=reduction, num_heads=num_heads, norm_layer=BatchNorm2d)

output = feature_fusion(modal_features)

print(f"Output shape: {output.shape}")