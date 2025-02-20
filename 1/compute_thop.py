from thop import profile
from thop import clever_format
import torch

from semseg.models import MutiModalTransformer  

device = torch.device("cuda:0")
model = MutiModalTransformer(
    in_channels=3,  # 输入通道
    embed_dims=32,  # 嵌入维度
    num_stages=4,    # 模型阶段数
    num_layers=[2, 2, 2, 2],  # 每个阶段的层数
    num_heads=[1, 2, 5, 8],  # 每个阶段的注意力头数
    patch_sizes=[7, 3, 3, 3],  # 每个阶段的 patch 尺寸
    strides=[4, 2, 2, 2],      # 步长
    sr_ratios=[8, 4, 2, 1],    # 注意力采样比例
    out_indices=(0, 1, 2, 3),  # 输出特征的阶段
    pretrained=False           # 是否使用预训练
).to(device)

# 创建一个虚拟输入张量
input_tensor = torch.randn(1, 4, 3, 512, 512).to(device)  # (batch_size, channels, height, width)

# 使用 thop 计算 FLOPs 和参数量
flops, params = profile(model, inputs=(input_tensor,))

# 格式化输出
flops, params = clever_format([flops, params], "%.3f")
print(f"FLOPs: {flops}, Params: {params}")