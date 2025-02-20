import torch
from torch import nn
from torch.nn import functional as F
import yaml
from fvcore.nn import FlopCountAnalysis
from semseg.models.heads import SegFormerHead
from semseg.models.base import BaseModel    


class MutiModalTransformer(BaseModel):
    def __init__(self, backbone: str = 'MutiModalTransformer-B0', num_classes: int = 19, **kwargs):
        super().__init__(backbone, num_classes, **kwargs)
        self.decode_head = SegFormerHead(self.backbone.channels, 256 if 'B0' in backbone or 'B1' in backbone else 512, num_classes)
        self.apply(self._init_weights)

    def forward(self, x):
        y = self.backbone(x)
        y = self.decode_head(y)
        y = F.interpolate(y, size=x[0].shape[2:], mode='bilinear', align_corners=False)
        y = y.mean(dim=0, keepdim=True)
        return y

    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            if self.backbone.num_modals > 0:
                load_dualpath_model(self.backbone, pretrained)
            else:
                checkpoint = torch.load(pretrained, map_location='cpu')
                if 'state_dict' in checkpoint.keys():
                    checkpoint = checkpoint['state_dict']
                if 'model' in checkpoint.keys():
                    checkpoint = checkpoint['model']
                msg = self.backbone.load_state_dict(checkpoint, strict=False)
                print(msg)


def load_dualpath_model(model: nn.Module, segformer_ckpt: str) -> None:
    """
    适配 SegFormer 预训练权重到 MutiModalTransformer：
    
    - `patch_embed`、`block`、`norm` 直接加载预训练权重
    - 忽略 `FeatureFusion` (layers[i][3])
    - 允许 `strict=False` 以防止 `FeatureFusion` 层报错

    Args:
        model (nn.Module): `MutiModalTransformer` 实例
        segformer_ckpt (str): 预训练 `SegFormer` `.pth` 文件路径
    """

    # 1. 加载预训练模型
    checkpoint = torch.load(segformer_ckpt, map_location='cpu')
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    if 'model' in checkpoint:
        checkpoint = checkpoint['model']

    # 2. 过滤 `patch_embed`、`block`、`norm`，跳过 `FeatureFusion`
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith('layers'):
            if 'decode_head' in k:
                continue  # 跳过 SegFormer 的 `decode_head`

            new_state_dict[k] = v  # 其余部分正常加载

    # 3. 加载权重，允许 `FeatureFusion` 层权重缺失
    msg = model.load_state_dict(new_state_dict, strict=False)
    print("Loaded SegFormer pre-trained weights:", msg)




if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = '/root/multimodal/configs/mcubes.yaml'
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    model_cfg = cfg['MODEL']
    model_structure = model_cfg['MODEL_STRUCTURE']

    model = MutiModalTransformer(backbone=model_cfg['BACKBONE'], num_classes=25).to(device)
    pretrained_weights = '/root/multimodal/pretrained/mit_b0_20220624-7e0fe6dd.pth'
    model.init_pretrained(pretrained=pretrained_weights)

    modal_1 = torch.zeros(1, 3, 1024, 1024, device=device)      # 模态 1
    modal_2 = torch.ones(1, 3, 1024, 1024, device=device)       # 模态 2
    modal_3 = torch.ones(1, 3, 1024, 1024, device=device)    # 模态 3
    modal_4 = torch.ones(1, 3, 1024, 1024, device=device)    # 模态 4

    x = torch.stack([modal_1, modal_2, modal_3, modal_4], dim=1).to(device)
    y = model(x)
    print(y.shape) 

