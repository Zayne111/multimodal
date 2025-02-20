import torch
from torch import nn
from torch.nn import functional as F
import math
from semseg.models.layers import trunc_normal_
from semseg.models.backbones import * 

def load_dualpath_model(model, model_file):
    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        if 'model' in raw_state_dict.keys():
            raw_state_dict = raw_state_dict['model']
    else:
        raw_state_dict = model_file
        
    state_dict = {}
    for k, v in raw_state_dict.items():
        if 'patch_embed' in k:
            state_dict[k] = v
        elif 'block' in k:
            state_dict[k] = v
        elif 'norm' in k:
            state_dict[k] = v

    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded pretrained weights with message: {msg}")
    del state_dict

class BaseModel(nn.Module):
    def __init__(self, backbone: str = 'mit-B0', num_classes: int = 19, **kwargs):
        super().__init__()
        backbone_name, variant = backbone.split('-')
        self.backbone = eval(backbone_name)(**kwargs)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            if self.backbone.num_modals>1:
                load_dualpath_model(self.backbone, pretrained)
            else:
                checkpoint = torch.load(pretrained, map_location='cpu')
                if 'state_dict' in checkpoint.keys():
                    checkpoint = checkpoint['state_dict']
                if 'model' in checkpoint.keys(): 
                    checkpoint = checkpoint['model']
                msg = self.backbone.load_state_dict(checkpoint, strict=False)
                print(msg)
