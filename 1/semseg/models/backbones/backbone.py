import math
import warnings

import torch
from torch import nn, Tensor
import torch.utils.checkpoint as cp
from mmcv.cnn import Conv2d, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmengine.model import BaseModule, ModuleList, Sequential
from mmengine.model.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)

from semseg.models.modules.utils import nlc_to_nchw, nchw_to_nlc, PatchEmbed
from semseg.models.modules.cmm import FeatureFusion


class MixFFN(BaseModule):
    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 dropout_layer=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        in_channels = embed_dims
        fc1 = Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        # 3x3 depth wise conv to provide positional encode information
        pe_conv = Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
            bias=True,
            groups=feedforward_channels)
        fc2 = Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = Sequential(*layers)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        out = nlc_to_nchw(x, hw_shape)
        out = self.layers(out)
        out = nchw_to_nlc(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class Attention(MultiheadAttention):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=None,
                 init_cfg=None,
                 batch_first=True,
                 qkv_bias=False,
                 norm_cfg=dict(type='LN'),
                 sr_ratio=1):
        super().__init__(
            embed_dims,
            num_heads,
            attn_drop,
            proj_drop,
            dropout_layer=dropout_layer,
            init_cfg=init_cfg,
            batch_first=batch_first,
            bias=qkv_bias)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = Conv2d(
                in_channels=embed_dims,
                out_channels=embed_dims,
                kernel_size=sr_ratio,
                stride=sr_ratio)
            # The ret[0] of build_norm_layer is norm name.
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

        from mmseg import digit_version, mmcv_version
        if mmcv_version < digit_version('1.3.17'):
            warnings.warn('The legacy version of forward function in'
                          'EfficientMultiheadAttention is deprecated in'
                          'mmcv>=1.3.17 and will no longer support in the'
                          'future. Please upgrade your mmcv.')
            self.forward = self.legacy_forward

    def forward(self, x, hw_shape, identity=None):

        x_q = x
        if self.sr_ratio > 1:
            x_kv = nlc_to_nchw(x, hw_shape)
            x_kv = self.sr(x_kv)
            x_kv = nchw_to_nlc(x_kv)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x

        if identity is None:
            identity = x_q

        if self.batch_first:
            x_q = x_q.transpose(0, 1)
            x_kv = x_kv.transpose(0, 1)

        out = self.attn(query=x_q, key=x_kv, value=x_kv)[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))

    def legacy_forward(self, x, hw_shape, identity=None):
        x_q = x
        if self.sr_ratio > 1:
            x_kv = nlc_to_nchw(x, hw_shape)
            x_kv = self.sr(x_kv)
            x_kv = nchw_to_nlc(x_kv)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x

        if identity is None:
            identity = x_q

        out = self.attn(query=x_q, key=x_kv, value=x_kv, need_weights=False)[0]

        return identity + self.dropout_layer(self.proj_drop(out))


class EncoderLayer(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 batch_first=True,
                 sr_ratio=1,
                 with_cp=False):
        super().__init__()

        # The ret[0] of build_norm_layer is norm name.
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.attn = Attention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            batch_first=batch_first,
            qkv_bias=qkv_bias,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratio)

        # The ret[0] of build_norm_layer is norm name.
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

        self.with_cp = with_cp

    def forward(self, x, hw_shape):

        def _inner_forward(x):
            H, W = hw_shape
            x = self.attn(self.norm1(x), hw_shape, identity=x)
            x = self.ffn(self.norm2(x), hw_shape, identity=x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


    

class MutiModalTransformer(BaseModule):
    def __init__(self, **kwargs):
        super().__init__()

        in_channels = kwargs.get('IN_CHANNELS', 3)
        embed_dims = kwargs.get('EMBED_DIMS', 32)
        num_stages = kwargs.get('NUM_STAGES', 4)
        num_layers = kwargs.get('NUM_LAYERS', [2, 2, 2, 2])
        num_heads = kwargs.get('NUM_LAYERS', [1, 2, 5, 8])
        patch_sizes = kwargs.get('PATCH_SIZES', [7, 3, 3, 3])
        strides = kwargs.get('STRIDES', [4, 2, 2, 2])
        sr_ratios = kwargs.get('SR_RATIOS', [8, 4, 2, 1])
        out_indices = kwargs.get('OUT_INDICES', (0, 1, 2, 3))
        mlp_ratio = kwargs.get('MLP_RATIO', 4)
        qkv_bias = True
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.
        act_cfg = dict(type='GELU')
        norm_cfg = dict(type='LN', eps=1e-6)
        with_cp = False
        num_modals = kwargs.get('NUM_MODALS', 4)
        fusion_reduction = kwargs.get('FUSION_REDUCTION', 2)

        self.num_modals = num_modals
        self.num_stages = num_stages
        self.out_indices = out_indices
        channels = [
            int(embed_dims * num_heads[0]),
            int(embed_dims * num_heads[1]),
            int(embed_dims * num_heads[2]),
            int(embed_dims * num_heads[3])
        ]

        self.channels = channels

        # Transformer Encoder
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]
        cur = 0
        self.layers = ModuleList()
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = embed_dims * num_heads[i]
            patch_embed = PatchEmbed(
                in_channels=in_channels if i==0 else embed_dims * num_heads[i-1],
                #in_channels=in_channels if i == 0 else embed_dims * num_heads[i - 1] // self.num_modals,
                embed_dims=embed_dims_i,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                padding=patch_sizes[i] // 2,
                norm_cfg=norm_cfg)
            feature_fusion = FeatureFusion(
                dim=embed_dims_i,  # 使用计算后的 embed_dims_i
                num_modals=num_modals,
                reduction=fusion_reduction,
                num_heads=num_heads[i],  # 使用对应阶段的 num_heads
                norm_layer=nn.BatchNorm2d)
            layer = ModuleList([
                EncoderLayer(
                    embed_dims=embed_dims_i,
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims_i,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    sr_ratio=sr_ratios[i]) for idx in range(num_layer)
            ])
            norm = build_norm_layer(norm_cfg, embed_dims_i)[1]
            self.layers.append(ModuleList([patch_embed, layer, norm, feature_fusion]))
            cur += num_layer


    def forward(self, x):
        """
        Args:
            x (Tensor): 输入多模态特征，形状为 (B, M, C, H, W)，
                        其中 B 为批量大小，M 为模态数，C 为通道数。

        Returns:
            List[Tensor]: 每个阶段的输出特征图。
        """
        B, M, C, H, W = x.shape
        
        x = x.view(B*M, C, H, W)
        outs = []
        for i, layer in enumerate(self.layers):
            # Patch Embedding
            x, hw_shape = layer[0](x)

            # Transformer Encoder Layers
            for block in layer[1]:
                x = block(x, hw_shape)

            # Norm Layer
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)

            x_fused = layer[3](x)
            x = x + x_fused

            if i in self.out_indices:
                outs.append(x)

        return outs
    

    
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modal_1 = torch.zeros(1, 3, 256, 256, device=device)      # 模态 1
    modal_2 = torch.ones(1, 3, 256, 256, device=device)       # 模态 2
    modal_3 = torch.ones(1, 3, 256, 256, device=device)    # 模态 3
    modal_4 = torch.ones(1, 3, 256, 256, device=device)    # 模态 4

    x = torch.stack([modal_1, modal_2, modal_3, modal_4], dim=1).to(device)

    model = MutiModalTransformer().to(device)
    outs = model(x)
    for y in outs:
        print(y.shape)
