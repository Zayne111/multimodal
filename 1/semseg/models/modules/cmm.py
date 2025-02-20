import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, N, C), where:
               B: Batch size
               N: Sequence length
               C: Feature dimension
        Returns:
            Tensor of shape (B, N, C) after applying self-attention.
        """
        x = x.to(self.qkv.weight.device)
        B, N, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Compute attention
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)

        # Weighted sum
        x = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, C)
        return x

class ModalityDropout(nn.Module):
    def __init__(self, dropout_prob=0.2):
        super(ModalityDropout, self).__init__()
        self.dropout_prob = dropout_prob

    def forward(self, modal_features):
        if self.training:
            B, M, C, H, W = modal_features.shape
            drop_mask = torch.rand(B, M, 1, 1, 1, device=modal_features.device) > self.dropout_prob
            return modal_features * drop_mask
        return modal_features


class ModalitySelector(nn.Module):
    def __init__(self, embed_dim, num_modals, lambda_scene=0.7, lambda_modal=0.3, reduction=2, num_clusters=8):
        super(ModalitySelector, self).__init__()
        self.num_modals = num_modals
        self.lambda_scene = nn.Parameter(torch.tensor(lambda_scene))
        self.lambda_modal = nn.Parameter(torch.tensor(lambda_modal))

        self.scene_classifier = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // reduction, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim // reduction),
            nn.ReLU(inplace=False),
            nn.Conv2d(embed_dim // reduction, num_clusters, kernel_size=1, bias=True), 
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Softmax(dim=-1)
        )

        self.scene_embeddings = nn.Parameter(F.normalize(torch.randn(num_clusters, embed_dim), dim=-1), requires_grad=True)
        self.modal_similarity_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(embed_dim // reduction, 1, kernel_size=1, bias=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.modality_dropout = ModalityDropout(dropout_prob=0.2)

    def forward(self, features):
        B, M, C, H, W = features.size()

        features = self.modality_dropout(features)
        scene_probs = self.scene_classifier(features.view(B * M, C, H, W)).view(B, M, -1).mean(dim=1)
        feature_means = features.view(B * M, C, H, W).mean(dim=[2, 3])
        scene_similarity = F.cosine_similarity(
            feature_means.unsqueeze(1),
            self.scene_embeddings.unsqueeze(0).expand(B * M, -1, -1),
            dim=-1
        ).view(B, M, -1)
        scene_weights = torch.einsum("bms, bs -> bm", scene_similarity, scene_probs)
        modal_similarity = self.modal_similarity_conv(features.view(B * M, C, H, W)).view(B, M)
        scores = self.lambda_scene * scene_weights + self.lambda_modal * modal_similarity
        final_scores = F.relu(scores, inplace=False)  
        residual_weights = F.softmax(features.mean(dim=[2, 3]), dim=2)
        final_scores = final_scores + 0.1 * residual_weights.mean(dim=-1)

        return final_scores
    

class FeatureFusion(nn.Module):
    def __init__(self, dim, num_modals, reduction=2, num_heads=2, norm_layer=nn.BatchNorm2d):
        super(FeatureFusion, self).__init__()
        self.num_modals = num_modals
        self.dim = dim

        # self.cross_modal_attention = CrossAttention(dim=dim, num_heads=num_heads)

        self.fpn = nn.ModuleList([
            nn.Conv2d(dim, dim // 2, kernel_size=1, bias=False),
            nn.Conv2d(dim // 2, dim // 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(dim // 4, dim // 8, kernel_size=3, stride=2, padding=1, bias=False)
        ])

        self.boundary_enhancer = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(dim // reduction, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid()
        )

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(dim, dim // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(dim // reduction, dim, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.feature_fusion = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            norm_layer(dim),
            nn.ReLU(inplace=False)
        )

        self.modality_selector = ModalitySelector(embed_dim=dim, num_modals=num_modals)

    def forward(self, modal_features):
        B, C, H, W = modal_features.shape
        M = self.num_modals
        B = M // B

        # modal_features_seq = modal_features.view(B * M, C, H * W).permute(0, 2, 1)
        # cross_features = self.cross_modal_attention(modal_features_seq)
        # cross_features = cross_features.permute(0, 2, 1).reshape(B * M, C, H, W)

        #modal_features = modal_features.view(B * M, C, H, W)
        boundary_map = self.boundary_enhancer(modal_features)
        cross_features = modal_features * boundary_map

        cross_features = cross_features.reshape(B, M, C, H, W)
        final_scores = self.modality_selector(cross_features)
        final_scores = F.softmax(final_scores, dim=1)  
        final_scores = final_scores.view(B, M, 1, 1, 1)

        weighted_features = cross_features * final_scores

        spatial_weights = self.spatial_attention(weighted_features.reshape(B * M, C, H, W))
        spatial_weights = spatial_weights.reshape(B, M, 1, H, W)
        weighted_features = weighted_features * spatial_weights

        channel_weights = self.channel_attention(weighted_features.reshape(B * M, C, H, W))
        channel_weights = channel_weights.reshape(B, M, C, 1, 1)
        weighted_features = weighted_features * channel_weights

        fused_features = torch.sum(weighted_features, dim=1) 
        fused_features = self.feature_fusion(fused_features)

        return fused_features

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.zeros(4, 256, 16, 16, device=device) 
    featurefusion = FeatureFusion(256, 4).to(device)
    out = featurefusion(x)
    print(out.shape)
