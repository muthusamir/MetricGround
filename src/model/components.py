import torch.nn as nn
import torch.nn.functional as F

class MetricInjector(nn.Module):
    """Simple 3D point cloud → metric feature extractor (placeholder)"""
    def __init__(self, in_channels=3, out_dim=1024):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, points):
        # points: [B, N, 3+feat]
        return self.proj(points.mean(dim=1))  # global pooling placeholder

class SemanticMetricFusion(nn.Module):
    def __init__(self, dim=1024, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim)
        )

    def forward(self, semantic, metric):
        # semantic: [B, L, D], metric: [B, 1, D] → treat as single token
        metric = metric.unsqueeze(1)
        fused, _ = self.cross_attn(semantic, metric, metric)
        fused = self.norm(semantic + fused)
        fused = self.norm(fused + self.ffn(fused))
        return fused

class DistillationHead(nn.Module):
    """Predict metric-related tokens for distillation"""
    def __init__(self, dim):
        super().__init__()
        self.head = nn.Linear(dim, dim)

    def forward(self, x):
        return self.head(x)
