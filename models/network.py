import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class HighPassResidualFilter(nn.Module):
    """SRM High-pass kernel to capture invisible AI smoothing & edge blending."""
    def __init__(self):
        super().__init__()
        kernel = torch.tensor([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8,-12,  8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1]
        ], dtype=torch.float32) / 12.0
        
        self.kernel = kernel.view(1, 1, 5, 5).repeat(3, 1, 1, 1)

    def forward(self, x):
        return F.conv2d(x, self.kernel, padding=2, groups=3)

class IDForgeryDetector(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.freq_filter = HighPassResidualFilter()
        
        # Backbone
        weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b4(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        
        # Custom Classification Head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 2)  # [0: Authentic, 1: Forged]
        )

    def forward(self, x):
        residuals = self.freq_filter(x)
        features = x + residuals
        return self.backbone(features)