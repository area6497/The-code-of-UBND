





import torch
import torch.nn as nn
import torch.nn.functional as F

# 深度可分离卷积
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding,
                                   groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


# 倒残差模块
class InvertedBottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, expansion=2):
        super().__init__()
        hidden_dim = int(in_ch * expansion)
        self.use_res = (stride == 1 and in_ch == out_ch)

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2,
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        out = self.conv(x)
        if self.use_res:
            out += x
        return F.relu(out)


# 轻量化注意力模块（ECA 注意力）
class ECABlock(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)


# MobileNetV4 Backbone
class MobileNetV4(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            # Conv2D-3x3x32
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # FusedIB-3x3
            InvertedBottleneck(32, 64, 3, 2, expansion=2),

            # ExtraDW-5x5
            DepthwiseSeparableConv(64, 128, 5, 2, 2),

            # IB-3x3 (2x)
            InvertedBottleneck(128, 128, 3, 1),
            InvertedBottleneck(128, 128, 3, 1),

            # Attention
            ECABlock(128),

            # ConvNext-3x3-384
            nn.Conv2d(128, 384, 3, 1, 1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            # ExtraDW-3x3
            DepthwiseSeparableConv(384, 256, 3, 2, 1),

            # IB-3x3
            InvertedBottleneck(256, 256, 3, 1)
        )

        self._initialize_weights()

    def forward(self, x):
        # 只返回特征图，不做池化和分类
        x = self.features(x)
        return x  # shape: [N, 256, H, W]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)








