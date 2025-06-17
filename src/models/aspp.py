import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()

        self.atrous_block1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, dilation=1, padding=0)
        self.atrous_block6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=6, padding=6)
        self.atrous_block12 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=12, padding=12)
        self.atrous_block18 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=18, padding=18)

        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.global_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.conv1 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.atrous_block1(x)
        x2 = self.atrous_block6(x)
        x3 = self.atrous_block12(x)
        x4 = self.atrous_block18(x)

        # Global average pooling branch
        global_feat = self.global_pool(x)
        global_feat = self.global_conv(global_feat)
        global_feat = F.interpolate(global_feat, size=x.shape[2:], mode='bilinear', align_corners=False)

        x_cat = torch.cat([x1, x2, x3, x4, global_feat], dim=1)
        x_cat = self.conv1(x_cat)
        x_cat = self.bn(x_cat)
        return self.relu(x_cat)
