import torch
import torch.nn as nn
from torchvision import models
from models.aspp import ASPP

class SegmentationModel(nn.Module):
    def __init__(self, n_classes=4, pretrained_backbone_path=None):
        super(SegmentationModel, self).__init__()

        # ResNet18 백본 생성
        backbone = models.resnet18(weights=None)

        if pretrained_backbone_path:
            checkpoint = torch.load(pretrained_backbone_path)

            # 만약 dict 형태라면 model_state_dict 키를 우선 탐색
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            # fc 레이어 weight 제거
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}

            backbone.load_state_dict(state_dict, strict=False)
            print("Loaded pretrained backbone weights from:", pretrained_backbone_path)

        # 백본 분리
        self.initial = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        self.encoder = nn.Sequential(
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )

        # 입력채널 6개로 변경 (SAR + Optical)
        self.initial[0] = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.aspp = ASPP(in_channels=512, out_channels=256)

        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, n_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.encoder(x)
        x = self.aspp(x)
        x = self.decoder(x)
        x = nn.functional.interpolate(x, scale_factor=32, mode='bilinear', align_corners=False)
        return x