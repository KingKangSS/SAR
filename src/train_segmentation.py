# src/train_segmentation.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset_loader import SentinelPairDataset
from models.segmentation_model import SegmentationModel

# 하이퍼파라미터
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-4
DATA_DIR = "/home/kang/PythonProject/Project 1./SAR/Data/v_2"
CHECKPOINT_DIR = "./checkpoints/segmentation"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ✅ Pretrained optical weight 경로 (이전 step에서 생성된 weight 경로로 교체)
PRETRAINED_BACKBONE_PATH = "./checkpoints/pretrain/resnet18_epoch10.pth"

# ✅ 데이터 증강 정의
train_transform = A.Compose([
    A.RandomCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
], additional_targets={"image2": "image"})

# ✅ Dataset 정의 (SAR + Optical 모두 활용)
class SARSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir, transform=None):
        self.dataset = SentinelPairDataset(base_dir, transform)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sar_img, optical_img, label = self.dataset[idx]
        # 두 이미지를 concatenate → [6, H, W]
        combined_img = torch.cat([sar_img, optical_img], dim=0)
        return combined_img, label

# ✅ DataLoader 준비
train_dataset = SARSegmentationDataset(DATA_DIR, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ✅ 모델 로딩 (ASPP + Pretrained Backbone)
model = SegmentationModel(n_classes=4, pretrained_backbone_path=PRETRAINED_BACKBONE_PATH)
model = model.cuda()

# ✅ 손실함수 및 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ✅ 학습 루프
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.cuda(), labels.cuda()

        outputs = model(imgs)
        # outputs: [N, C, H, W] → segmentation output
        # labels: [N] → scene classification (임시 목적)
        # (이부분은 추후 true segmentation mask 있으면 pixel-wise loss로 변경 가능)

        # 현재 구조에선 simple classification 형태 유지 (디버깅 목적)
        outputs = outputs.mean(dim=[2,3])  # global average pooling 임시 적용
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

    # 체크포인트 저장
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"seg_epoch{epoch+1}.pth"))
