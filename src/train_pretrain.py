import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset_loader import SentinelPairDataset

# 하이퍼파라미터
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
DATA_DIR = "/home/kang/PythonProject/Project 1./SAR/Data/v_2"
CHECKPOINT_DIR = "./checkpoints/pretrain"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Optical 전용 Dataset
class Sentinel2Dataset(torch.utils.data.Dataset):
    def __init__(self, base_dir, transform=None):
        self.dataset = SentinelPairDataset(base_dir, transform)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        _, optical_img, label = self.dataset[idx]
        return optical_img, label

# 데이터 증강
train_transform = A.Compose([
    A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
], additional_targets={"image2": "image"})

# 데이터셋 및 데이터로더
train_dataset = Sentinel2Dataset(DATA_DIR, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 모델 정의 (ImageNet pretrained 사용)
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 4)  # 4개 클래스 분류
model = model.cuda()

# 손실함수 및 최적화기
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 학습 루프
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.cuda(), labels.cuda()

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

    # 체크포인트 저장
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"resnet18_epoch{epoch+1}.pth"))
