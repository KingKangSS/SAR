# src/dataset.py

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SentinelPairDataset(Dataset):
    def __init__(self, base_dir, transform=None):
        self.pair_list = []
        self.transform = transform

        terrains = sorted(os.listdir(base_dir))
        self.label_map = {terrain: idx for idx, terrain in enumerate(terrains)}

        for terrain in terrains:
            sar_dir = os.path.join(base_dir, terrain, "s1")
            optical_dir = os.path.join(base_dir, terrain, "s2")

            sar_files = sorted(os.listdir(sar_dir))
            for sar_file in sar_files:
                optical_file = sar_file.replace("_s1_", "_s2_")
                sar_path = os.path.join(sar_dir, sar_file)
                optical_path = os.path.join(optical_dir, optical_file)

                if os.path.exists(optical_path) and os.path.exists(sar_path):
                    self.pair_list.append((sar_path, optical_path, terrain))
                else:
                    print(f"[Warning] Missing file pair: {sar_path} or {optical_path}")

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, idx):
        sar_path, optical_path, terrain = self.pair_list[idx]

        sar_img = cv2.imread(sar_path, cv2.IMREAD_GRAYSCALE)
        sar_img = np.stack([sar_img]*3, axis=-1)  # SAR → 3채널 확장

        optical_img = cv2.imread(optical_path, cv2.IMREAD_COLOR)
        optical_img = cv2.cvtColor(optical_img, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=sar_img, image2=optical_img)
            sar_img = augmented['image']
            optical_img = augmented['image2']

        label = self.label_map[terrain]
        return sar_img, optical_img, label

# 예시 transform
train_transform = A.Compose([
    A.RandomCrop(256, 256),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
], additional_targets={"image2": "image"})

# 사용 예시:
# dataset = SentinelPairDataset("data/v_2", transform=train_transform)
