import os
from os import listdir
from os.path import join, splitext
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Normalize, Compose


# 랜덤 시드 고정
np.random.seed(42)

# Descanning 모델
class RCAB(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(RCAB, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(num_channels, num_channels // reduction_ratio, kernel_size=1)
        self.fc2 = nn.Conv2d(num_channels // reduction_ratio, num_channels, kernel_size=1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)

        out_pool = self.avg_pool(out)
        out_pool = self.fc1(out_pool)
        out_pool = self.relu(out_pool)
        out_pool = self.fc2(out_pool)
        out_pool = torch.sigmoid(out_pool)
        out = out * out_pool

        out = out + residual
        return out

class Descanning(nn.Module):
    def __init__(self, num_layers=25, num_channels=64):
        super(Descanning, self).__init__()
        layers = [nn.Conv2d(3, num_channels, kernel_size=3, padding=1), nn.BatchNorm2d(num_channels), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 2):
            rcab = RCAB(num_channels)
            layers.append(rcab)
        layers.append(nn.Conv2d(num_channels, 3, kernel_size=3, padding=1))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out

model = Descanning()
model.load_state_dict(torch.load('model.pth'))
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 데이터셋 경로
noisy_data_path = './dataset/test/scan'
output_path = './output'

if not os.path.exists(output_path):
    os.makedirs(output_path)

# 이미지 로드 함수
def load_img(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# 커스텀 데이터셋
class CustomDatasetTest(data.Dataset):
    def __init__(self, noisy_image_paths, transform=None):
        self.noisy_image_paths = [join(noisy_image_paths, x) for x in listdir(noisy_image_paths)]
        self.transform = transform

    def __len__(self):
        return len(self.noisy_image_paths)

    def __getitem__(self, index):
        
        noisy_image_path = self.noisy_image_paths[index]
        noisy_image = load_img(self.noisy_image_paths[index])
        
        if self.transform:
            noisy_image = self.transform(noisy_image)

        return noisy_image, noisy_image_path

test_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.65,0.65,0.65], std=[0.35,0.35,0.35])
])

# 데이터셋 로드 및 전처리
noisy_dataset = CustomDatasetTest(noisy_data_path, transform=test_transform)

# 데이터 로더 설정
noisy_loader = DataLoader(noisy_dataset, batch_size=1, shuffle=False)

# 이미지 denoising 및 저장
for noisy_image, noisy_image_path in noisy_loader:
    noisy_image = noisy_image.to(device)
    noise = model(noisy_image)
    denoised_image = noisy_image - noise

    # denoised_image를 CPU로 이동하여 이미지 저장
    denoised_image = denoised_image.cpu().squeeze(0)
    mean = torch.tensor([0.65,0.65,0.65])
    std = torch.tensor([0.35,0.35,0.35])
    denoised_image = denoised_image * std.unsqueeze(1).unsqueeze(2) + mean.unsqueeze(1).unsqueeze(2)
    denoised_image = torch.clamp(denoised_image, 0, 1)
    denoised_image = transforms.ToPILImage()(denoised_image)

    # Save denoised image
    output_filename = noisy_image_path[0]
    denoised_filename = output_path + '/' + output_filename.split('/')[-1][:-4] + '.png'
    denoised_image.save(denoised_filename) 
