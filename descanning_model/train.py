import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from os.path import join
from os import listdir
import time
import torchvision.models as models
import torch.nn.functional as F
import lr_scheduler

# 랜덤시드 고정
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

# 이미지 로드 함수
def load_img(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# 커스텀 데이터셋 클래스
class CustomDataset(data.Dataset):
    def __init__(self, noisy_image_paths, clean_image_paths, patch_size = 128, transform=None):
        self.clean_image_paths = [join(clean_image_paths, x) for x in listdir(clean_image_paths)]
        self.noisy_image_paths = [join(noisy_image_paths, x) for x in listdir(noisy_image_paths)]
        self.transform = transform
        self.patch_size = patch_size

    def __len__(self):
        return len(self.noisy_image_paths)

    def __getitem__(self, index):
        # 이미지 불러오기
        noisy_image = load_img(self.noisy_image_paths[index])
        clean_image = load_img(self.clean_image_paths[index])

        H, W, _ = clean_image.shape

        # 이미지 랜덤 크롭
        rnd_h = random.randint(0, max(0, H - self.patch_size))
        rnd_w = random.randint(0, max(0, W - self.patch_size))
        noisy_image = noisy_image[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
        clean_image = clean_image[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
        
        # transform 적용
        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)
        
        return noisy_image, clean_image
    
# 데이터셋 전처리
train_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.65,0.65,0.65], std=[0.35,0.35,0.35])
])

# 하이퍼파라미터 설정
num_epochs = 100
batch_size = 32

# 데이터 로더
noisy_image_paths = './dataset/train/scan'
clean_image_paths = './dataset/train/clean'
train_dataset = CustomDataset(noisy_image_paths, clean_image_paths, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# GPU 사용 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DnCNN 모델 인스턴스 생성 및 GPU로 이동
model = Descanning().to(device)

# 손실함수
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        features_x = self.feature_extractor(x)
        features_y = self.feature_extractor(y)
        loss = nn.MSELoss()(features_x, features_y)
        return loss

perceptual_loss = PerceptualLoss().to(device)

def combined_loss(out_images, real_images):
    l1_loss = F.l1_loss(out_images, real_images)
    ploss = perceptual_loss(out_images, real_images)

    combined_loss = l1_loss + ploss * 0.1

    return combined_loss

# 옵티마이저 및 lr 스케쥴러
optimizer = optim.AdamW(model.parameters(), lr = 0, weight_decay=0.01)
scheduler = lr_scheduler.CosineAnnealingWarmUpRestarts(optimizer, T_0=30, T_mult=1, eta_max=0.1,  T_up=2, gamma=0.5)

# 훈련 시작
start_time = time.time()

# 훈련
model.train()
best_loss = 9999.0
for epoch in range(num_epochs):
    running_loss = 0.0
    for noisy_images, clean_images in train_loader:
        noisy_images = noisy_images.to(device)
        clean_images = clean_images.to(device)
        optimizer.zero_grad()
        outputs = model(noisy_images)
        loss = combined_loss(outputs, noisy_images - clean_images)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * noisy_images.size(0)
    epoch_loss = running_loss / len(train_dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# 현재 epoch의 loss가 최소 loss보다 작으면 모델 갱신
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), 'model.pth')
        print(f"{epoch+1}epoch 모델 저장 완료")
    
    scheduler.step()

# 훈련 종료
end_time = time.time()

# 훈련 소요 시간 출력
training_time = end_time - start_time

minutes = int(training_time // 60)
seconds = int(training_time % 60)
hours = int(minutes // 60)
minutes = int(minutes % 60)

print(f"훈련 소요 시간: {hours}시간 {minutes}분 {seconds}초")

