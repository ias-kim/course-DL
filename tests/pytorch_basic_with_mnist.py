# mnist_mlp.py
# PyTorch 기본 구조를 보여주는 MNIST 분류 예제

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# -----------------------------
# 1. 데이터셋 & 데이터로더
# -----------------------------
transform = transforms.ToTensor()  # PIL 이미지 → FloatTensor (0~1 범위), HWC → CHW 변환

train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform) 
test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)

# dataset class 주요 속성
# self.data             → 이미지 데이터 (torch.Tensor, uint8, [N,28,28])
# self.targets          → 라벨(정답) 정보 (torch.Tensor, [N])
# self.transform        → 입력 데이터 변환 함수 (ToTensor, Normalize 등)
# self.target_transform → 라벨 변환 함수 (예: one-hot 인코딩)
#   - ToTensor: 0~255 정수 → 0.0~1.0 실수
#   - Normalize: 평균/표준편차 정규화 (추가 변환 시)

# 예제
# data, target = train_dataset[0]
# print(data.shape, target)  # data: torch.Size([1,28,28]), target: 5

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
# DataLoader 동작:
#  - batch_size=64 → 64장 묶음 단위로 모델에 입력
#  - shuffle=True → epoch마다 데이터 순서를 무작위로 섞음
#  - 1 step(iteration) = 1 배치 학습 (64장)
#  - 1 epoch = 전체 데이터셋을 한 번 모두 학습
# 사용 예:
# images, labels = next(iter(train_loader))


# -----------------------------
# 2. 모델 정의
# -----------------------------
# class MLP(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(28*28, 128)  # 입력층 → 은닉층z
#         self.fc2 = nn.Linear(128, 10)     # 은닉층 → 출력층 (10 클래스)

#     def forward(self, x):
#         x = x.view(-1, 28*28)             # (배치, 채널, 높이, 너비) → (배치, 784)
#         x = torch.relu(self.fc1(x))       # 활성화 함수 ReLU
#         x = self.fc2(x)                   # 출력 (로짓)
#         return x

# model = MLP()

# -----------------------------
# 3. 손실 함수 & Optimizer
# -----------------------------
# criterion = nn.CrossEntropyLoss()           # 분류 문제 → 교차 엔트로피 손실
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# -----------------------------
# 4. 학습 루프
# -----------------------------
# for epoch in range(1, 6):  # 5 epoch 학습
#     model.train()
#     for images, labels in train_loader:
#         # 순전파
#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         # 역전파
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     print(f"Epoch [{epoch}/5], Loss: {loss.item():.4f}")

# -----------------------------
# 5. 평가
# -----------------------------
# model.eval()
# correct, total = 0, 0
# with torch.no_grad():  # 평가 단계에서는 기울기 계산 불필요
#     for images, labels in test_loader:
#         outputs = model(images)
#         _, predicted = torch.max(outputs, 1)  # 가장 높은 값의 인덱스 = 예측 클래스
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f"Test Accuracy: {100 * correct / total:.2f}%")
