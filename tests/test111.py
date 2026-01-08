import torch
from torch import nn, optim, Tensor
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# nn.Conv2d 어떻게 처리할 지에 따라 차원구조를 바꿔야한다. -> 이미지일 경우 2D

# reproducibility
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

# transform
transform = transforms.Compose([
    transforms.ToTensor(),
])

# dataset (MNIST는 28x28 작은 이미지 → GPU 이득이 매우 작음)
train_dataset = datasets.CIFAR10(
    root="data", train=True, download=True, transform=transform
)
test_dataset = datasets.CIFAR10(
    root="data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)

img1: Tensor
img1, label = train_dataset[0]

print(img1.shape)
print(label)
conv1 = nn.Conv2d()