import torch
from torch import nn, Tensor
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

# fc1:nn.Linear = nn.Linear()

# model = nn.Sequential(
#     nn.Linear(2, 2),
#     nn.sigmoid(),
#     nn.Linear(2, 2),
#     nn.Sigmoid()
# )

transform = transforms.Compose([
    transforms.ToTensor()
])

mnist_raw = datasets.MNIST(
    root="data",
    download=True,
    train=True,
    transfrom=transform,
)

img, label = mnist_raw[0]


print(img)
# img_raw = np.array(img)

# print(img_raw)