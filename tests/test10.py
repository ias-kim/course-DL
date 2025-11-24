import torch
from torch import nn, Tensor
from torchvision import datasets, transforms

model = torch.load("models/model_full.pth", weights_only=False)

for module in model:
    # 실행한 AF함수
    print(module)

    # 학습된 레이어(파라미터)값들을 출력
    if isinstance(module, nn.Linear):
        print(module.weight)

    print("*"*20)

transform = transforms.Compose([ transforms.ToTensor() ])
test_ddataset = datasets.MNIST(root="data", download=True, train=False, transform=transform) # 값을 가져올때 텐서로 가져와야하며 정규화를 시켜야함.

img: Tensor
label: int
img, label = test_ddataset[0] # feature, label

pred = model(img) # 저장방식 모델(예전방식임!)
pred = torch.argmax(pred, dim=1)
print(pred) # 벡터형식으로 출력됨.
