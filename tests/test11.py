import torch
from torch import nn,Tensor
from torchvision import datasets, transforms


model = torch.load("models/model_full.pth", weights_only=False)

transform = transforms.Compose([ transforms.ToTensor()])
test_dataset = datasets.MNIST(root="data",download=True,train=False)

# state_dict -> dictionary 정보
save_state = torch.load("models/model_state_5.pth")

print(f"epoch: {save_state['epoch']}")
print(f"epoch_loss: {save_state['epoch_loss']}")

model:nn.Sequential = nn.Sequential(
  nn.Flatten(),
  nn.Linear(28 * 28, 30),
  nn.ReLU(),
  nn.Linear(30, 120),
  nn.ReLU(),
  nn.Linear(120, 10),
  # 마지막 Linear 뒤엔 CrossEntropyLoss가 내부적으로 Softmax를 적용하므로 activation(ReLU)을 넣지 않는다.
)

model.load_state_dict(save_state['model_state'])

for module in model:
    if isinstance(module, nn.Linear):
        print(module.weight)

img: Tensor
label: int
img, label = test_dataset[0]
img, label = test_dataset[0]

print(img)
print(label)

pred = model(img)
pred = torch.argmax(pred, dim=1)
print(pred)