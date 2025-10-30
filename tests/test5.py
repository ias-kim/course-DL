import torch
from torch import nn


x = torch.randint(1, 4, (1, 3), dtype=torch.float32)
print(x)
w = torch.randint(1, 13, (3, 4), dtype=torch.float32)
print(w)
y = x @ w
print(y)

# Layer -> Forward 
class MyLayer(nn.Module):
    # input, output
    # input: X1 -> Xn: 입력값의 개수
    # output:  N1 -> Nn: 노드의 개수
    def __init__(self, input: int, output: int)->None:
        super().__init__()
        self.weight:nn.Parameter = ...
        self.bias:nn.Parameter = ...
        # 노드의 개수는 편향값에 맞게 조절

    def forward(self, x: nn.Tensor) -> nn.Tensor:
        ...

