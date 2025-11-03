import torch
from torch import nn, Tensor


# x = torch.randint(1, 4, (1, 3), dtype=torch.float32)
# print(x)
# w = torch.randint(1, 13, (3, 4), dtype=torch.float32)
# print(w)
# y = x @ w
# print(y)


# Layer -> Forward 
class MyLayer(nn.Module):
    # input, output
    # input: X1 -> Xn: 입력값의 개수
    # output:  N1 -> Nn: 노드의 개수
    def __init__(self, input: int, output: int)->None:
        super().__init__()
        self.weight:nn.Parameter = nn.Parameter(torch.randn(input, output, dtype=torch.float32))
        self.bias:nn.Parameter = nn.Parameter(torch.randn(output, dtype=torch.float32))
        # 노드의 개수는 편향값에 맞게 조절

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias

    def af(self):
        ...

class SigmoidFunc(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor)->Tensor:
        return 1.0 / (1.0 + torch.exp(-x))

class XorModel(nn.Module):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        # 모델의 요소들을 객체화 (레이어, AF)
        self.fc1:MyLayer = MyLayer(2, 2)
        self.fc2:MyLayer = MyLayer(2, 1)
        self.af:SigmoidFunc = SigmoidFunc()

        # 연결 텐서가 흘러가는 길
        def forward(self, x: Tensor) -> Tensor:
            x = self.fc1(x)
            x = self.af(x)
            x = self.fc2(x)
            x = self.af(x)
            return x

obj = MyLayer(2, 2)
a = XorModel(obj)

for str, param in obj.named_parameters():
    print(f"name: {str}, paramse: {param}")

