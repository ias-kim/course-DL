import torch

from torch import nn, Tensor

torch.manual_seed(0)

# 예측값
class MLR(nn.Module):
    def __init__(self, input:int, output:int) -> None:
        super().__init__()
        self.weight: nn.Parameter = nn.Parameter(torch.randn(input, output, dtype=torch.float32)) # 인풋의 개수 3, 아우풋의 개수 1
        self.bias: nn.Parameter = nn.Parameter(torch.randn(output, dtype=torch.float32))
    
    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias

obj = MLR(3, 1)

# h(x1, x2, x3), 2 * 1  1.5  * 2 + 1 * 3 + 0.5
# 정답값
raw_feature: Tensor = torch.arange(1, 16, dtype=torch.float64).reshape(-1, 3) # 일부러 3행으로 5줄이 되게끔 설계
label_weight: Tensor = torch.tensor([2, 1.5, 1], dtype=torch.float64).reshape(-1, 1)
label_bias: Tensor = torch.tensor(0.5, dtype=torch.float64)
label_y:Tensor = raw_feature @ label_weight + label_bias

print(raw_feature) # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 '''' 15
print(label_weight)
print(label_y)