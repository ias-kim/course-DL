import torch 
from torch import nn, Tensor

class MLR(nn.Module):

    def __init__(self, input: int, output: int) -> None:
        super().__init__()
        self.weight: nn.Parameter = nn.Parameter(torch.randn(input, output, dtype=torch.float64)) # 인풋의 개수 3, 아우풋의 개수 1
        self.bias: nn.Parameter = nn.Parameter(torch.randn(output, dtype=torch.float64))
    
    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias

    
mlr = (MLR(3, 1))

raw_feature: Tensor = torch.arange(1, 16, dtype=torch.float64).reshape(-1, 3) # 일부러 3행으로 5줄이 되게끔 설계
label_weight: Tensor = torch.tensor([2, 1.5, 1], dtype=torch.float64).reshape(-1, 1)
label_bias: Tensor = torch.tensor(0.5, dtype=torch.float64)
label_y:Tensor = raw_feature @ label_weight + label_bias

critierion: nn.MSELoss = nn.MSELoss()
# 자동으로 파라미터 딕서녀리로 관리가 되어서 쉽게 사용 가능
optimizer = torch.optim.SGD(mlr.parameters(), 0.001) # 원래 값, 기울기 값이 존재해서 -> 업데이트 가능

for epoch in range(1000):   
    prediction: Tensor = mlr(raw_feature) # 이 모델의 output (y`)

    print(f"Prediction : {prediction}")
    print(f"label: {label_y}")

    # Loss 값 계산
    loss: Tensor = critierion(prediction, label_y)

    # 각 Parameter의 grad 값을 0으로 초기화
    optimizer.zero_grad()

    # 역전파
    loss.backward() # 기울기 속성을 계산

    print(f"before backward: {mlr.weight}") # 역전파를 진행해야지 grad값이 생김
    optimizer.step() # 현재 계산된 기울기로 파라미터를 업데이트 하는 것
    print(f"after backward {mlr.weight}")