import torch
from torch import nn, Tensor, optim

# computation Graph

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=False)
c = a ** 2 + b ** 2

c.backward()

print(f"a.gard: {a.grad}")
print(f"b.gard: {b.grad}")
print(c.grad)

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
        
raw_feature:Tensor = Tensor([0, 0], [0, 1], [1, 0], [1, 1])
raw_label:Tensor = Tensor([0, 1, 1, 0]).reshape(-1, 1) # 차원을 맞추기 위한 reshape

model:XorModel = XorModel()
loss_fn:nn.MSELoss = nn.MSELoss()
optimizer:optim.SGD = optim.SGD(model.parameters(), lr=0.01) # optimizer와 model을 연결

# neural network로 구성
for epoch in range(1000):
    # 활성화 함수, 기울기 값 계산을 위한 back pragation 
    pred:Tensor = model(raw_feature) # call
    loss:Tensor = loss_fn(pred, raw_label)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step() # 지정된 step 알고리즘에 따라서 업데이트! 

# --------여기까지 Forward 과정 ------------

# --------지금부터 Back 과정 ---------------



# # obj = MyLayer(2, 2)
# # a = XorModel(obj)

# # for str, param in obj.named_parameters():
# #     print(f"name: {str}, paramse: {param}")

