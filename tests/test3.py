import torch

from torch import nn, Tensor

class MLR(nn.Module):
    def __init__ (self, input: int, output: int) -> None:
        super().__init__()
        # 파라미터 클래스형으로 생성해 상속받아 사용
        self.weights = nn.Parameter(torch.randn(input, output, dtype=torch.float32)) 
        self.bias = nn.Parameter(torch.randn(input, dtype=torch.float32))

    def pre_hooks(module, input):
        print(f"invoked pre-hook: {type(module)}, input: {input}")
 plmk5
    def forward(self, x: Tensor) -> Tensor:
        # H(x1, x2, x3) = W1*X1+W2*X2+W3*X3 + b
        return x @ self.weights + self.bias
    
    def post_hooks(module, input, output):
        print(f"invoked post-hook: {type(module)}")
        print(f"input:{input}, output:{output}")

mlr = MLR(3, 1) # 데이터셋의 feature를 체크해야하며, feature 

mlr.forward(Tensor(1, 2, 3))

class Bar:
    # 특정 객체의 멤버 변수를 등록할 시 동작
    def __setattr__(self, name, value):
        print(name, value)

obj = Bar()
obj.test = 10
print(obj.test)