import torch
from torch import nn

a = torch.tensor(2.0, requires_grad=True)
b = a ** 2
b.retain_grad()
c = b ** 3

# for debug
# d = c.item() * 4

# d = c.detach() * 4
# detach -> return type : Tensor

c.backward() # 역전파를 진행후 기울기 확인

print(f"b.grad: {b.grad}")


# print(f"a.grad: {a.grad:.2f}")
# print(f"a.grad: {a.grad.item():.2f}")


# a = torch.tensor(2.0)
# b = nn.Parameter(torch.tensor(3.0))

print(f"a.grad: {a.grad}")
print(f"b.grad: {b.grad}")
print(f"a.grad: {a.grad_fn}")
print(f"b.grad: {b.grad_fn}")
print(f"c.grad: {c.grad_fn}")

# y = 3 * X + b

raw_x = torch.tensor([ _ for _ in range(11)], dtype=torch.float32).reshape(-1, 1) # vector
raw_y = raw_x * 3 + 2 # input -> 1, output -> 1
print(raw_x, raw_y)

model:nn.Linear = nn.Linear(1, 1)
loss_fn:nn.MSELoss = nn.MSELoss()
optimizer:optim.SGD = optim.SGD(model.parameters(), lr=0.01) 