import torch

x = torch.Tensor([[1, 2, 3, 4], [4, 5, 6, 7]])
print(x.storage()) # 1, 2, 3, 4, 4, 5, 6, 7
print(x.shape) # -> [2, 3]
print(x.ndim) # -> 2
print(x.dtype) # -> float 32

y = x.view(3, -1)
print (y.shape) # -> [3, 2]
print (y.storage()) # 1, 2, 3, 4, 5, 6

print(y[1, 1]) 


print(x._base) # -> None
print(y._base) # X - storage

z = x.clone()
