import torch

x = torch.tensor([[[1], [2]], [[3], [4]]])

print(f"Raw 데이터:\n", x.storage())

print("Tensor 데이터", x)

print("차원 수", x.ndim)

print("모양(shape)", x.shape)

print("자료형(dtype)", x.dtype) 

origin = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
print(origin.storage())
print(origin.shape)
print(origin.ndim)
print(origin.dtype)

m2_x = origin.view(2, -1)
print(m2_x)
print("m2_x shape:", m2_x.shape)
print("m2_x ndim:", m2_x.ndim)

m3_x = origin.view(3, -1)
print(m3_x)
print("m3_x shape:", m3_x.shape)
print("mx_x ndim", m3_x.ndim)

m1_2_x = origin.view(2, 1, -1)
print(m1_2_x)
print("m1_2_x shape:", m1_2_x.shape)
print("m1_2_x ndim:", m1_2_x.shape)

print("ptr(origin)", origin.storage().data_ptr())
print("ptr(m2_x):", m2_x.storage().data_ptr())
print("ptr(m3_x):", m3_x.storage().data_ptr())
print("ptr(m1_2_x):", m1_2_x.storage().data_ptr())

m1_2_x[0, 0, 0] = 100
print("after edit, m2_x:\n", m2_x)
print("after edit, m3_x:\n", m3_x)
print("after edit, origin:\n", origin)

x = torch.Tensor([1, 2, 3, 4])
y = x.view(2, -1)
z = x.clone()

print(x._base)
print(y._base)
print(z._base)

print(y._base is x)