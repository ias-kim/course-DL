
class Bar:
    count = 0
    def __getitem__(self, index):
        return f"__getitem is invokded with index: {index}"
obj = Bar()
print(obj[0], obj[1]) # 특정 객체를 브라켓을 이용해 인덱싱을 할때에 호출이 되며 구현이 안되어 있으면 에러 발생


import torch
from torch.utils.data import Dataset, DataLoader

raw_x = torch.arange(1, 11)
raw_y = raw_x * 2.5 + 0.5 # h (x) = 2.5 * X + -5

print(f"raw_X: {raw_x}") 
print(f"raw_T: {raw_y}") 
# 파이토치의 오브젝트로 데이터를 넘겨줘야함

# 데이터셋 상속
class MyDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
        return super().__getitem__(index) # 해당 인덱스에 해당하는 feature, label 값 반환

    def __len__(self):
        return len(self.x)

dataset = MyDataset(raw_x, raw_y)

s1_x, s1_y = dataset[0]

print(s1_x, s1_y)

dataloader = DataLoader(dataset, shuffle=True, batch_size=5)

for epoch in range(10):
    print(f"{epoch + 1} th epoch")
    for index, (x_val, y_val) in enumerate(dataloader):
        print(f"{index + 1}th step, x_val: {x_val}, y_val: {y_val}")
        # 학습 진행