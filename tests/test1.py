import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

class Bar:
    def __init__(self):
        self.data = [10, 20, 30, 40, 50]
    # magic method -> 특별한 기능을 수행하는 메서드
    # 파이썬 인터프리터에 의해 호출, 사전에 정의된 메서드

    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)

obj = Bar()
print(obj[2]) # 인덱스 연산을 할시에 인덱스 숫자와, 객체의 주소가 주어지고, 어떤 값을 반환할지 명시를 해준다.
print(len(obj)) # 리스트, 튜플 등을 매직메서드가 구현이 되어 있다는 것.


x = torch.arange(6) # 0, 1, 2, 3, 4, 5 -> Tensor
y = torch.arange(10, 15) # 10, 11, 12, 13, 14, 15 -> Tensor

x = np.random.randint(1, 100, (6,))
y = np.random.randint(200, 300, (6,))
# Dataset
class MyDataset(Dataset):
    def __init__(self, features, lables):
        self.features = features.from_numpy(features)
        self.lables = lables.from_numpy(lables)

        def __getitem__(self, index):
            return self.features[index], self.lables[index]
        
        def __len__(self):
            return len(self.features)

obj = MyDataset(x, y)
print(obj[4]) # 4 -> 14
print(len(obj)) # 6


# 텐서 데이터셋 구성

x = torch.arange(4)         #[0, 1, 2, 3] # -> list
y = torch.arange(10, 14)    #[10, 11, 12, 13] # -> list

dataset = TensorDataset(x, y)
loader = DataLoader(dataset, shuffle=True, batch_size=2)

# 자동적으로 데이터셋의 샘플들을 읽어오며 각 샘플들의 feature과 label 값을 튜플 값으로 나눔.
# DataLoader Class -> __next__ -> dataset[idx] : __getitem__(self,idx) -> idex: feature, lable
# 1th sample -> last sample
for epoch in range(2): # 에폭이 시작될 때에 셔플링이 시작됨.
    print(f"{epoch + 1} : th Epoch")
    for index, (feature, lable) in enumerate(loader, start=1): # batch size -> step() parameter update 
        print(f"{index}th: feature: {feature}, lable: {lable}")

# 1th Epoch
# 1th (0, 1) (10, 11)
# 2th (2, 3) (12, 13)

# 2th Epoch
# 1th (0, 1) (10, 11)
# 2th (2, 3) (12, 13)


class Bar:
    def __init__(self):
        self.data = [10, 20, 30]
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current < len(self.data):
            index = self.current
            self.currnet += 1
            return self.data[index]
        else:
            raise StopIteration()
        
obj = Bar()

for value in obj:
    print(value)
