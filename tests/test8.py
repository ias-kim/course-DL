
from pickletools import optimize
import torch
from torch import nn, optim, Tensor
from torch.utils import data
from torchvision import datasets, transforms
# x = torch.tensor(2.0, requires_grad=True)
# y = x ** 2
# z = y ** 3
# z.backward()
# print(x.grad) # 내부적으로 데이터를 지워버림.
# z.backward()
# # y = x.item()
# print(x, f"type: {x}")

# print(x, f"type: {y}")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu') # mac -> meta
transform = transforms.Compose([
  transforms.ToTensor()
])  

train_dataset = datasets.MNIST(
  root="data",
  download=True,
  train=True,
  transform=transform
)

test_dataset = datasets.MNIST(
  root="data",
  download=True,
  train=False,
  transform=transform
)

train_dataloader = data.DataLoader(
  train_dataset,
  batch_size=300,
  shuffle=True
  )

test_dataloader = data.DataLoader(
  test_dataset,
  batch_size=128, 
  # batch_size는 데이터셋의 크기가 크면 있는게 좋고 작거나 적당하면 없어도 된다.
  # shuffle은 테스트하는거기 때문에 필요가없다.
  )

model:nn.Sequential = nn.Sequential(
  nn.Flatten(),
  nn.Linear(28 * 28, 30),
  nn.ReLU(),
  nn.Linear(30, 120),
  nn.ReLU(),
  nn.Linear(120, 10),
  # 마지막 Linear 뒤엔 CrossEntropyLoss가 내부적으로 Softmax를 적용하므로 activation(ReLU)을 넣지 않는다.
).to(device) # 모델이 생성할 때에 파라미터 값을 GPU에 올림.

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

##################################################
# - 학습 (에폭, 배치)
##################################################
epoch = 10
for epoch in range(1, epoch + 1):
    epoch_loss = 0.0
    for X, y in train_dataloader:
        X: Tensor
        y: Tensor

        x = X.to(device) # Feature/samples -> Memory in GPU
        y = y.to(device) # Label/samples -> Memory in GPU

        ## Forward propagation
        logits:Tensor = model(X)
        loss:Tensor = criterion(logits, y) # y값의 범위 0 ~ 11 인덱스, 레이블의 범위가 반드시 일치해야함.
        
        ## Backward protpagation
        optimizer.zero_grad()
        loss.backward()
        
        ##update parameters
        optimizer.step()     
        
        epoch_loss += loss.item() * X.size(0) #item을 한이유 
        
        
    print(f"{epoch}th epoch: loss:{epoch_loss/len(train_dataset)}")


# torch.save()
torch.save(model, "models/model_full.pth")

# state_dic
checkpoints = {
    "model_state": model.state_dict(), # 상태값 파라미터 값들을 저장.
    "optimizer": optimizer.state_dict(),
    "epoch": epoch,
    "epoch_loss": epoch_loss,
}

torch.save(checkpoints, f"models/model_state_{epoch}.pth") # 매 에폭마다 달리하여 저장


# #########################################
# #### 평가(test) #################################
# 우항에 있는 매직메서드를 찾는다. 구문이 다 실행시 exit라는 구문을 실행함.
# 현재 torch computation graph 생성을 끊어버린다.
with torch.no_grad():
    correct = 0
    for images, target in test_dataloader:
        images = images.to(device)
        target = target.to(device)
        print(images.size())
        ## forward
        legits = model(images)
        
        pred = torch.argmax(logits, dim=1) # 가장 큰 값을 찾아서 몇번째 인덱스인지 확인 (0 -> 첫번재 행 기준, 1 -> 첫번째 열 기준)

        ## comparison
        correct += (pred == target).sum().item() # 그 텐서 값에서 본연의 자료형(스칼라)로 변형

        exit(0)

        print(images.size())

        print(pred[0]) # -> Vector의 요소가 10개

    
    print(f"correct answer ratio: {correct/len(test_dataset):.4f}")

