import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.is_available()) # cuda 환경인지 확인

x = torch.tensor(2, 0, device=device) # 옵션으로 어디에 놓을지 (GPU, CPU)인가?
