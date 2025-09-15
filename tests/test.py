import torch

# 현재 설치된 PyTorch 버전 확인
print("PyTorch version:", torch.__version__)

# CUDA(즉, GPU) 사용 가능 여부 확인
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    # 현재 사용 가능한 GPU(디바이스) 개수
    print("CUDA device count:", torch.cuda.device_count())
    
    # 현재 선택된 GPU ID
    current_device = torch.cuda.current_device()
    print("Current device ID:", current_device)
    
    # 현재 GPU 이름
    print("Current device name:", torch.cuda.get_device_name(current_device))
    
    # 모든 GPU 정보 출력
    for i in range(torch.cuda.device_count()):
        print(f"\n[Device {i}] {torch.cuda.get_device_name(i)}")
        print("  Memory Allocated:", torch.cuda.memory_allocated(i) / 1024**2, "MB")
        print("  Memory Cached:   ", torch.cuda.memory_reserved(i) / 1024**2, "MB")
else:
    print("CUDA is not available. Running on CPU.")
