import torch
from torch import nn

class Gsc(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # net input이 계산됨
        print(f"GSC forward in invoked")
        return x

# 객체를 함수처럼 호출 가능한 매직 메소드
class Bar:
    def __call__(self, arg):
        print(f"__call__ is invoked, {arg}")

obj = Bar()
obj(3)


def pre_hooks(module, input):
    print("pre_hooks is invoked")

def post_hooks(module, input):
    print("pre_hooks is invoked")

gsc = Gsc()
# 디버깅 하기 위해서 확인하는 용도
gsc.register_forward_pre_hook(pre_hooks)
gsc.register_forward_hook(post_hooks)
# 하지만 값을 입력받아 그 입력으로 다음 스테이징으로 출력해야 하므로 forward가 가장 중요 -> 메서드를 오버라이딩
# 콜이라는 매직메서드를 호출 받으면 상속받을 수 있음
y = gsc(2)


print(f"output {y}")