import torch


# tensor : PyTorch에서 데이터를 저장하는 기본 자료형
# - 구조(차원) : Scholar, Vector, Matrix, N-dimentional Matrix
# - 자료형 : 
#   . 텐서 내 모든 데이터의 자료형은 동일해야 한다.
#   . 텐서 내 모든 차원은 정방 격자 구조를 유지 (regular n-dimentional array)
 
# 1) 텐서의 핵심 속성
#  shape / ndim: 차원과 크기 x.shape, x.ndim
#  dtype: 자료형(연산·정밀도·메모리 영향) x.dtype
#  device: CPU/GPU 위치 x.device (.to("cuda"), .cuda(), .cpu())

x = torch.tensor( ((1, 2), (3, 4)))
print(x.shape)
print(x.ndim)
print(x.device)
print(x.dtype)
print(x.numel())

# 2) 생성 
# 기본 생성
#  - 직접/난수/범위: tensor, zeros/ones/full, rand/randn, arange/linspace
#  - 직접 입력 값 : 스칼라, 튜플/리스트, 넘파이
#  - 형태 지정 : 

y0 = torch.zeros(10)
y1 = torch.ones(10)
yf = torch.full((2, 5), 7)

print(y0)  
print(y1)  
print(yf)


ru = torch.rand(3, 4)
rn = torch.randn(2, 3)
ri = torch.randint(0, 10, (2, 3, 3))

print(ru)
print(rn)  
print(ri)

# 크기/속성 따라하기: zeros_like, rand_like
# - shape, dtype, device는 같고 해당 값으로 초기화

origin = torch.tensor((2, 3), dtype=torch.float32, device="cpu")
zlike = torch.zeros_like(origin)
olike = torch.ones_like(origin)
rlike = torch.rand_like(origin)
rnlike = torch.randn_like(origin)

print(f"zlike : {zlike}, dtype : {zlike.dtype}, shape: {zlike.shape}, device : {zlike.device}")
print(f"zlike : {olike}, dtype : {olike.dtype}, shape: {olike.shape}, device : {olike.device}")
print(f"zlike : {rlike}, dtype : {rlike.dtype}, shape: {rlike.shape}, device : {rlike.device}")
print(f"zlike : {rnlike}, dtype : {rnlike.dtype}, shape: {rnlike.shape}, device : {rnlike.device}")

# 3) 인덱싱/슬라이싱/마스킹
# 기본/슬라이스: x[0], x[:, 1:3]
# 불리언 마스크: x[x > 0]
# 고급 인덱싱: gather/scatter, 임베딩/룩업
# 인덱스 dtype은 보통 int64(long)

# 4) 브로드캐스팅 규칙
# 뒤에서부터 차원 맞춰 비교, 1 또는 동일 크기면 확장 가능
# 예) (N,1) + (1,D) → (N,D)
# expand vs repeat
# expand: 메모리 복제 없이 뷰(읽기 위주)
# repeat: 실제 복제(메모리 증가)

# 5) 형태 변환(메모리 포함)
# view / reshape: 원소 재배치 없이 보기 변경
# view는 contiguous 필요, 아니면 x.contiguous().view(...)
# permute / transpose: 차원 순서 바꿈(메모리 stride 변경)
# squeeze / unsqueeze: 크기 1 차원 제거/추가
# cat / stack: 결합(기존 축/새 축)

# 6) 연산·축소(reduction)
# 합/평균/최댓값 등: sum, mean, max, min
# dim=으로 축 지정, keepdim=True로 차원 유지
# 안정성: 로그·소프트맥스는 logsumexp, cross_entropy 등 안정화 버전 사용

# 7) dtype 규칙 & 프로모션
# 기본: float32, int는 int64
# 연산 시 암묵 승격: bool < int < float < complex
# float16 + bfloat16 → float32, float32 + float64 → float64
# 혼합정밀(AMP): with torch.autocast(..., dtype=torch.bfloat16|float16): ...
# 보통 연산은 BF16/FP16, 가중치는 FP32(마스터 웨이트)

# 8) Autograd 핵심
# requires_grad=True 파라미터는 leaf node(사용자 생성 텐서)
# loss.backward() → w.grad에 누적
# 다음 스텝 전 grad 초기화: optimizer.zero_grad() 또는 w.grad.zero_()
# in-place 연산(접미사 _) 주의: 역전파에 필요한 저장 텐서를 덮어쓰면 에러/불안정
# no_grad / detach
# with torch.no_grad(): → 추론/업데이트 시 그래프 추적 중단
# y = x.detach() → 같은 데이터 공유, 계산 그래프에서 분리
# x.clone()은 데이터 복제(grad 경로 유지), x.detach().clone()은 둘 다

# 9) 성능 & 메모리 팁
# GPU 이동 일괄 처리: x = x.to(device, dtype=torch.float32)
# 핀 메모리 DataLoader: pin_memory=True(CPU→GPU 전송 효율↑)
# contiguous 필요 시 명시: x = x.contiguous()
# 큰 텐서 조각 참조 vs 복제: 뷰(view/expand) 우선 고려
# 정수·bool 텐서는 grad 불가 (손실/가중치는 float 계열)

# 10) 랜덤성 & 재현성
# 시드: torch.manual_seed(0) (+ CUDA면 torch.cuda.manual_seed_all(0))
# 완전 재현: 백엔드 설정 필요(CUDNN 등). 디버깅 때만 권장.

# 11) 도메인별 shape 관례
# CV: NCHW (배치, 채널, 높이, 너비)
# NLP: (batch, seq_len) 또는 (seq_len, batch) 프레임워크별 상이
# 시계열: (batch, time, features)가 흔함