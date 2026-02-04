import torch
import triton
import triton.language as tl

# Triton 커널 정의
@triton.jit
def int8_quant_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    scale, zpA, zpB, zpC,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # 1. 프로그램 ID (Grid 위치) 가져오기
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # 2. 오프셋 초기화
    # A는 (M, K) 행렬, B는 (K, N) 행렬
    # A의 블록 포인터: (pid_m * BLOCK_M) 행부터 시작
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # 포인터 계산
    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # 3. Accumulator 초기화 (정확도를 위해 float32 사용)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # 4. K 차원을 따라 블록 단위로 루프 (Loop over K)
    for k in range(0, K, BLOCK_SIZE_K):
        # 메모리 로드 (범위 벗어남 방지 mask는 K가 블록의 배수라 가정하거나 mask 추가 필요)
        # 여기서는 단순화를 위해 K가 BLOCK_SIZE_K의 배수라고 가정하거나, 로드 시 마스킹
        a_mask = (offs_am[:, None] < M) & (k + offs_k[None, :] < K)
        b_mask = (k + offs_k[:, None] < K) & (offs_bn[None, :] < N)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # De-quantize (연산을 위해 float 변환 및 Zero Point 차감)
        # int8 -> float32 변환 후 연산
        a_f = a.to(tl.float32) - zpA
        b_f = b.to(tl.float32) - zpB

        # Dot Product Accumulation
        accumulator += tl.dot(a_f, b_f)

        # 포인터 이동
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # 5. Re-quantization (Scale -> Add ZP -> Round -> Clamp -> Cast)
    # 결과 = (Accumulator * scale) + zpC
    output = accumulator * scale + zpC
    
    # 반올림 (round)
    output = tl.libdevice.round(output)
    
    # 클램핑 (int8 범위: -128 ~ 127)
    output = tl.clamp(output, -128, 127)
    
    # int8로 캐스팅
    c = output.to(tl.int8)

    # 6. 결과 저장
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    tl.store(c_ptrs, c, mask=c_mask)

def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, scale: float, zpA: int, zpB: int, zpC: int):
    # Check constraints
    assert A.is_cuda and B.is_cuda and C.is_cuda
    M, K = A.shape
    K_Check, N = B.shape
    assert K == K_Check, "Matrix dimensions mismatch"

    # Block sizes (튜닝 가능)
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32

    # Grid definition (M과 N을 커버하도록 설정)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N'])
    )

    # Kernel launch
    int8_quant_matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        scale, zpA, zpB, zpC,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA(GPU)를 찾을 수 없습니다.")
    else:
        torch.manual_seed(0)
        
        # Dimensions
        M = 64
        N = 32
        K = 64

        # 1. 데이터 생성 (Int8)
        # -128 ~ 127 사이의 랜덤 정수 생성
        A = torch.randint(-128, 127, (M, K), dtype=torch.int8, device='cuda')
        B = torch.randint(-128, 127, (K, N), dtype=torch.int8, device='cuda')
        # 출력 텐서 (초기화)
        C_triton = torch.zeros((M, N), dtype=torch.int8, device='cuda')
        
        # Quantization Parameters
        scale = 0.005
        zpA = 5   # 임의의 Zero Point
        zpB = -5
        zpC = 10
        
        # 2. Triton 실행
        solve(A, B, C_triton, scale, zpA, zpB, zpC)
        
        # 3. PyTorch 검증 (Reference Implementation)
        # 로직: Output = Clamp(Round( ( (A-zpA) @ (B-zpB) ) * scale + zpC ))
        
        # 계산을 위해 float32로 변환
        A_f = A.to(torch.float32) - zpA
        B_f = B.to(torch.float32) - zpB
        
        # 행렬 곱
        acc = torch.matmul(A_f, B_f)
        
        # Re-quantization
        res = acc * scale + zpC
        res = torch.round(res)
        res = torch.clamp(res, -128, 127)
        output_torch = res.to(torch.int8)

        # 4. 결과 비교
        if torch.all(C_triton == output_torch):
            print("✅ Triton 연산 성공! PyTorch 결과와 완벽하게 일치합니다.")
            print(f"Sample Output (Triton):\n{C_triton[:2, :4]}")
        else:
            print("❌ 연산 결과가 다릅니다.")
            # 차이 분석
            diff = (C_triton.float() - output_torch.float()).abs()
            print(f"최대 오차(값 차이): {diff.max().item()}")
            print(f"불일치 개수: {(C_triton != output_torch).sum().item()} / {M*N}")
            
            print("Triton Sample:\n", C_triton[:2, :4])
            print("Torch Sample:\n", output_torch[:2, :4])