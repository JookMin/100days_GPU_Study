from random import randint, random
import torch
import triton
import triton.language as tl

@triton.jit
def vector_scalar_kernel(
    x_ptr, y_ptr, alpha,  # 벡터 a, b, c의 메모리 포인터
    n_elements,           # 요소 개수
    BLOCK_SIZE: tl.constexpr  # 메타파라미터 (컴파일 타임 상수)
):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    output = x * alpha
    tl.store(y_ptr + offsets, output, mask=mask)

def solve(x: torch.Tensor, y: torch.Tensor, alpha: float, N: int) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    vector_scalar_kernel[grid](x, y, alpha, N, BLOCK_SIZE=BLOCK_SIZE)
    return y

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA(GPU)를 찾을 수 없습니다.")
    else:
        torch.manual_seed(0)
        size = 98432

        x = torch.rand(size, device='cuda')
        alpha = torch.rand(1, device='cuda')
        y = torch.zeros(size, device='cuda')
        
        solve(x, y, alpha, size)
        output_torch = x * alpha

        if torch.allclose(y, output_torch):
            print("✅ Triton 연산 성공! PyTorch 결과와 일치합니다.")
        else:
            print("❌ 연산 결과가 다릅니다.")
