import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(
    a_ptr, b_ptr, c_ptr,  # 벡터 a, b, c의 메모리 포인터
    n_elements,           # 요소 개수
    BLOCK_SIZE: tl.constexpr  # 메타파라미터 (컴파일 타임 상수)
):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    output = a + b
    tl.store(c_ptr + offsets, output, mask=mask)


def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    assert a.is_cuda and b.is_cuda and c.is_cuda

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE=BLOCK_SIZE)
    return c

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA(GPU)를 찾을 수 없습니다.")
    else:
        torch.manual_seed(0)
        size = 98432

        x = torch.rand(size, device='cuda')
        y = torch.rand(size, device='cuda')
        output_triton = torch.zeros(size, device='cuda')
        
        solve(x, y, output_triton, size)
        output_torch = x + y

        if torch.allclose(output_triton, output_torch):
            print("✅ Triton 연산 성공! PyTorch 결과와 일치합니다.")
        else:
            print("❌ 연산 결과가 다릅니다.")