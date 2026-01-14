import random
import torch
import triton
import triton.language as tl

@triton.jit
def fusedAXPBY(
    x_ptr, y_ptr, z_ptr,
    a, b,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = a * x + b * y
    tl.store(z_ptr + offsets, output, mask=mask)

def solve(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, a: float, b: float, N: int):
    assert x.is_cuda and y.is_cuda and z.is_cuda

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    fusedAXPBY[grid](x, y, z, a, b, N, BLOCK_SIZE=BLOCK_SIZE)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA(GPU)를 찾을 수 없습니다.")
    else:
        torch.manual_seed(0)
        size = 98432
        a = random.uniform(0.0, 10.0)
        b = random.uniform(0.0, 10.0)

        x = torch.rand(size, device='cuda')
        y = torch.rand(size, device='cuda')
        z = torch.zeros(size, device='cuda')
        
        solve(x, y, z, a, b, size)
        output_torch = (a * x) + (b * y)

        if torch.allclose(z, output_torch):
            print("✅ Triton 연산 성공! PyTorch 결과와 일치합니다.")
        else:
            print("❌ 연산 결과가 다릅니다.")
