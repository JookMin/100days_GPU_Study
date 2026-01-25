import random
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def Clip(
    x_ptr, z_ptr,
    lo, hi,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.where(x > hi, hi, x)
    output = tl.where(x < lo, lo, x)
    tl.store(z_ptr + offsets, output, mask=mask)

def solve(x: torch.Tensor, z: torch.Tensor, lo: float, hi: float, N: int):
    assert x.is_cuda and z.is_cuda

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    Clip[grid](x, z, lo, hi, N, BLOCK_SIZE=BLOCK_SIZE)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA(GPU)를 찾을 수 없습니다.")
    else:
        torch.manual_seed(0)
        size = 98432

        x = torch.rand(size, device='cuda')
        z = torch.zeros(size, device='cuda')
        lo = random.uniform(-100.0, 100.0)
        hi = random.uniform(-100.0, 100.0)
        real_lo = min(lo, hi);
        real_hi = max(lo, hi);
        
        solve(x, z, real_lo, real_hi, size)
        output_torch = torch.clip(x, real_lo, real_hi)

        if torch.allclose(z, output_torch):
            print("✅ Triton 연산 성공! PyTorch 결과와 일치합니다.")
        else:
            print("❌ 연산 결과가 다릅니다.")
