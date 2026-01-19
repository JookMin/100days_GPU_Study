import random
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def GELU(
    x_ptr, z_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    SQRT_2_OVER_PI = 0.7978845608 
    COEF = 0.044715
    x3 = x * x * x
    tanh_in = SQRT_2_OVER_PI * (x + COEF * x3)
    output = 0.5 * x * (1 + libdevice.tanh(tanh_in))
    tl.store(z_ptr + offsets, output, mask=mask)

def solve(x: torch.Tensor, z: torch.Tensor, N: int):
    assert x.is_cuda and z.is_cuda

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    GELU[grid](x, z, N, BLOCK_SIZE=BLOCK_SIZE)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA(GPU)를 찾을 수 없습니다.")
    else:
        torch.manual_seed(0)
        size = 98432

        x = torch.rand(size, device='cuda')
        z = torch.zeros(size, device='cuda')
        
        solve(x, z, size)
        m = torch.nn.GELU(approximate='tanh')
        output_torch = m(x);

        if torch.allclose(z, output_torch):
            print("✅ Triton 연산 성공! PyTorch 결과와 일치합니다.")
        else:
            print("❌ 연산 결과가 다릅니다.")
