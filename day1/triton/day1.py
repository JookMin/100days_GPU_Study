import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(a, b, c, n_elements, BLOCK_SIZE: tl.constexpr):
    blockIdx = tl.program_id(0)
    blockStartIdx = BLOCK_SIZE * blockIdx
    offset = blockStartIdx + tl.arange(0,BLOCK_SIZE)

    mask = offset < n_elements
    x = tl.load(a + offset, mask=mask)
    y = tl.load(b + offset, mask=mask)

    result = x + y;
    tl.store(c + offset, result, mask=mask);

# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE)
