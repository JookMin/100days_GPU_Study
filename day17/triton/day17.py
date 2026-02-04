import random
import torch
import triton
import numpy as np
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def float2Int8(
  input_ptr,
  output_ptr,
  N,
  BLOCK_SIZE: tl.constexpr
):
  pid = tl.program_id(axis=0)
  block_start = pid * BLOCK_SIZE
  offsets = block_start + tl.arange(0, BLOCK_SIZE)
  mask = offsets < N

  x = tl.load(input_ptr + offsets, mask=mask)
  output = tl.clamp(x, -128, 127).to(tl.int8)
  tl.store(output_ptr + offsets, output, mask=mask)

def solve(input: torch.Tensor, output: torch.Tensor, N: int):
  assert input.is_cuda and output.is_cuda

  BLOCK_SIZE = 1024
  grid_size = triton.cdiv(N, BLOCK_SIZE)

  float2Int8[(grid_size,)](
    input,
    output,
    N,
    BLOCK_SIZE=BLOCK_SIZE
  )

if __name__ == "__main__":
  if not torch.cuda.is_available():
    print("CUDA(GPU)를 찾을 수 없습니다.")
  else:
    torch.manual_seed(0)
    N = 1024 * 32

    input = torch.randn(N, device='cuda')
    output = torch.zeros(N, device='cuda')
    eps = 1e-5

    solve(input, output, N)
    output_torch = torch.clamp(input, -128, 127).to(torch.int8)

    if torch.equal(output, output_torch):
      print("✅ Triton 연산 성공! PyTorch 결과와 일치합니다.")
    else:
      print("❌ 연산 결과가 다릅니다.")
      max_diff = (output.float() - output_torch.float()).abs().max()
      print(f"최대 오차: {max_diff}")
