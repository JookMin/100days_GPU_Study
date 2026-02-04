import random
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def Dot(
  A_ptr,
  B_ptr,
  output_ptr,
  N,
  BLOCK_SIZE: tl.constexpr
):
  pid = tl.program_id(axis=0)
  block_start = pid * BLOCK_SIZE
  offsets = block_start + tl.arange(0, BLOCK_SIZE)
  mask = offsets < N

  A = tl.load(A_ptr + offsets, mask=mask)
  B = tl.load(B_ptr + offsets, mask=mask)
  output = A * B
  tl.store(output_ptr + offsets, output, mask=mask)

def solve(A: torch.Tensor, B: torch.Tensor, output: torch.Tensor, N: float):
  assert A.is_cuda and B.is_cuda and output.is_cuda

  BLOCK_SIZE = 1024
  grid_size = triton.cdiv(N, BLOCK_SIZE)

  Dot[(grid_size,)](
    A,
    B,
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

    A = torch.randn(N, device='cuda')
    B = torch.randn(N, device='cuda')
    output = torch.zeros(N, device='cuda')
    eps = 1e-5

    solve(A, B, output, N)
    output_torch = A * B

    if torch.allclose(output, output_torch, atol=1e-4):
      print("✅ Triton 연산 성공! PyTorch 결과와 일치합니다.")
    else:
      print("❌ 연산 결과가 다릅니다.")
      max_diff = (output - output_torch).abs().max()
      print(f"최대 오차: {max_diff}")
