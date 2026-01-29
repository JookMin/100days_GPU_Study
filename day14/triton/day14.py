import random
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def partialSum(
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

  x_sq = x * x
  block_sum = tl.sum(x_sq, axis=0)

  tl.store(output_ptr + pid, block_sum)

@triton.jit
def RMSNorm(
  input_ptr,
  output_ptr,
  gamma,
  beta,
  smr,
  N,
  BLOCK_SIZE: tl.constexpr
):
  pid = tl.program_id(axis=0)
  block_start = pid * BLOCK_SIZE
  offsets = block_start + tl.arange(0, BLOCK_SIZE)
  mask = offsets < N

  x = tl.load(input_ptr + offsets, mask=mask)
  output = (x / smr) * gamma + beta
  tl.store(output_ptr + offsets, output, mask=mask)

def solve(input: torch.Tensor, gamma: float, beta: float, output: torch.Tensor, eps: float):
  assert input.is_cuda and output.is_cuda
  N = input.numel()

  BLOCK_SIZE = 1024
  grid_size = triton.cdiv(N, BLOCK_SIZE)

  partial_sums = torch.empty(grid_size, device=input.device, dtype=torch.float32)

  partialSum[(grid_size,)](
    input, partial_sums,
    N,
    BLOCK_SIZE=BLOCK_SIZE
  )

  total_sum_val = torch.sum(partial_sums)
  mean_val = total_sum_val / N
  smr = torch.sqrt(mean_val + eps).item()

  RMSNorm[(grid_size,)](
    input,
    output,
    gamma,
    beta,
    smr,
    N,
    BLOCK_SIZE=BLOCK_SIZE
  )

if __name__ == "__main__":
  if not torch.cuda.is_available():
    print("CUDA(GPU)를 찾을 수 없습니다.")
  else:
    torch.manual_seed(0)
    N = 1024 * 32;
    
    input = torch.randn(N, device='cuda')
    output = torch.zeros(N, device='cuda')
    gamma = random.uniform(0.1, 10.0)
    beta = random.uniform(-10.0, 10.0)
    eps = 1e-5
    
    solve(input, gamma, beta, output, eps)
    rms_torch = torch.sqrt(torch.mean(input ** 2) + eps)
    output_torch = (input / rms_torch) * gamma + beta

    if torch.allclose(output, output_torch, atol=1e-4):
      print("✅ Triton 연산 성공! PyTorch 결과와 일치합니다.")
    else:
      print("❌ 연산 결과가 다릅니다.")
      max_diff = (output - output_torch).abs().max()
      print(f"최대 오차: {max_diff}")
