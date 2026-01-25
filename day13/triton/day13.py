import random
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def batchNormKernel(
    input_ptr, gamma_ptr, beta_ptr, output_ptr,
    N, C,
    stride_input_n, stride_input_c,
    stride_output_n, stride_output_c,
    eps,
    BLOCK_N: tl.constexpr
):
    c = tl.program_id(axis=0)
    n_offsets = tl.arange(0, BLOCK_N)
    mask = n_offsets < N

    offsets = n_offsets * stride_input_n + c * stride_input_c

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / N
    diff = x - mean
    diff = tl.where(mask, diff, 0.0)
    var = tl.sum(diff * diff, axis=0) / N

    x_hat = diff / tl.sqrt(var + eps)
    gamma = tl.load(gamma_ptr + c)
    beta = tl.load(beta_ptr + c)

    output = gamma * x_hat + beta

    tl.store(output_ptr + offsets, output, mask=mask)


def solve(input: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, output: torch.Tensor, eps: float):
    assert input.is_cuda and gamma.is_cuda and beta.is_cuda and output.is_cuda
    N, C = input.shape

    BLOCK_N = triton.next_power_of_2(N)
    grid = (C,)

    batchNormKernel[grid](
        input, gamma, beta, output,
        N, C, 
        input.stride(0), input.stride(1),
        output.stride(0), output.stride(1),
        eps, 
        BLOCK_N=BLOCK_N
    )

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA(GPU)를 찾을 수 없습니다.")
    else:
        torch.manual_seed(0)
        
        N = 32
        C = 64
        
        input = torch.randn((N, C), device='cuda')
        gamma = torch.rand(C, device='cuda')
        beta = torch.rand(C, device='cuda')
        output = torch.zeros((N, C), device='cuda')
        eps = 1e-5
        
        solve(input, gamma, beta, output, eps)
        m = torch.nn.BatchNorm1d(C, eps=eps, momentum=None, affine=True, track_running_stats=False)
        m.weight.data = gamma  # Gamma
        m.bias.data = beta    # Beta
        
        output_torch = m(input)

        if torch.allclose(output, output_torch, atol=1e-4):
            print("✅ Triton 연산 성공! PyTorch 결과와 일치합니다.")
        else:
            print("❌ 연산 결과가 다릅니다.")
            max_diff = (output - output_torch).abs().max()
            print(f"최대 오차: {max_diff}")
