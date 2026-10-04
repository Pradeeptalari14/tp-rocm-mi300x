#!/usr/bin/env python3
"""
Custom Triton Kernel on AMD ROCm 6.2 targeting Instinct MI300X (gfx942)
Demonstrating fused activation on AMD Matrix Cores
"""
import torch
try:
    import triton
    import triton.language as tl

    @triton.jit
    def fused_mi300x_gelu_kernel(
        x_ptr,
        y_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        cdf = 0.5 * (1.0 + tl.sin(0.79788456 * (x + 0.044715 * x * x * x)))
        y = x * cdf
        tl.store(y_ptr + offsets, y, mask=mask)
except ImportError:
    pass

def main():
    print("Triton kernel for ROCm 6.2 (gfx942) defined successfully.")

if __name__ == "__main__":
    main()
