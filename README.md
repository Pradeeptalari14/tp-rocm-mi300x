# AMD ROCm 6.2 & Instinct MI300X AI Training & Inference Blueprint

[![ROCm 6.2 CI](https://github.com/Pradeeptalari14/tp-rocm-mi300x/actions/workflows/rocm-ci.yml/badge.svg)](https://github.com/Pradeeptalari14/tp-rocm-mi300x/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Target: gfx942](https://img.shields.io/badge/AMD%20Instinct-MI300X%20(192GB)-rose)](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)

Production-grade deployment templates, Kubernetes device manifests, and Triton/HIP kernels for scaling large-scale LLMs (Llama-3.3 70B, DeepSeek-V2.5, Qwen2.5) on **AMD ROCm 6.2** and **AMD Instinct™ MI300X** accelerators.

![AMD ROCm 6.2 & Instinct MI300X Flow](docs/rocm_mi300x_flow.png)

## Key Architecture Advantages

- **192 GB HBM3 per GPU**: 1.536 TB aggregate high-speed VRAM per 8-GPU node.
- **5.3 TB/s Memory Bandwidth**: Extreme throughput for memory-bound generative LLM decode phases.
- **vLLM ROCm Native**: PagedAttention and FlashAttention-2 optimized for AMD Matrix Cores (`gfx942`).
- **Open Standards**: Fully open HIP and Triton software stack with Zero Vendor Lock-in.

## Repository Contents

- `vllm_mi300x_serve.py`: vLLM high-throughput inference server with ROCm FlashAttention-2.
- `train_fsdp_mi300x.py`: PyTorch FSDP multi-node distributed training script.
- `rocm_triton_kernel.py`: High-performance custom Triton kernel targeting `gfx942`.
- `k8s-mi300x-cluster.yaml`: Kubernetes Deployment utilizing `amd.com/gpu` device plugin.
- `scripts/validate.sh`: Validation script for ROCm SMI and compilation checks.

## Quickstart

```bash
# Clone the repository
git clone https://github.com/Pradeeptalari14/tp-rocm-mi300x.git
cd tp-rocm-mi300x

# Run validation suite
./scripts/validate.sh
```

## License

MIT © 2026 Pradeep Talari
