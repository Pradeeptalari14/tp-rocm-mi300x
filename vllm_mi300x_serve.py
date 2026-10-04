#!/usr/bin/env python3
"""
High-Throughput vLLM Serving on AMD ROCm 6.2 & Instinct MI300X
Model: meta-llama/Llama-3.3-70B-Instruct
Backend: ROCm HIP + FlashAttention-2 + PagedAttention
"""
import os
import torch
from vllm import LLM, SamplingParams

os.environ["HIP_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["VLLM_ROCM_FLASH_ATTN"] = "1"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"

def main():
    print("Initializing vLLM Engine on ROCm 6.2 with AMD Instinct MI300X...")
    llm = LLM(
        model="meta-llama/Llama-3.3-70B-Instruct",
        tensor_parallel_size=8,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.92,
        enforce_eager=False
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=256
    )

    prompts = [
        "Explain the high-density HBM3 architecture of AMD Instinct MI300X."
    ]

    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        print(f"Output: {output.outputs[0].text.strip()[:160]}...")

if __name__ == "__main__":
    main()
