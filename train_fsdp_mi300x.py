#!/usr/bin/env python3
"""
PyTorch FSDP Distributed Training on AMD ROCm 6.2 & MI300X
Architecture: Multi-GPU Fully Sharded Data Parallel
"""
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    CPUOffload
)
from transformers import AutoModelForCausalLM

def main():
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    else:
        print("Distributed environment not detected; validation simulation mode.")
        return

    mixed_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.3-70B-Instruct",
        torch_dtype=torch.bfloat16
    )

    fsdp_model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_policy,
        device_id=local_rank,
        cpu_offload=CPUOffload(offload_params=False)
    )
    print("FSDP initialized on AMD MI300X device.")

if __name__ == "__main__":
    main()
