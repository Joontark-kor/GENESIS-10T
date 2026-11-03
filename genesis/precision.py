"""Pick a numeric precision that the GPU supports natively.

``torch.cuda.is_bf16_supported()`` also returns True on GPUs that only
emulate bf16 (for example the T4 on free Colab), which makes them slow.
These helpers use the compute capability instead: bf16 from Ampere
(8.x: A100, L4, RTX 30/40) on, fp16 on older CUDA GPUs (T4, V100),
fp32 on CPU.
"""
from __future__ import annotations


def native_bf16(torch, use_cuda: bool) -> bool:
    return bool(use_cuda) and torch.cuda.get_device_capability()[0] >= 8


def inference_dtype(torch, use_cuda: bool):
    if not use_cuda:
        return torch.float32
    return torch.bfloat16 if native_bf16(torch, use_cuda) else torch.float16


def describe(torch, use_cuda: bool) -> str:
    return {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}[inference_dtype(torch, use_cuda)]
