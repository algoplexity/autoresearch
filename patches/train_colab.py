"""
Colab-oriented notes and helpers for adapting karpathy/autoresearch training.

This is not a full upstream replacement. It documents the main changes needed
for Colab-friendly experimentation:
- replace missing kernels.py integration
- prefer flash-attn directly
- reduce batch size
- allow float16 fallback on T4
"""

import torch

if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    DTYPE = torch.bfloat16 if major >= 8 else torch.float16
else:
    DTYPE = torch.float32

DEVICE_BATCH_SIZE = 16

def recommended_dtype():
    return DTYPE

def recommended_device_batch_size():
    return DEVICE_BATCH_SIZE

def flash_attention_note():
    return {
        "replace_missing_kernels_py": True,
        "import": "from flash_attn import flash_attn_func",
        "call": "flash_attn_func(q, k, v, causal=True, window_size=window_size)",
    }

if __name__ == "__main__":
    print("Recommended dtype:", recommended_dtype())
    print("Recommended DEVICE_BATCH_SIZE:", recommended_device_batch_size())
    print("Flash attention adaptation:", flash_attention_note())