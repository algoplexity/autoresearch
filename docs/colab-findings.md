# Colab findings for `karpathy/autoresearch`

## Summary

This project can be adapted to Google Colab, but not fully out of the box.

## Main blockers

### Missing `kernels.py`
`train.py` imports a custom kernel helper:

```python
from kernels import get_kernel
```

The referenced helper is not available in the repository snapshot we analyzed, so Colab execution will fail unless this is replaced.

## Recommended code adaptation

Remove the custom Flash Attention 3 wrapper usage and instead use the standard Flash Attention package.

### Replace
```python
from kernels import get_kernel
cap = torch.cuda.get_device_capability()
repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
fa3 = get_kernel(repo).flash_attn_interface
```

### With
```python
from flash_attn import flash_attn_func
```

### And replace
```python
y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
```

### With
```python
y = flash_attn_func(q, k, v, causal=True, window_size=window_size)
```

## Precision guidance

### Preferred
- Colab Pro with A100 or L4
- use `torch.bfloat16`

### Workaround
- Colab Free with T4
- replace `bfloat16` with `float16`

## Memory guidance

The default batch size is likely too large for Colab. Start lower:

```python
DEVICE_BATCH_SIZE = 16
```

Try:
- 16 first
- then 32 if stable
- 64 only if memory allows

Because gradient accumulation is already used, lowering the device batch size should preserve effective training behavior while fitting into memory.

## Suggested Colab steps

### Install dependencies
```bash
pip install rustbpe tiktoken pyarrow requests
pip install flash-attn --no-build-isolation
```

### Prepare small test data
```bash
python prepare.py --num-shards 4
```

### Run training
```bash
python train.py
```

## Notes

- `uv run train.py` is not required in Colab.
- Standard `python train.py` is sufficient.
- Start with a smoke test before attempting longer runs.