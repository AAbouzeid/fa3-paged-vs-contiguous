# FA3 Paged vs Contiguous Decode Benchmark

This experiment compares FlashAttention-3 decode latency for three KV-cache layouts:

- `contiguous`: dense KV cache, no page table.
- `paged_identity`: paged KV cache with ordered physical blocks.
- `paged_shuffled`: paged KV cache with randomized physical blocks.

The benchmark is FA3-only. It does not fall back to vLLM, FA2, FA4, Torch SDPA, Triton, or custom CUDA.

## Why Decode Only

The vLLM RFC discussion asks whether modern FA3 paged attention decode kernels on H100 remove the performance reason for vAttention. Decode is the relevant isolated kernel path because it reads an existing long KV cache for each generated token. This benchmark therefore uses `q_len=1`, an already-populated KV cache, and no new `k`/`v` append.

## Requirements

- PyTorch with CUDA.
- FlashAttention-3 installed from the `hopper` package so this import works:

```python
from flash_attn_interface import flash_attn_with_kvcache
```

DGX Spark should run the same code path. If FA3 cannot launch on its GB10 Blackwell GPU, the script exits with device and FA3 error metadata instead of producing fallback data.

## Quick Subset

Run the same benchmark path on a small grid:

```bash
cd experiments/fa3_paged_vs_contiguous
python bench_fa3_kvcache.py \
  --batch-sizes 1 4 \
  --seq-lens 1024 2048 \
  --page-sizes 16 \
  --warmup 10 \
  --iters 30
```

## Full Grid

```bash
cd experiments/fa3_paged_vs_contiguous
python bench_fa3_kvcache.py \
  --batch-sizes 1 2 4 8 16 32 64 128 \
  --seq-lens 1024 2048 4096 8192 16384 32768 \
  --page-sizes 16 128 \
  --warmup 100 \
  --iters 500
```

Results are written under `results/*.jsonl`.

## Analyze

```bash
python analyze_results.py results/<run>.jsonl
```

The analyzer writes `results/<run>_summary.csv` with contiguous latency, paged latency, and paged-overhead percentages.

## Useful Sanity Check

Before running on a GPU, list the planned cases:

```bash
python bench_fa3_kvcache.py --list-cases
```
