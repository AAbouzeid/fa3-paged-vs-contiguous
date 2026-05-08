# FA3 Paged vs Contiguous Decode Benchmark

This experiment compares FlashAttention-3 decode latency for three KV-cache layouts:

- `contiguous`: dense KV cache, no page table.
- `paged_identity`: paged KV cache with ordered physical blocks.
- `paged_shuffled`: paged KV cache with randomized physical blocks.

The benchmark is FA3-only. It does not fall back to vLLM, FA2, FA4, Torch SDPA, Triton, or custom CUDA.

## Why Decode Only

The vLLM RFC discussion asks whether modern FA3 paged attention decode kernels on H100 remove the performance reason for vAttention. Decode is the relevant isolated kernel path because it reads an existing long KV cache for each generated token. This benchmark therefore uses `q_len=1`, an already-populated KV cache, and no new `k`/`v` append.

## What Is Measured

For each `(batch_size, seq_len, page_size)` case, the script creates the same logical tensors for all layouts:

```text
Q: [batch_size, 1, q_heads, head_dim]
K: [batch_size, seq_len, kv_heads, head_dim]
V: [batch_size, seq_len, kv_heads, head_dim]
```

Defaults match a Llama-style GQA decode shape:

```text
dtype = bf16
q_heads = 32
kv_heads = 8
head_dim = 128
causal = true
```

The three layouts represent the same logical KV contents:

- `contiguous`: passes K/V directly as `[batch_size, seq_len, kv_heads, head_dim]` with `page_table=None`.
- `paged_identity`: scatters K/V into `[num_blocks, page_size, kv_heads, head_dim]` and uses an ordered `page_table`.
- `paged_shuffled`: scatters K/V into the same paged shape, but randomly permutes physical block IDs in the `page_table`.

The timed region is only:

```python
flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    cache_seqlens=seq_len,
    page_table=page_table_or_none,
    causal=True,
)
```

Timing uses CUDA events after warmup. The output token count is `batch_size` per decode call, so `tokens/sec` is computed from median latency.

## What Is Not Measured

This is intentionally a kernel-layout microbenchmark. It does not measure:

- CUDA VMM mapping overhead from vAttention.
- vLLM scheduling, block manager, request batching, or prefix-cache logic.
- prefill or prefix-cache prefill skipping.
- appending new K/V into the cache during decode.
- end-to-end serving throughput.

Those are useful follow-up experiments, but this benchmark isolates the specific FA3 contiguous-vs-paged decode question.

## Correctness Checks

Before timing each case, the script checks:

- FA3 output shape is `[batch_size, 1, q_heads, head_dim]`.
- output contains no NaN/Inf.
- paged outputs match the contiguous output with relaxed BF16/FP16 tolerance.

## Requirements

- PyTorch with CUDA.
- FlashAttention-3 installed from the `hopper` package so this import works:

```python
from flash_attn_interface import flash_attn_with_kvcache
```

DGX Spark should run the same code path. If FA3 cannot launch on its GB10 Blackwell GPU, the script exits with device and FA3 error metadata instead of producing fallback data.

## Install

Use `python3` on Ubuntu/DGX Spark. The plain `python` command may not exist until a virtual environment is active.

First make sure basic system tools exist:

```bash
python3 --version
nvidia-smi
nvcc --version
```

If `venv`, `git`, or compiler tooling is missing on Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv git build-essential
```

Then create the environment and build FA3:

```bash
./install_fa3_env.sh
source .venv/bin/activate
python check_env.py
```

By default the installer performs a minimal FA3 build for this experiment: H100/SM90, BF16, head dimension 128, forward-only, with paged KV enabled. This avoids compiling hundreds of unused FA3 variants. To build all FA3 variants instead:

```bash
MINIMAL_FA3_BUILD=0 ./install_fa3_env.sh
```

You do not need to rerun the installer after every login. If the `.venv` directory is still there, just activate it:

```bash
source .venv/bin/activate
```

The installer is idempotent: it skips the large PyTorch install if CUDA PyTorch is already importable, and it skips the FA3 build if `flash_attn_interface` is already importable. To force reinstall:

```bash
REINSTALL=1 ./install_fa3_env.sh
```

If PyTorch works but FA3 is missing, rebuild only FA3:

```bash
REINSTALL_FA3=1 ./install_fa3_env.sh
python check_env.py
```

The installer chooses a PyTorch CUDA wheel index from `nvcc --version` first, because FA3 is compiled locally and PyTorch must match the CUDA toolkit used by `nvcc`. If `nvcc` cannot be queried, it falls back to `nvidia-smi`:

- CUDA 13.x -> `https://download.pytorch.org/whl/cu130`
- CUDA 12.8/12.9 -> `https://download.pytorch.org/whl/cu128`
- CUDA 12.6/12.7 -> `https://download.pytorch.org/whl/cu126`

Override if needed:

```bash
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 ./install_fa3_env.sh
```

If FA3 fails with a CUDA mismatch like `detected CUDA version (12.8) mismatches PyTorch (13.0)`, reinstall Torch to match `nvcc`:

```bash
REINSTALL_TORCH=1 REINSTALL_FA3=1 PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 ./install_fa3_env.sh
```

If FA3 compile jobs are killed, lower parallelism and keep the minimal build:

```bash
rm -rf .deps/flash-attention/hopper/build
MAX_JOBS=8 REINSTALL_FA3=1 ./install_fa3_env.sh
```

FA3 is built from the upstream FlashAttention `hopper` directory. If this fails on DGX Spark, that is likely an FA3/Hopper-vs-Blackwell compatibility issue rather than a benchmark issue.

## Quick Subset

Run the same benchmark path on a small grid:

```bash
python bench_fa3_kvcache.py \
  --batch-sizes 1 4 \
  --seq-lens 1024 2048 \
  --page-sizes 16 \
  --warmup 10 \
  --iters 30
```

## Full Grid

```bash
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

Key columns:

- `contiguous_median_us`
- `paged_identity_median_us`
- `paged_shuffled_median_us`
- `paged_identity_overhead_pct`
- `paged_shuffled_overhead_pct`

Interpretation:

- Near-zero overhead means FA3's paged decode path is effectively as fast as contiguous decode for that shape.
- Higher `paged_identity` overhead suggests page-table/layout overhead in the FA3 kernel.
- Higher `paged_shuffled` than `paged_identity` suggests physical block ordering/fragmentation matters.

## Useful Sanity Check

Before running on a GPU, list the planned cases:

```bash
python bench_fa3_kvcache.py --list-cases
```
