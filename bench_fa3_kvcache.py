#!/usr/bin/env python3
"""Benchmark FA3 decode latency for contiguous vs paged KV-cache layouts."""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any


LAYOUTS = ("contiguous", "paged_identity", "paged_shuffled")
DEFAULT_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128)
DEFAULT_SEQ_LENS = (1024, 2048, 4096, 8192, 16384, 32768)
DEFAULT_PAGE_SIZES = (16, 128)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "FA3-only decode benchmark comparing contiguous KV cache against "
            "paged KV cache with identity and shuffled page tables."
        )
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=list(DEFAULT_BATCH_SIZES))
    parser.add_argument("--seq-lens", nargs="+", type=int, default=list(DEFAULT_SEQ_LENS))
    parser.add_argument("--page-sizes", nargs="+", type=int, default=list(DEFAULT_PAGE_SIZES))
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-splits", type=int, default=0)
    parser.add_argument("--sm-margin", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print the benchmark cases that would run, then exit without importing FA3.",
    )
    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help="Skip paged-vs-contiguous output comparison. Shape and finite checks still run.",
    )
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def dtype_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype is torch.bfloat16:
        return 2e-1, 2e-1
    return 5e-2, 5e-2


def load_fa3() -> tuple[Any, Any, dict[str, Any]]:
    try:
        import flash_attn_interface as fa3_interface  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError(
            "Could not import FA3 module 'flash_attn_interface'. Install FlashAttention-3 "
            "from the Dao-AILab/flash-attention 'hopper' package and run this script from "
            "an environment where that module is importable."
        ) from exc

    try:
        flash_attn_with_kvcache = fa3_interface.flash_attn_with_kvcache
    except AttributeError as exc:
        raise RuntimeError(
            "Imported 'flash_attn_interface', but it does not expose "
            "'flash_attn_with_kvcache'. This does not look like the FA3/Hopper interface."
        ) from exc

    try:
        op_namespace = getattr(torch.ops, "flash_attn_3")
        getattr(op_namespace, "fwd")
    except Exception as exc:
        raise RuntimeError(
            "FA3 imported, but torch.ops.flash_attn_3.fwd is unavailable. The CUDA "
            "extension may not be built/loaded for this environment."
        ) from exc

    package_versions: dict[str, str | None] = {}
    for package_name in ("flash-attn", "flash_attn_3", "flash-attn-3"):
        try:
            package_versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package_name] = None

    metadata = {
        "fa3_module_file": inspect.getfile(fa3_interface),
        "fa3_versions": package_versions,
        "fa3_import_status": "ok",
    }
    return fa3_interface, flash_attn_with_kvcache, metadata


def device_metadata(device: torch.device, fa3_metadata: dict[str, Any]) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(device)
    return {
        "device": str(device),
        "device_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_bytes": props.total_memory,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        **fa3_metadata,
    }


def validate_args(args: argparse.Namespace) -> None:
    positive_fields = {
        "warmup": args.warmup,
        "iters": args.iters,
        "q_heads": args.q_heads,
        "kv_heads": args.kv_heads,
        "head_dim": args.head_dim,
    }
    for name, value in positive_fields.items():
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive, got {value}")
    for name, values in (
        ("batch-sizes", args.batch_sizes),
        ("seq-lens", args.seq_lens),
        ("page-sizes", args.page_sizes),
    ):
        if not values or any(v <= 0 for v in values):
            raise ValueError(f"--{name} must contain positive integers")
    if args.q_heads % args.kv_heads != 0:
        raise ValueError("--q-heads must be divisible by --kv-heads for GQA/MQA")


def iter_cases(args: argparse.Namespace):
    for batch_size in args.batch_sizes:
        for seq_len in args.seq_lens:
            for page_size in args.page_sizes:
                for layout in LAYOUTS:
                    yield {
                        "batch_size": batch_size,
                        "seq_len": seq_len,
                        "page_size": page_size,
                        "layout": layout,
                    }


def print_cases(args: argparse.Namespace) -> None:
    cases = list(iter_cases(args))
    for case in cases:
        print(
            "batch={batch_size} seq={seq_len} page={page_size} layout={layout}".format(
                **case
            )
        )
    print(f"total_cases={len(cases)}")


def make_logical_inputs(
    batch_size: int,
    seq_len: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    q = torch.randn(
        (batch_size, 1, q_heads, head_dim),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    k = torch.randn(
        (batch_size, seq_len, kv_heads, head_dim),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    v = torch.randn(
        (batch_size, seq_len, kv_heads, head_dim),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    return q, k.contiguous(), v.contiguous()


def make_paged_cache(
    logical: torch.Tensor,
    page_size: int,
    shuffle: bool,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len, kv_heads, head_dim = logical.shape
    blocks_per_seq = math.ceil(seq_len / page_size)
    padded_len = blocks_per_seq * page_size

    if padded_len == seq_len:
        padded = logical
    else:
        padded = torch.zeros(
            (batch_size, padded_len, kv_heads, head_dim),
            device=logical.device,
            dtype=logical.dtype,
        )
        padded[:, :seq_len].copy_(logical)

    logical_blocks = padded.view(batch_size, blocks_per_seq, page_size, kv_heads, head_dim)
    logical_blocks = logical_blocks.reshape(batch_size * blocks_per_seq, page_size, kv_heads, head_dim)

    num_blocks = logical_blocks.shape[0]
    if shuffle:
        generator = torch.Generator(device=logical.device)
        generator.manual_seed(seed)
        physical_by_logical = torch.randperm(num_blocks, device=logical.device, generator=generator)
    else:
        physical_by_logical = torch.arange(num_blocks, device=logical.device)

    paged = torch.empty_like(logical_blocks)
    paged[physical_by_logical] = logical_blocks
    page_table = physical_by_logical.view(batch_size, blocks_per_seq).to(torch.int32)
    return paged.contiguous(), page_table.contiguous()


def call_fa3_decode(
    flash_attn_with_kvcache: Any,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_table: torch.Tensor | None,
    num_splits: int,
    sm_margin: int,
) -> torch.Tensor:
    return flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        k=None,
        v=None,
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        causal=True,
        num_splits=num_splits,
        sm_margin=sm_margin,
    )


def percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of an empty list")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def summarize_times_us(times_us: list[float], batch_size: int) -> dict[str, float]:
    median_us = statistics.median(times_us)
    mean_us = statistics.fmean(times_us)
    return {
        "median_latency_us": median_us,
        "mean_latency_us": mean_us,
        "p05_latency_us": percentile(times_us, 0.05),
        "p95_latency_us": percentile(times_us, 0.95),
        "min_latency_us": min(times_us),
        "max_latency_us": max(times_us),
        "tokens_per_sec": batch_size * 1_000_000.0 / median_us,
    }


def assert_output_sane(out: torch.Tensor, batch_size: int, q_heads: int, head_dim: int) -> None:
    expected = (batch_size, 1, q_heads, head_dim)
    if tuple(out.shape) != expected:
        raise RuntimeError(f"unexpected FA3 output shape: got {tuple(out.shape)}, expected {expected}")
    if not torch.isfinite(out).all().item():
        raise RuntimeError("FA3 output contains NaN or Inf")


def compare_outputs(
    candidate: torch.Tensor,
    baseline: torch.Tensor,
    dtype: torch.dtype,
) -> dict[str, Any]:
    diff = (candidate.float() - baseline.float()).abs()
    baseline_abs = baseline.float().abs()
    max_abs = float(diff.max().item())
    max_base = float(baseline_abs.max().item())
    max_rel = max_abs / max(max_base, 1e-6)
    atol, rtol = dtype_tolerances(dtype)
    passed = torch.allclose(candidate.float(), baseline.float(), atol=atol, rtol=rtol)
    return {
        "correctness_compared": True,
        "correctness_passed": bool(passed),
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "atol": atol,
        "rtol": rtol,
    }


def benchmark_callable(
    fn: Any,
    warmup: int,
    iters: int,
) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times_us: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        times_us.append(start.elapsed_time(end) * 1000.0)
    torch.cuda.synchronize()
    return times_us


def make_output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return args.output
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"fa3_decode_{timestamp}.jsonl"
    return args.out_dir / filename


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def run_one_shape(
    args: argparse.Namespace,
    flash_attn_with_kvcache: Any,
    device: torch.device,
    dtype: torch.dtype,
    metadata: dict[str, Any],
    output_path: Path,
    batch_size: int,
    seq_len: int,
    page_size: int,
) -> None:
    case_seed = args.seed + batch_size * 1_000_003 + seq_len * 101 + page_size
    q, logical_k, logical_v = make_logical_inputs(
        batch_size=batch_size,
        seq_len=seq_len,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        dtype=dtype,
        device=device,
        seed=case_seed,
    )
    cache_seqlens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)

    baseline_out: torch.Tensor | None = None

    for layout in LAYOUTS:
        page_table = None
        if layout == "contiguous":
            k_cache = logical_k
            v_cache = logical_v
        else:
            shuffle = layout == "paged_shuffled"
            k_cache, page_table = make_paged_cache(logical_k, page_size, shuffle, case_seed + 17)
            v_cache, v_page_table = make_paged_cache(logical_v, page_size, shuffle, case_seed + 17)
            if not torch.equal(page_table, v_page_table):
                raise RuntimeError("internal error: K/V page tables differ")

        def invoke() -> torch.Tensor:
            return call_fa3_decode(
                flash_attn_with_kvcache=flash_attn_with_kvcache,
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                cache_seqlens=cache_seqlens,
                page_table=page_table,
                num_splits=args.num_splits,
                sm_margin=args.sm_margin,
            )

        torch.cuda.reset_peak_memory_stats(device)
        check_out = invoke()
        torch.cuda.synchronize()
        assert_output_sane(check_out, batch_size, args.q_heads, args.head_dim)

        correctness = {
            "correctness_compared": False,
            "correctness_passed": None,
            "max_abs_diff": None,
            "max_rel_diff": None,
            "atol": None,
            "rtol": None,
        }
        if layout == "contiguous":
            baseline_out = check_out.detach()
        elif not args.skip_correctness and baseline_out is not None:
            correctness = compare_outputs(check_out, baseline_out, dtype)
            if not correctness["correctness_passed"]:
                raise RuntimeError(
                    f"{layout} output diverged from contiguous baseline for "
                    f"batch={batch_size}, seq={seq_len}, page={page_size}: "
                    f"max_abs_diff={correctness['max_abs_diff']}, "
                    f"max_rel_diff={correctness['max_rel_diff']}"
                )

        times_us = benchmark_callable(invoke, warmup=args.warmup, iters=args.iters)
        peak_memory = torch.cuda.max_memory_allocated(device)

        row = {
            **metadata,
            "layout": layout,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "page_size": page_size,
            "q_len": 1,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "dtype": args.dtype,
            "causal": True,
            "num_splits": args.num_splits,
            "sm_margin": args.sm_margin,
            "warmup": args.warmup,
            "iters": args.iters,
            "peak_memory_allocated_bytes": peak_memory,
            **correctness,
            **summarize_times_us(times_us, batch_size),
        }
        write_jsonl(output_path, row)
        print(
            "layout={layout:15s} batch={batch:4d} seq={seq:6d} page={page:4d} "
            "median_us={median:10.3f} tok_s={tok_s:10.1f}".format(
                layout=layout,
                batch=batch_size,
                seq=seq_len,
                page=page_size,
                median=row["median_latency_us"],
                tok_s=row["tokens_per_sec"],
            ),
            flush=True,
        )

        del check_out
        if layout != "contiguous":
            del k_cache, v_cache, page_table
        torch.cuda.empty_cache()

    del q, logical_k, logical_v, cache_seqlens, baseline_out
    torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    validate_args(args)

    if args.list_cases:
        print_cases(args)
        return 0

    global torch
    try:
        import torch
    except Exception as exc:
        print("ERROR: PyTorch is required to run the FA3 decode benchmark.", file=sys.stderr)
        print(json.dumps({"error": str(exc)}, indent=2, sort_keys=True), file=sys.stderr)
        return 2

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available; FA3 decode benchmark requires a CUDA GPU.", file=sys.stderr)
        return 2

    device = torch.device("cuda")
    torch.cuda.set_device(device)
    dtype = torch_dtype(args.dtype)

    try:
        _, flash_attn_with_kvcache, fa3_metadata = load_fa3()
    except Exception as exc:
        props = {}
        if torch.cuda.is_available():
            current_device = torch.device("cuda")
            device_props = torch.cuda.get_device_properties(current_device)
            props = {
                "device_name": device_props.name,
                "compute_capability": f"{device_props.major}.{device_props.minor}",
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
            }
        print("ERROR: FA3 is unavailable in this environment.", file=sys.stderr)
        print(json.dumps({"error": str(exc), **props}, indent=2, sort_keys=True), file=sys.stderr)
        return 3

    metadata = device_metadata(device, fa3_metadata)
    output_path = make_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    print(f"writing_results={output_path}", flush=True)

    try:
        for batch_size in args.batch_sizes:
            for seq_len in args.seq_lens:
                for page_size in args.page_sizes:
                    print(
                        f"\ncase batch={batch_size} seq={seq_len} page={page_size}",
                        flush=True,
                    )
                    run_one_shape(
                        args=args,
                        flash_attn_with_kvcache=flash_attn_with_kvcache,
                        device=device,
                        dtype=dtype,
                        metadata=metadata,
                        output_path=output_path,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        page_size=page_size,
                    )
    except Exception as exc:
        print("ERROR: benchmark failed.", file=sys.stderr)
        print(
            json.dumps(
                {
                    "error": str(exc),
                    "device_name": metadata.get("device_name"),
                    "compute_capability": metadata.get("compute_capability"),
                    "torch_version": metadata.get("torch_version"),
                    "torch_cuda_version": metadata.get("torch_cuda_version"),
                    "fa3_module_file": metadata.get("fa3_module_file"),
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 4

    print(f"\ncompleted_results={output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
