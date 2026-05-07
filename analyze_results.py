#!/usr/bin/env python3
"""Summarize FA3 paged-vs-contiguous decode benchmark JSONL output."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


KEY_FIELDS = (
    "device_name",
    "compute_capability",
    "dtype",
    "q_heads",
    "kv_heads",
    "head_dim",
    "batch_size",
    "seq_len",
    "page_size",
    "num_splits",
    "sm_margin",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze FA3 decode benchmark JSONL results.")
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on {path}:{line_no}: {exc}") from exc
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in KEY_FIELDS)


def overhead_pct(candidate_us: float | None, baseline_us: float | None) -> float | None:
    if candidate_us is None or baseline_us is None or baseline_us == 0:
        return None
    return 100.0 * (candidate_us / baseline_us - 1.0)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(group_key(row), {})[row["layout"]] = row

    summaries: list[dict[str, Any]] = []
    for key, by_layout in sorted(grouped.items()):
        base = by_layout.get("contiguous")
        identity = by_layout.get("paged_identity")
        shuffled = by_layout.get("paged_shuffled")

        base_us = base.get("median_latency_us") if base else None
        identity_us = identity.get("median_latency_us") if identity else None
        shuffled_us = shuffled.get("median_latency_us") if shuffled else None

        summary = {field: value for field, value in zip(KEY_FIELDS, key)}
        summary.update(
            {
                "contiguous_median_us": base_us,
                "paged_identity_median_us": identity_us,
                "paged_shuffled_median_us": shuffled_us,
                "paged_identity_overhead_pct": overhead_pct(identity_us, base_us),
                "paged_shuffled_overhead_pct": overhead_pct(shuffled_us, base_us),
                "contiguous_tokens_per_sec": base.get("tokens_per_sec") if base else None,
                "paged_identity_tokens_per_sec": identity.get("tokens_per_sec") if identity else None,
                "paged_shuffled_tokens_per_sec": shuffled.get("tokens_per_sec") if shuffled else None,
                "all_layouts_present": all(
                    layout in by_layout
                    for layout in ("contiguous", "paged_identity", "paged_shuffled")
                ),
                "identity_correctness_passed": (
                    identity.get("correctness_passed") if identity else None
                ),
                "shuffled_correctness_passed": (
                    shuffled.get("correctness_passed") if shuffled else None
                ),
                "identity_max_abs_diff": identity.get("max_abs_diff") if identity else None,
                "shuffled_max_abs_diff": shuffled.get("max_abs_diff") if shuffled else None,
            }
        )
        summaries.append(summary)
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("no summary rows to write")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def default_output_path(jsonl_path: Path) -> Path:
    return jsonl_path.with_name(f"{jsonl_path.stem}_summary.csv")


def main() -> int:
    args = parse_args()
    rows = read_rows(args.jsonl)
    summary = summarize(rows)
    output_path = args.output or default_output_path(args.jsonl)
    write_csv(output_path, summary)

    missing = sum(1 for row in summary if not row["all_layouts_present"])
    print(f"rows={len(rows)}")
    print(f"summary_rows={len(summary)}")
    print(f"missing_layout_groups={missing}")
    print(f"summary_csv={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
