#!/usr/bin/env python3
"""Check whether the FA3 benchmark environment is ready."""

from __future__ import annotations

import json
import sys


def main() -> int:
    status = {
        "python": sys.version,
        "torch_import": "missing",
        "cuda_available": None,
        "device_name": None,
        "compute_capability": None,
        "torch_version": None,
        "torch_cuda_version": None,
        "fa3_import": "missing",
        "fa3_module_file": None,
        "fa3_op": "missing",
    }

    try:
        import torch
    except Exception as exc:
        status["torch_error"] = str(exc)
        print(json.dumps(status, indent=2, sort_keys=True))
        print("\nPyTorch is missing. Run: ./install_fa3_env.sh", file=sys.stderr)
        return 2

    status["torch_import"] = "ok"
    status["torch_version"] = torch.__version__
    status["torch_cuda_version"] = torch.version.cuda
    status["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        status["device_name"] = props.name
        status["compute_capability"] = f"{props.major}.{props.minor}"

    try:
        import flash_attn_interface
        from flash_attn_interface import flash_attn_with_kvcache

        status["fa3_import"] = "ok"
        status["fa3_module_file"] = flash_attn_interface.__file__
        status["fa3_function"] = flash_attn_with_kvcache.__name__
    except Exception as exc:
        status["fa3_error"] = str(exc)
        print(json.dumps(status, indent=2, sort_keys=True))
        print("\nFA3 is missing. Run: ./install_fa3_env.sh", file=sys.stderr)
        return 3

    try:
        getattr(torch.ops, "flash_attn_3").fwd
        status["fa3_op"] = "ok"
    except Exception as exc:
        status["fa3_op_error"] = str(exc)
        print(json.dumps(status, indent=2, sort_keys=True))
        print("\nFA3 imported, but the torch op is missing. Rebuild FA3:", file=sys.stderr)
        print("  REINSTALL_FA3=1 ./install_fa3_env.sh", file=sys.stderr)
        return 4

    print(json.dumps(status, indent=2, sort_keys=True))
    print("\nEnvironment looks ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
