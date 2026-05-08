#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
FLASH_ATTN_REPO_DIR="${FLASH_ATTN_REPO_DIR:-.deps/flash-attention}"
MAX_JOBS="${MAX_JOBS:-8}"
REINSTALL="${REINSTALL:-0}"
REINSTALL_TORCH="${REINSTALL_TORCH:-${REINSTALL}}"
REINSTALL_FA3="${REINSTALL_FA3:-${REINSTALL}}"
MINIMAL_FA3_BUILD="${MINIMAL_FA3_BUILD:-1}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found. Install Python 3 first." >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -m venv --help >/dev/null 2>&1; then
  echo "ERROR: Python venv support is missing." >&2
  echo "On Ubuntu, install it with: sudo apt-get install -y python3-venv" >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git not found. Install git first." >&2
  exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not found. FlashAttention-3 builds from CUDA sources." >&2
  echo "Install a CUDA toolkit >= 12.3, then retry." >&2
  exit 1
fi

if [[ -z "${PYTORCH_INDEX_URL:-}" ]]; then
  cuda_from_nvcc=""
  if command -v nvcc >/dev/null 2>&1; then
    cuda_from_nvcc="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\),.*/\1/p' | head -1)"
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    cuda_from_smi="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1)"
  fi
  cuda_for_torch="${cuda_from_nvcc:-${cuda_from_smi:-}}"

  case "${cuda_for_torch}" in
    13.*)
      PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu130"
      ;;
    12.8*|12.9*)
      PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
      ;;
    12.6*|12.7*)
      PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"
      ;;
    *)
      PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
      ;;
  esac
fi

echo "Using Python: ${PYTHON_BIN}"
echo "Using venv: ${VENV_DIR}"
echo "Using PyTorch index: ${PYTORCH_INDEX_URL}"
echo "Using nvcc: $(nvcc --version | sed -n 's/.*release \([0-9.]*\),.*/\1/p' | head -1)"
echo "Using MAX_JOBS=${MAX_JOBS} for FA3 build"
echo "Using REINSTALL_TORCH=${REINSTALL_TORCH}, REINSTALL_FA3=${REINSTALL_FA3}"
echo "Using MINIMAL_FA3_BUILD=${MINIMAL_FA3_BUILD}"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip "setuptools<82" wheel
if [[ "${REINSTALL_TORCH}" == "1" ]] || ! python - <<'PY'
try:
    import torch
    raise SystemExit(0 if torch.cuda.is_available() else 1)
except Exception:
    raise SystemExit(1)
PY
then
  if [[ "${REINSTALL_TORCH}" == "1" ]]; then
    python -m pip install --force-reinstall torch --index-url "${PYTORCH_INDEX_URL}"
  else
    python -m pip install --upgrade torch --index-url "${PYTORCH_INDEX_URL}"
  fi
else
  echo "PyTorch with CUDA is already importable; skipping torch install."
  python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY
fi
python -m pip install -r requirements.txt

if [[ "${REINSTALL_FA3}" == "1" ]] || ! python - <<'PY'
try:
    import torch
    from flash_attn_interface import flash_attn_with_kvcache
    getattr(torch.ops, "flash_attn_3").fwd
    raise SystemExit(0)
except Exception:
    raise SystemExit(1)
PY
then
  mkdir -p "$(dirname "${FLASH_ATTN_REPO_DIR}")"
  if [[ ! -d "${FLASH_ATTN_REPO_DIR}/.git" ]]; then
    git clone https://github.com/Dao-AILab/flash-attention.git "${FLASH_ATTN_REPO_DIR}"
  else
    git -C "${FLASH_ATTN_REPO_DIR}" pull --ff-only
  fi

  (
    cd "${FLASH_ATTN_REPO_DIR}/hopper"
    if [[ "${MINIMAL_FA3_BUILD}" == "1" ]]; then
      export FLASH_ATTENTION_DISABLE_BACKWARD=TRUE
      export FLASH_ATTENTION_DISABLE_SM80=TRUE
      export FLASH_ATTENTION_DISABLE_FP16=TRUE
      export FLASH_ATTENTION_DISABLE_FP8=TRUE
      export FLASH_ATTENTION_DISABLE_HDIM64=TRUE
      export FLASH_ATTENTION_DISABLE_HDIM96=TRUE
      export FLASH_ATTENTION_DISABLE_HDIM192=TRUE
      export FLASH_ATTENTION_DISABLE_HDIM256=TRUE
      export FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE
      export FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE
      export FLASH_ATTENTION_DISABLE_APPENDKV=TRUE
      export FLASH_ATTENTION_DISABLE_LOCAL=TRUE
      export FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE
      echo "Building minimal FA3: sm90 BF16 hdim128 forward/paged decode support."
    fi
    MAX_JOBS="${MAX_JOBS}" python setup.py install
  )
else
  echo "FlashAttention-3 is already importable; skipping FA3 build."
fi

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
from flash_attn_interface import flash_attn_with_kvcache
import flash_attn_interface
print("FA3 module:", flash_attn_interface.__file__)
print("FA3 kvcache function:", flash_attn_with_kvcache.__name__)
PY

echo
echo "Environment ready. Activate it with:"
echo "  source ${VENV_DIR}/bin/activate"
