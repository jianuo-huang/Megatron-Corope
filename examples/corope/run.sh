#!/usr/bin/env bash
set -euo pipefail
COROPE_EXAMPLE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
COROPE_MEGATRON_DIR=$(cd -- "$COROPE_EXAMPLE_DIR/../.." && pwd)
export PYTHONPATH="$COROPE_MEGATRON_DIR:$COROPE_EXAMPLE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export NVTE_FLASH_ATTN=${NVTE_FLASH_ATTN:-0}
export NVIDIA_TF32_OVERRIDE=${NVIDIA_TF32_OVERRIDE:-0}
exec python -m torch.distributed.run --standalone --nproc-per-node="${COROPE_GPUS:-1}" "$COROPE_EXAMPLE_DIR/train.py" "$@"
