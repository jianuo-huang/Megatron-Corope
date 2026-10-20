# CoRoPE context pretraining on Megatron Core

Train a token-only Qwen3 context expert with Megatron Core DDP, distributed
AdamW, and Transformer Engine. Token embeddings and the full vocabulary head
are frozen. Windows have independent causal attention; gradients are normalized
by the global number of valid labels. Dense and shared/routed MoE configurations
are included.

The base is official NVIDIA/Megatron-LM **core_v0.19.2**,
`4b4acac9a1d28ea6829c8d4f566d75698a21249d`. The signed upstream commit and
its complete source tree were verified before the example commits were added.
All implementation changes live in this directory. See [COMMITS.md](COMMITS.md)
for the functional commit sequence and [PROVENANCE.json](PROVENANCE.json) for
source revisions, licenses, and hashes.

## Runtime

Use a CUDA development container. The validated H100 runtime is Python 3.12,
PyTorch **2.11.0+cu130**, Transformer Engine **2.20.2**, Transformers **5.12.1**,
Liger Kernel **0.8.2**, and NumPy **2.3.5**. MCore is imported from this checkout.
Exact Python package pins are in [requirements-training.txt](requirements-training.txt).
The example needs PyTorch's grouped-MM support; older stock Torch environments
are not a tested substitute. Compiled TE extensions must match Torch/CUDA.

In a compatible CUDA 13 container, create an isolated environment with `uv`:

```bash
uv venv examples/corope/.venv --python 3.12
source examples/corope/.venv/bin/activate
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install -r examples/corope/requirements-training.txt
```

This is a runtime recipe, not a newly built container image. The verification
record identifies the environment actually tested. Keep upstream `uv.lock` and
container recipes intact. `run.sh` puts this checkout and the example on
`PYTHONPATH`; it defaults `NVTE_FLASH_ATTN=0` and `NVIDIA_TF32_OVERRIDE=0`.

## Data

Provide local Qwen3-8B tokenizer, embedding, and head files with `--target`.
No model weights or corpus shards are included. Data preparation reads bounded,
provenance-preserving JSONL records or pinned source manifests. Document/group
splitting happens before window packing, and windows never cross documents.

```bash
python examples/corope/prepare_domino_pretrain_data.py --help
python examples/corope/prepare_domino_pretrain_scale.py --help
```

The resulting data directory contains the manifest and per-source window arrays
used by `WindowData`. Use `--window 64` consistently with preparation. Source
names and mixture validation are defined in `pretraining/data.py`. Download
helpers accept `--proxy` and otherwise use `HTTPS_PROXY`/`https_proxy`.

## Training

From the repository root, set local inputs and use a fresh output directory:

```bash
export TARGET=/path/to/Qwen3-8B
export DATA=/path/to/windows
export OUTPUT=/path/to/new-run
export SHUFFLE_CACHE=/path/to/permutations
python examples/corope/prepare_shuffle.py \
  --data "$DATA" --cache "$SHUFFLE_CACHE" --max-tokens 1000000000 --seed 42
COROPE_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 bash examples/corope/run.sh \
  --target "$TARGET" --draft-config examples/corope/configs/moe1536.json \
  --data "$DATA" --output "$OUTPUT" --window 64 \
  --sampler global-shuffle --shuffle-cache "$SHUFFLE_CACHE" \
  --micro-batch 64 --accumulation 1 --max-tokens 1000000000 \
  --log-interval 50 --eval-interval 1000 --save-interval 1000 \
  --validation-windows 1024 --deterministic --wandb-mode disabled \
  --moe-expert-dtype bfloat16
```

`--moe-expert-dtype input` is the default and preserves the original precision
path. `bfloat16` only changes routed-expert grouped GEMMs; FP32 master parameters,
optimizer states, routing, SwiGLU arithmetic, and weighted accumulation remain.
BF16 changes rounding and the training trajectory. It is an explicit experiment
choice recorded in the strict checkpoint recipe.

`configs/dense1536.json` selects the dense control. `--engine hf-fused` and
`--engine hf-chunked` are reference/benchmark controls; the distributed resume
path is MCore. The standalone HF CLI is `reference_train.py`.

W&B is disabled by default. Opt in with `--wandb-mode online --wandb-project NAME`
and, if needed, `--wandb-entity ENTITY --wandb-base-url URL`. Supply credentials
through the normal W&B environment/login mechanism.

## Resume and transfer

Each saved step contains native distributed optimizer/model state, per-rank RNG,
and an `export/` with transferable expert weights and metadata. Resume with the
same command and `--resume "$OUTPUT/step_001000"`. Change only `--stop-after` to
extend a deliberately interrupted token-budget run; keep the original schedule,
paths, sampling, precision, world size, and source files.

Strict resume checks the recipe, target/data identities, software versions, and
source hashes. Moving from the original Dmax checkout to this example changes
source identity, so old native checkpoints are intentionally rejected. For a
new experiment, use `--init-expert /path/to/old/step_N/export` to import validated
weights with fresh optimizer/sampler state. Do not change precision midway
through an existing comparison run.

## Verification

CPU tests need the packages in `requirements-cpu.txt`:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest --import-mode=importlib -q examples/corope/tests
```

GPU tests skip automatically when CUDA is unavailable. Real tokenizer and
external joint-model checks are opt-in through `COROPE_TARGET` and
`COROPE_JOINT_SOURCE` (an existing Dmax-DFlash checkout). Synthetic CPU tests do
not require either asset.

In the validated GPU runtime:

```bash
export PYTHONPATH="$PWD:$PWD/examples/corope${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES=0 python examples/corope/check_models.py \
  --config examples/corope/configs/moe1536.json --result /path/to/model-check.json
CUDA_VISIBLE_DEVICES=0,1 python examples/corope/tests/verify_gpu.py \
  --target "$TARGET" --config examples/corope/configs/moe1536.json \
  --data /path/to/small-validation-windows --output /path/to/new-gate-run
```

The GPU harness checks continuous 6 steps against 2 steps plus resume to 6,
including weights, optimizer, RNG, sampler, and per-step numerical metrics. It
also verifies precision-changing resume is rejected. Add `--joint-source` to
exercise the actual external joint loader; add `--reference-export` to compare
the final export SHA256 with a prior identical experiment. Omitted joint checks
are explicitly recorded as not run.

## Performance and follow-up

The pre-relocation two-H100 benchmark measured **42.406 → 32.900 ms/step**
(**+28.9% timed training throughput**) for opt-in BF16 expert GEMMs. Four paired
400-step runs used the same real checkpoint and batches, discarding 50 warmup
steps. Timing includes sampling, forward/backward, optimizer, communication,
and scheduled diagnostics; it excludes subsequent logging, validation, and
checkpoint I/O. These numbers are neither full end-to-end throughput nor a
measurement of production 16-GPU MFU.

[PERFORMANCE.md](PERFORMANCE.md) records numerical checks, profile attribution,
and remaining work. The largest profiled kernel group is now the frozen
vocabulary head/CE input gradient. Next steps are long-run BF16 quality checks,
measuring the actual 16-GPU communication critical path, and validating dispatch/
combine fusion on full training steps. Changing the vocabulary or objective is
outside these implementation optimizations.

## Licensing

Megatron-LM keeps its upstream license. Imported CoRoPE/SpecForge-derived code
retains the MIT notice in [LICENSES/SpecForge-MIT.txt](LICENSES/SpecForge-MIT.txt).
The import comes from `jianuo-huang/Dmax-DFlash`; provenance is recorded per file.
