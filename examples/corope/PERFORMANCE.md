# BF16 expert GEMM evidence (2026-10-20)

These measurements were made **before relocation** in Dmax-DFlash commit
`6e30151307c41a5aeeec92975ef70de11e7a5628`, against its `input` precision control.
The packaged example is verified separately in `verification/relocation.json`.
Raw traces, weights and data are not stored in Git.

## Measurement protocol

Two H100 80GB HBM3 GPUs; Torch 2.11.0+cu130, MCore 0.19.2, TE 2.20.2;
deterministic algorithms, TF32 disabled. W64, microbatch 64 per GPU,
accumulation 1, head chunk 512; MoE1536 with one shared expert and 16 routed
experts, top-k 3, expert intermediate size 1024. All four runs imported the same
real step 50,000 export, initialized new optimizers, and used the same real data,
global shuffle, seed 42, and 400-step LR schedule.

Runs were serialized in input/BF16/BF16/input order. Discard steps 1–50; aggregate
steps 51–400 using the maximum rank duration per step. These durations include
sampling, forward/backward, optimizer, communication, and scheduled diagnostics.
They exclude subsequent logging, evaluation, saving, and the timing reduction.

| Run | Mean ms/step | Median ms/step | Valid labels/s, two GPUs |
|---|---:|---:|---:|
| input 1 | 42.504 | 38.254 | 184,000 |
| BF16 1 | 33.516 | 29.261 | 233,400 |
| BF16 2 | 32.284 | 29.205 | 242,300 |
| input 2 | 42.308 | 38.482 | 184,900 |

Mean **42.406 → 32.900 ms**, timed training throughput **+28.895%**. Repeated
same-precision runs produced identical final export SHA256 values. This is not
a replay of production optimizer/sampler state, an end-to-end throughput result,
or a production 16-GPU MFU measurement.

## Numerical checks

Four real initial batches: maximum absolute loss delta 0.000386, maximum
parameter-gradient relative RMS 1.18%, minimum gradient cosine 0.99993557.
Initial routing indices/counts were exact; gradients remained finite.
An independent per-expert BF16 oracle covers forward/backward and empty experts.
The 2-GPU continuous 6 versus 2 + resume to 6 check was bitwise identical for model,
optimizer, RNG, and numerical metrics. Changing precision on resume was rejected.

After 400 steps, weighted validation NLL was 3.920111 (`input`) versus 3.919762
(BF16); each domain used 256 fixed validation windows. These short runs do not
establish long-run convergence or joint-model quality. BF16 intentionally
changes rounding and training trajectories.

## Profile after the change

An alternating 60-pair microbenchmark using real MoE activations and upstream
gradients measured **12.734 → 4.097 ms** for MoE forward/backward. Trace kernels
confirmed SM90 BF16 Tensor Core GEMMs instead of per-expert FP32 SGEMM fallback.

A separate 80-step two-GPU diagnostic captured steps 61–65 on both ranks. Rank 0:

| Kernel group | Sum ms/step | Fraction of kernel-time sum |
|---|---:|---:|
| Frozen head / CE / input gradient | 13.928 | 56.8% |
| Other model / optimizer | 7.893 | 32.2% |
| NCCL | 2.032 | 8.3% |
| Expert grouped GEMMs | 0.656 | 2.7% |

Kernel-time sums are not wall-clock critical paths. NCCL time may contain waits;
profiler synchronization perturbs execution. Full timing runs had no profiler.
`verification/performance.json` retains the structured aggregate, probe and
per-rank profile data with local paths removed.

## Next experiments

1. Establish long-run BF16 quality, then measure the actual 16-GPU training job.
2. Profile 16-GPU gradient reduction and parameter gathering before enabling
   overlap; both are currently disabled in the verified recipe.
3. Fuse MoE dtype copies, dispatch/combine or SwiGLU only with tests preserving
   FP32 weighted accumulation, stable routing, masked counts and empty experts.
4. Revisit the head only with an exact-objective proposal. Earlier chunk 512–4096
   sweeps had no stable gain; vocabulary changes alter the learning objective.

A fixed-length scatter count prototype saved only about 0.065 ms in a MoE
microbenchmark. It has no verified whole-training benefit and is not included.
